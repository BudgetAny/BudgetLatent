"""The frozen text encoders: Qwen3-4B for FLUX.2 [klein] (Qwen3Text), Qwen3-VL-4B-Instruct for FLUX 3 (Qwen3VLText).

FLUX.2 [klein]: prompts -> (B, 512, 7680).

The prompt is wrapped in Qwen3's chat template, padded to 512 tokens, and the hidden states after layers 9, 18 and
27 are concatenated.  Only the first 27 of the 36 layers are needed, so the rest (and the final norm and the LM
head) are dropped; the outputs are unchanged.

Two templates (model.chat_template in configs/flux2_*.yaml):

  "bfl_nothink"  <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
                 what BFL's and diffusers' FLUX.2 [klein] pipelines produce (apply_chat_template(...,
                 enable_thinking=False)), hence what the released weights were trained with; the empty prompt is
                 12 tokens.  The default.
  "no_think"     the same without the think block (the empty prompt is 8 tokens): only to continue a run that was
                 started without it.  The released model's velocity moves by ~6 % rms between the two.

Qwen3 is causal, so the two differ only in the positions after the prompt (4 extra tokens and the padding states).
Teacher, student, the empty prompt of the caption dropout, stage 2 and sampling must all use the same one;
scripts/train/train.py records it in the weights it writes and the sampling scripts read it back.
"""
import os

import torch
import torch.nn as nn

PAD_ID = 151643                         # <|endoftext|>
MAX_LENGTH = 512
OUT_LAYERS = (9, 18, 27)
THINK = "<think>\n\n</think>\n\n"
TEMPLATES = ("bfl_nothink", "no_think", "flux3")
DEFAULT_TEMPLATE = "bfl_nothink"
EMPTY_PROMPT_IDS = {                    # the unconditional prompt, before padding
    "bfl_nothink": (151644, 872, 198, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271),
    "no_think": (151644, 872, 198, 151645, 198, 151644, 77091, 198),
}
_ALIASES = {"official": "bfl_nothink", "spec": "no_think"}      # older names of the two templates


def template_name(template=None) -> str:
    name = _ALIASES.get(template, template) or DEFAULT_TEMPLATE
    if name not in TEMPLATES:
        raise ValueError(f"chat template must be one of {TEMPLATES}, not {template!r}")
    return name


def wrap(prompt: str, template: str = DEFAULT_TEMPLATE) -> str:
    """Whitespace-collapsed prompt in the chat template (FLUX 3: the prompt as it is, see wrap_flux3)."""
    if template_name(template) == "flux3":
        return wrap_flux3(prompt)
    text = "<|im_start|>user\n" + " ".join(str(prompt).split()) + "<|im_end|>\n<|im_start|>assistant\n"
    return text + (THINK if template_name(template) == "bfl_nothink" else "")


def load_tokenizer(path):
    """The tokenizer of a snapshot: its tokenizer/ (FLUX.2), else text_encoder/ (FLUX 3), else the path itself."""
    from transformers import AutoTokenizer
    sub = path
    for rel in ("tokenizer", "text_encoder"):
        if os.path.exists(os.path.join(path, rel, "tokenizer_config.json")):
            sub = os.path.join(path, rel)
            break
    tok = AutoTokenizer.from_pretrained(sub)
    tok.padding_side = "right"
    return tok


def tokenize(tokenizer, prompts, template=DEFAULT_TEMPLATE, max_length=MAX_LENGTH):
    """-> (input_ids (B, 512) long, attention_mask (B, 512) long), right-padded with <|endoftext|>."""
    enc = tokenizer([wrap(p, template) for p in prompts], padding="max_length", truncation=True,
                    max_length=max_length, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"].long()
    mask = (ids != PAD_ID).long()
    mask[:, 0] = 1
    return ids, mask


class Qwen3Text(nn.Module):
    family = "flux2"

    def __init__(self, path, device="cpu", dtype=torch.bfloat16, template=DEFAULT_TEMPLATE, truncate=True):
        """path: a snapshot of black-forest-labs/FLUX.2-klein-base-4B (text_encoder/ and tokenizer/), or of
        Qwen/Qwen3-4B."""
        super().__init__()
        from transformers import AutoModelForCausalLM
        sub = os.path.join(path, "text_encoder") if os.path.isdir(os.path.join(path, "text_encoder")) else path
        try:
            lm = AutoModelForCausalLM.from_pretrained(sub, dtype=dtype)
        except TypeError:                                            # transformers < 4.56 calls it torch_dtype
            lm = AutoModelForCausalLM.from_pretrained(sub, torch_dtype=dtype)
        self.model = lm.model                                        # the decoder stack without the LM head
        if truncate:
            self.model.layers = self.model.layers[: max(OUT_LAYERS)]
            self.model.config.num_hidden_layers = len(self.model.layers)
            self.model.norm = nn.Identity()                          # hidden_states[-1] stays the raw layer output
        self.truncated = truncate
        self.tokenizer = load_tokenizer(path)
        self.template = template_name(template)
        self.model.to(device).eval().requires_grad_(False)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def tokenize(self, prompts):
        return tokenize(self.tokenizer, prompts, self.template)

    @torch.no_grad()
    def encode_ids(self, ids, mask):
        out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                         output_hidden_states=True, use_cache=False)
        return torch.cat([out.hidden_states[k] for k in OUT_LAYERS], dim=-1)

    @torch.no_grad()
    def forward(self, prompts):
        return self.encode_ids(*self.tokenize(prompts))

    @torch.no_grad()
    def encode(self, prompts):
        """-> (states, None): FLUX.2's window is the whole 512 tokens."""
        return self.forward(prompts), None

    def lengths(self, mask):
        return None

    def trim(self, ids, mask):
        return ids, mask

    def empty_ids(self, batch=1, pad_to=None):
        ids, mask = self.tokenize([""])
        return ids.expand(batch, -1).clone(), mask.expand(batch, -1).clone()


# ---------------------------------------------------------------------------------------------------------
# FLUX 3: Qwen3-VL-4B-Instruct, eight layers stacked, a text window per prompt
# ---------------------------------------------------------------------------------------------------------
"""FLUX 3 Image Klein's text features, as the release builds them (diffusers 0.42.1 Flux3TextEncoderStep and BFL's
flux3/text_encoder.py, which agree):

  * the prompt as one user message in Qwen3-VL's chat template with the generation prompt, no system message, no
    thinking block: "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" (the empty prompt: 8 tokens);
  * right-padded with <|endoftext|> to the window: the token count rounded up to a multiple of 80 (capped);
  * Qwen3-VL's language model with the padding masked inside it, position ids 0 .. n-1 for the real tokens and 1
    for every padding token (what transformers 4.57 computes, which the release pins explicitly);
  * the hidden states after layers 4, 8, ..., 32, concatenated per token: (B, window, 8 x 2560).
The transformer then attends to the whole window, padding included, so the window length is part of the input:
batches of prompts are padded to the longest window and each row carries its own (`lengths`), which the trunk
masks.  Only the first 32 of the 36 layers are needed (the vision tower, the last layers, the final norm and the
LM head are dropped; the outputs are unchanged).  With the padding masked and every padding token at position 1,
all padding tokens of a prompt have the same states, so padding beyond a row's window changes nothing in it."""
F3_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
F3_PAD_MULTIPLE = 80
F3_MAX_TOKENS = 8192                    # the release's cap (diffusers); configs/flux3_*.yaml model.text_max_tokens for training
F3_EMPTY_PROMPT_IDS = (151644, 872, 198, 151645, 198, 151644, 77091, 198)


def wrap_flux3(prompt: str) -> str:
    return "<|im_start|>user\n" + str(prompt) + "<|im_end|>\n<|im_start|>assistant\n"


def window(n_tokens, max_tokens=F3_MAX_TOKENS):
    """The text window of a prompt of n_tokens tokens (tensor or int): rounded up to 80, capped at max_tokens."""
    if torch.is_tensor(n_tokens):
        return torch.clamp(torch.div(n_tokens + F3_PAD_MULTIPLE - 1, F3_PAD_MULTIPLE, rounding_mode="floor") * F3_PAD_MULTIPLE,
                           max=max_tokens)
    return min(-(-int(n_tokens) // F3_PAD_MULTIPLE) * F3_PAD_MULTIPLE, max_tokens)


def tokenize_flux3(tokenizer, prompts, max_tokens=F3_MAX_TOKENS, pad_to=None):
    """-> (input_ids (B, L) long, attention_mask (B, L) long), right-padded with <|endoftext|>.  L = pad_to (a fixed
    length, for stacking in a data loader) or the longest window of the batch.  Truncation at max_tokens."""
    enc = tokenizer([wrap_flux3(p) for p in prompts], padding=False, truncation=True, max_length=max_tokens,
                    add_special_tokens=False)
    real = [len(x) for x in enc["input_ids"]]
    length = pad_to or max(window(n, max_tokens) for n in real)
    ids = torch.full((len(prompts), length), PAD_ID, dtype=torch.long)
    mask = torch.zeros(len(prompts), length, dtype=torch.long)
    for i, x in enumerate(enc["input_ids"]):
        ids[i, :len(x)] = torch.tensor(x[:length])
        mask[i, :len(x)] = 1
    return ids, mask


class Qwen3VLText(nn.Module):
    """FLUX 3's text encoder.  encode(prompts) -> (states (B, L, 20480), windows (B,) long)."""
    family = "flux3"

    def __init__(self, path, device="cpu", dtype=torch.bfloat16, template="flux3", truncate=True, max_tokens=F3_MAX_TOKENS):
        """path: a snapshot of black-forest-labs/flux-3-image-klein-base or flux-3-image-klein (its text_encoder/,
        the same files in both), or of Qwen/Qwen3-VL-4B-Instruct."""
        super().__init__()
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        if template_name(template) != "flux3":
            raise ValueError(f"FLUX 3 has one chat template ('flux3'), not {template!r}")
        sub = os.path.join(path, "text_encoder") if os.path.isdir(os.path.join(path, "text_encoder")) else path
        try:
            full = Qwen3VLForConditionalGeneration.from_pretrained(sub, dtype=dtype)
        except TypeError:
            full = Qwen3VLForConditionalGeneration.from_pretrained(sub, torch_dtype=dtype)
        self.model = full.model.language_model                       # the decoder stack: no vision tower, no LM head
        del full
        if truncate:
            self.model.layers = self.model.layers[: max(F3_LAYERS)]
            self.model.config.num_hidden_layers = len(self.model.layers)
            self.model.norm = nn.Identity()                          # hidden_states[-1] stays the raw layer output
        self.truncated = truncate
        self.tokenizer = AutoTokenizer.from_pretrained(sub)
        self.tokenizer.padding_side = "right"
        self.template, self.max_tokens = "flux3", int(max_tokens)
        self.model.to(device).eval().requires_grad_(False)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def tokenize(self, prompts, pad_to=None):
        return tokenize_flux3(self.tokenizer, prompts, self.max_tokens, pad_to)

    def lengths(self, mask):
        """The text window of every row, (B,) long, on mask's device."""
        return window(mask.long().sum(dim=1), self.max_tokens)

    def trim(self, ids, mask):
        """Drop the columns beyond the longest window (host synchronisation: one integer)."""
        n = int(self.lengths(mask).max())
        return ids[:, :n], mask[:, :n]

    @torch.no_grad()
    def encode_ids(self, ids, mask):
        """ids / mask (B, L) -> (B, L, 8 x 2560).  mask may also be the ready (B, 1, L, L) causal + padding mask
        (training/stage1.text_mask_4d); the position ids are then taken from its diagonal."""
        ids = ids.to(self.device)
        flat = mask.to(self.device)
        if flat.dim() == 4:
            flat = torch.diagonal(flat[:, 0], dim1=1, dim2=2)
        flat = flat.long()
        pos = torch.where(flat.bool(), flat.cumsum(dim=1) - 1, torch.ones_like(flat))
        out = self.model(input_ids=ids, attention_mask=mask.to(self.device), position_ids=pos,
                         output_hidden_states=True, use_cache=False)
        return torch.cat([out.hidden_states[k] for k in F3_LAYERS], dim=-1)

    @torch.no_grad()
    def encode(self, prompts):
        ids, mask = self.tokenize(prompts)
        return self.encode_ids(ids, mask), self.lengths(mask).to(self.device)

    def forward(self, prompts):
        return self.encode(prompts)[0]

    def empty_ids(self, batch=1, pad_to=None):
        ids, mask = self.tokenize([""], pad_to=pad_to)
        return ids.expand(batch, -1).clone(), mask.expand(batch, -1).clone()


FLUX3_FAMILIES = ("flux3", "flux3_distilled")     # FLUX 3 Image Klein Base and Klein: the same Qwen3-VL-4B files


def text_encoder(family, path, device="cpu", dtype=torch.bfloat16, template=None, max_tokens=None):
    """The family's text encoder (FLUX.2: Qwen3Text; FLUX 3 Image Klein Base and Klein: Qwen3VLText)."""
    if family in FLUX3_FAMILIES:
        return Qwen3VLText(path, device, dtype, template or "flux3", max_tokens=max_tokens or F3_MAX_TOKENS)
    return Qwen3Text(path, device, dtype, template=template or DEFAULT_TEMPLATE)
