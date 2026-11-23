"""CPU checks of the implementation speed-ups of the fast path (budgetlatent/inference/fast.py) that need no GPU:

  fused projections  fast.fuse_projections: a FLUX 3 block's k, v, q and MLP-input projections as one matmul and its
                     attention and MLP outputs as one -- the same output as the six matmuls (fp32, both FLUX 3 trunks,
                     per-row text windows, the last block's image-only queries), the same parameters and state dict
                     (the Linear weights become views of the fused tensors), back to the six matmuls with enable=False,
                     nothing to fuse in FLUX.2
  matmul library     sampling.blas_key / BLAS_CHOICE: the key the warm-up times is the key the sampler looks up
                     (rows: one per image without guidance, two with it; the text window), and a CPU sample ignores it
  text-mode blocks   FLUX 3 Image Klein Base: run once per sample (txt_cache) -- the same latents and images, bit for
                     bit, as running them at every step, dense and at a budget; the flow map runs them at every step
  switches           fast.add_arguments / settings / overrides: every part of --fast on or off alone, defaults as --fast
  text tokens        FLUX.2's --text_tokens (sampling.text_window): 512 keeps everything; 'auto' / a number keep the
                     first L states of the padded encoding (never a prompt token less), cond and uncond alike, and the
                     model samples with them; FLUX 3 refuses it

    python tests/test_units_fast.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch

from budgetlatent.inference import fast
from budgetlatent.inference import sampling as S
from budgetlatent.models.flux2_trunk import Flux2Trunk
from budgetlatent.models.flux3_trunk import Block, Flux3Config, Flux3Trunk
from budgetlatent.models.budgetpix import BPConfig, BudgetPixFlux2, BudgetPixFlux3
from budgetlatent.tokenization import layout as L
from test_units import OK, TINY, check, randomize
from test_units_flux2_distilled import PixelVAE
from test_units_flux3 import TINY3, inputs3, perturb_norms
from test_units_flux3_distilled import SpyText3

TINY3D = Flux3Config(**{**TINY3.to_dict(), "flow_map": True})


def test_fused_projections():
    print("[fused projections]")
    for name, cfg in (("FLUX 3 Image Klein Base", TINY3), ("FLUX 3 Image Klein (flow map)", TINY3D)):
        gen = torch.Generator().manual_seed(0)
        torch.manual_seed(0)
        trunk = perturb_norms(Flux3Trunk(cfg).eval(), gen)
        z, t, ctx, lens = inputs3(gen=gen)
        kw = dict(txt_len=lens, t_ref=torch.clamp(t + 0.2, max=1.0) if cfg.flow_map else None)
        before = {k: v.clone() for k, v in trunk.state_dict().items()}
        with torch.no_grad():
            plain = trunk.forward_dense(z, t, ctx, **kw)
            n = fast.fuse_projections(trunk)
            fused = trunk.forward_dense(z, t, ctx, **kw)
            fast.fuse_projections(trunk, enable=False)
            again = trunk.forward_dense(z, t, ctx, **kw)
        blocks = [m for m in trunk.modules() if isinstance(m, Block)]
        check(f"{name}: every block fused", n == len(blocks) == 2 * cfg.depth + cfg.depth_single_blocks, f"{n} blocks")
        d = float((fused - plain).abs().max())
        check(f"{name}: == the six matmuls (per-row text windows, image-only queries in the last block)", d < 1e-5,
              f"max |diff| {d:.1e} (output scale {float(plain.abs().max()):.2f})")
        after = trunk.state_dict()
        same = set(after) == set(before) and all(torch.equal(after[k], before[k]) for k in before)
        check(f"{name}: the same parameters and state dict", same)
        b0 = blocks[0]
        views = all(lin.weight.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()
                    for lin, w in ((b0.q_proj, b0._w_in), (b0.mlp_in, b0._w_in), (b0.attn_out, b0._w_out), (b0.mlp_out, b0._w_out)))
        check(f"{name}: the Linear weights are views of the fused tensors (no second copy)", views)
        check(f"{name}: enable=False: the six matmuls again, bit for bit", torch.equal(again, plain) and b0.w_in is None)
    check("FLUX.2: nothing to fuse", fast.fuse_projections(Flux2Trunk(TINY).eval()) == 0)


def test_blas_key():
    print("[matmul library per shape]")

    class Fake:
        family = "flux3"

    cond, uncond = torch.zeros(2, 80, 8), torch.zeros(2, 96, 8)
    guided = S._guided(Fake, uncond, None), S._text_len(Fake, cond, uncond, None)
    check("guided (FLUX 3 Image Klein Base): two rows per image, the longer text window", guided == (True, 96))
    Fake.family = "flux3_distilled"
    check("no guidance (FLUX 3 Image Klein): one row, the prompt's window", (S._guided(Fake, None, None),
                                                                            S._text_len(Fake, cond, None, None)) == (False, 80))
    check("cfg 1 given: no unconditional row", S._guided(type("F2", (), {"family": "flux2"}), uncond, 1.0) is False)
    key = S.blas_key(Fake, S.snap_tokens(0.1), 2, 80)
    check("the key: family, image tokens, rows, text window", key == ("flux3_distilled", S.snap_tokens(0.1), 2, 80), str(key))
    with S.blas_library(None):
        pass
    check("no library given, or no GPU: nothing is switched", True)


def test_text_cache():
    print("[FLUX 3 text-mode blocks once per sample]")
    for fam, cfg, kw in (("flux3", TINY3, dict(cfg=3.5)), ("flux3_distilled", TINY3D, {})):
        gen = torch.Generator().manual_seed(5)
        torch.manual_seed(0)
        model = randomize(BudgetPixFlux3(cfg, BPConfig()).eval(), gen)
        model.family = fam
        text = SpyText3(cfg.context_in_dim)
        cond, uncond, lens = S.encode_prompts(text, model, ["a red fox", "a corgi on the moon"], amp=False)
        vae = PixelVAE(cfg.in_channels, gen)
        noise = torch.randn(2, L.GRID, L.GRID, cfg.in_channels, generator=gen)
        calls = [0]
        hooks = [blk.register_forward_hook(lambda *a: calls.__setitem__(0, calls[0] + 1)) for blk in model.txt_mode_blocks]
        runs = {}
        try:
            with torch.no_grad():
                for cached in (True, False):
                    type(model).caches_text = cached
                    for budget in (1.0, 0.3):
                        calls[0] = 0
                        runs[cached, budget] = (S.sample(model, vae, cond, uncond, budget=budget, noise=noise, amp=False,
                                                         steps=6, **kw, **lens), calls[0])
        finally:
            type(model).caches_text = True
            for h in hooks:
                h.remove()
        depth = cfg.depth
        for budget in (1.0, 0.3):
            (a, n_a), (b, n_b) = runs[True, budget], runs[False, budget]
            same = torch.equal(a["latent"], b["latent"]) and torch.equal(a["image"], b["image"])
            if fam == "flux3":
                check(f"Klein Base at {budget:.0%}: the text-mode blocks run once, the same latent and image bit for bit",
                      same and n_a == depth and n_b == 6 * depth, f"{n_a} text-block calls instead of {n_b}")
            else:
                check(f"Klein (flow map) at {budget:.0%}: t_ref changes the text's time vector, its text-mode blocks run at every step",
                      same and n_a == n_b == 6 * depth, f"{n_a} text-block calls")


def test_switches():
    print("[fast path switches]")
    import argparse

    def parse(*args):
        ap = argparse.ArgumentParser()
        ap.add_argument("--fast", action="store_true")
        ap.add_argument("--compile", action="store_true")
        ap.add_argument("--deterministic", action="store_true")
        ap.add_argument("--final_decode", default=None)
        fast.add_arguments(ap)
        return ap.parse_args(list(args))
    f = fast.settings(parse("--fast"))
    check("--fast: compiled, fused, matmul library by timing, text cache, compiled bf16 previews, NCHW VAE",
          f["compile"] and f["fuse"] and f["blas"] == "timed" and f["text_cache"] and f["preview"] == "bf16_compiled"
          and f["compile_layout"] and not f["channels_last"], str(f))
    p = fast.settings(parse())
    check("plain: nothing compiled or fused, cuBLAS, the text cache still on (exact)",
          not p["compile"] and not p["fuse"] and p["blas"] == "cublas" and p["text_cache"] and p["preview"] == "autocast", str(p))
    d = fast.settings(parse("--fast", "--deterministic"))
    check("--fast --deterministic: no timed library (cuBLAS)", d["blas"] == "cublas" and d["deterministic"])
    o = fast.settings(parse("--fast", "--no_fuse", "--blas", "cublaslt", "--no_text_cache", "--preview_decoder", "bf16",
                            "--no_compile_layout"))
    check("each part off alone", not o["fuse"] and o["blas"] == "cublaslt" and not o["text_cache"] and o["preview"] == "bf16"
          and not o["compile_layout"] and o["compile"], str(o))
    check("--fuse without --fast", fast.settings(parse("--fuse"))["fuse"])
    try:
        fast.settings(parse("--deterministic", "--blas", "timed"))
        ok = False
    except SystemExit:
        ok = True
    check("--deterministic with --blas timed is refused", ok)
    check("bench.py's overrides: only the switches given", fast.overrides(parse("--no_fuse", "--blas", "cublas")) ==
          {"fuse": False, "blas": "cublas"} and fast.overrides(parse("--fast")) == {})


class PaddedText:
    """Qwen3Text's interface: tokenize -> (ids, mask) padded to 512; encode -> (B, 512, dim) states that depend only on
    the prompt and the position (a causal encoder: the first L of them are those of any longer padding)."""

    def __init__(self, dim):
        self.dim = dim

    def tokenize(self, prompts):
        mask = torch.zeros(len(prompts), 512, dtype=torch.long)
        for i, p in enumerate(prompts):
            mask[i, :12 + len(p.split())] = 1                        # the template's 12 tokens + one per word
        return torch.zeros_like(mask), mask

    def encode(self, prompts):
        g = [torch.Generator().manual_seed(len(p)) for p in prompts]
        return torch.stack([torch.randn(512, self.dim, generator=x) for x in g]), None


def test_text_tokens():
    print("[FLUX.2 --text_tokens]")
    gen = torch.Generator().manual_seed(3)
    torch.manual_seed(0)
    model = randomize(BudgetPixFlux2(TINY, BPConfig()).eval(), gen)
    model.family = "flux2"
    text = PaddedText(TINY.context_in_dim)
    prompts = ["a red fox in the snow", "a corgi"]                    # 12 + 6 and 12 + 2 tokens; the empty prompt 12
    full = S.encode_prompts(text, model, prompts, amp=False, uncond=True)
    check("512 (the default): the whole padded window", full[0].shape[1] == 512 and
          S.encode_prompts(text, model, prompts, amp=False, uncond=True, text_tokens="512")[0].shape[1] == 512)
    auto = S.encode_prompts(text, model, prompts, amp=False, uncond=True, text_tokens="auto")
    check("'auto': the longest prompt's 18 tokens, rounded up to 16 -> 32, cond and uncond alike",
          auto[0].shape[1] == 32 and auto[1].shape[1] == 32, str(tuple(auto[0].shape)))
    check("the kept states are the first 32 of the padded encoding", torch.equal(auto[0], full[0][:, :32]) and
          torch.equal(auto[1], full[1][:, :32]))
    check("a number below the longest prompt keeps the prompt (18)",
          S.encode_prompts(text, model, prompts, amp=False, uncond=True, text_tokens="8")[0].shape[1] == 18)
    check("a number above it: that many", S.encode_prompts(text, model, prompts, amp=False, uncond=True,
                                                            text_tokens="100")[0].shape[1] == 100)
    vae = PixelVAE(TINY.in_channels, gen)
    noise = torch.randn(2, L.GRID, L.GRID, TINY.in_channels, generator=gen)
    with torch.no_grad():
        out = S.sample(model, vae, auto[0], auto[1], budget=0.3, noise=noise, amp=False, steps=3, cfg=4.0, **auto[2])
    check("FLUX.2 samples with the cut text (guided, at a budget)", out["image"].shape[0] == 2 and
          bool(torch.isfinite(out["latent"]).all()))
    m3 = BudgetPixFlux3(TINY3, BPConfig()).eval()
    m3.family = "flux3"
    try:
        S.encode_prompts(SpyText3(TINY3.context_in_dim), m3, prompts, amp=False, text_tokens="auto")
        ok = False
    except ValueError:
        ok = True
    check("FLUX 3 refuses --text_tokens (it has its own window)", ok)


if __name__ == "__main__":
    test_fused_projections()
    test_blas_key()
    test_text_cache()
    test_switches()
    test_text_tokens()
    print(f"\nall fast-path checks passed ({len(OK)} checks)")
