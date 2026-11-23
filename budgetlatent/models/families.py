"""The model families BudgetLatent runs on, and where their weights live.

    fam = family_of("flux3")                 # Family: released repo, classes, sampling defaults
    family_from_keys(state_dict_keys)        # "flux2" / "flux3" / "flux3_distilled" / None, from the tensor names alone

  flux2             FLUX.2 [klein] 4B Base (black-forest-labs/FLUX.2-klein-base-4B, Apache-2.0): Qwen3-4B text,
                    50 steps, guidance 4.0
  flux2_distilled   FLUX.2 [klein] 4B (black-forest-labs/FLUX.2-klein-4B, Apache-2.0), BFL's step-distilled release of
                    the same transformer: 4 steps on get_schedule(4, tokens), no guidance (one row through the trunk, no
                    unconditional pass), the layout from the noise at step 0 and once more, at step 1, from a 256 px
                    preview of the decoded estimate
  flux3             FLUX 3 Image Klein Base (black-forest-labs/flux-3-image-klein-base, FLUX Kommunity License):
                    Qwen3-VL-4B text, 50 steps, guidance 3.5, the schedule's constant shift 6.93
  flux3_distilled   FLUX 3 Image Klein (black-forest-labs/flux-3-image-klein, FLUX Kommunity License), BFL's
                    step-distilled release of the same transformer, a flow map: 16 steps on the same shift-6.93
                    schedule, each step told the time it jumps to (t_ref, the trunk's ref_time_in), no guidance (one row
                    through the trunk, no unconditional pass), the layout from the noise at step 0 and once, at step 8, from a 256 px preview

flux2 and flux2_distilled share the trunk, the text encoder, the VAE and every tensor name (Family.arch is "flux2" for
both), so a file's tensors cannot tell them apart: family_from_keys says "flux2", and the family a BudgetLatent file
records in its metadata (weights.checkpoint_family) decides.  flux3 and flux3_distilled share the trunk, the text
encoder and the VAE too (Family.arch "flux3"), but the flow map has two tensors more (ref_time_in), so the names do tell
them apart: a file with ref_time_in is flux3_distilled, one without it never is.

BUDGETLATENT_REPOS names the Hugging Face model repositories of the BudgetLatent weights, one file each
(BUDGETLATENT_FILE), after BFL's names ("klein-base-4B" the base model, "klein-4B" the distilled one; FLUX 3:
"Image-Klein-Base" and "Image-Klein"); the scripts' --ckpt accepts such a repository id as well as a local file
(weights.resolve_checkpoint).
"""
from dataclasses import dataclass, replace

from .budgetpix import BudgetPixFlux2, BudgetPixFlux3
from .convert import is_flux3_cond
from .flux2_trunk import Flux2Trunk, TrunkConfig
from .flux3_trunk import Flux3Config, Flux3Trunk, snapshot_config

# The BudgetLatent weights on the Hugging Face Hub: change the names here.
BUDGETLATENT_REPOS = {
    "flux2": "BudgetAny/BudgetLatent-FLUX.2-klein-base-4B",
    "flux2_distilled": "BudgetAny/BudgetLatent-FLUX.2-klein-4B",
    "flux3": "BudgetAny/BudgetLatent-FLUX-3-Image-Klein-Base",
    "flux3_distilled": "BudgetAny/BudgetLatent-FLUX-3-Image-Klein",
}
BUDGETLATENT_FILE = "budgetlatent.safetensors"

_DIFFUSERS_FLUX2 = "transformer/diffusion_pytorch_model.safetensors"


@dataclass(frozen=True)
class Family:
    name: str
    label: str                          # for galleries and logs
    repo: str                           # Hugging Face repository of the released base model
    arch: str                           # the trunk: "flux2" (FLUX.2 [klein], base or distilled) or "flux3" (FLUX 3 Image
                                        # Klein, base or distilled)
    trunk_cls: type
    model_cls: type
    probe: str                          # a trunk key every full checkpoint of the family has
    trunk_files: tuple                  # the released trunk inside a snapshot, in order of preference
    template: str                       # default chat template (encoders/text.py)
    guidance: float                     # sampling defaults of the release; 1.0: no guidance, no unconditional pass
    steps: int
    relayout_every: int = 4             # the sampler's layout refreshes: steps 0, every, 2 every, ... up to until
    relayout_until: object = 20         # (None: to the last step)
    preview_px: int = 1024              # side of the decoded preview the refreshed layouts are computed from (1024:
                                        # the full image; smaller: the latent average-pooled before the decode)
    flow_map: bool = False              # the trunk takes the time each step jumps to (FLUX 3 Image Klein's ref_time_in)

    @property
    def guided(self) -> bool:
        """Whether the family samples with classifier-free guidance by default (an unconditional row per image)."""
        return self.guidance != 1.0

    def config(self, root=None, cfg=None):
        """The trunk config: FLUX.2's fixed one, or FLUX 3's from the snapshot's config.json, a flow map or not as the
        family says (a snapshot whose config says otherwise is refused: Klein Base is not Klein); cfg overrides."""
        if self.arch == "flux2":
            return TrunkConfig() if cfg is None else TrunkConfig(**cfg)
        shipped = snapshot_config(root) if root else None
        base = Flux3Config() if shipped is None else Flux3Config.from_dict(shipped)
        if shipped is not None and base.flow_map != self.flow_map:
            other = next(f for f in FAMILIES.values() if f.arch == self.arch and f.flow_map == base.flow_map)
            raise SystemExit(f"{root} holds {other.label} ({'a flow map' if base.flow_map else 'not a flow map'}), "
                             f"not {self.label}: use --family {other.name} (or --weights a {self.label} snapshot)")
        base = replace(base, flow_map=self.flow_map)
        return base if cfg is None else Flux3Config(**{**base.to_dict(), **cfg})

    def keeps(self, name, cfg):
        """Whether the trunk this config builds has the released tensor `name` (FLUX 3 without image_cond drops the
        reference-image branch)."""
        return not (self.arch == "flux3" and not cfg.image_cond and is_flux3_cond(name))


# The 50-step models refresh the layout from a 512 px preview (the latent pooled 2x before the decode): on the 32
# validation prompts at 50 / 25 / 10 % its images score as the full 1024 px preview's on ImageReward, PickScore,
# HPS v2.1, MUSIQ, TOPIQ and NIQE (every difference within the prompts' noise, neither side favoured) and are as close
# to the 100 % image (PSNR 17.89 / 18.11 dB at 10 %, FLUX.2 / FLUX 3, against 17.76 / 17.84), and bench.py --fast, batch
# 1, at 10 % takes 2.32 s / 1.69 s instead of 2.56 s / 1.92 s (4.61x / 7.59x instead of 4.19x / 6.69x); 256 px is a
# little faster still (4.69x / 7.84x) with scores a little lower on most metrics (--preview_px 256)
FAMILIES = {
    "flux2": Family("flux2", "FLUX.2 [klein] 4B Base", "black-forest-labs/FLUX.2-klein-base-4B", "flux2", Flux2Trunk,
                    BudgetPixFlux2, "img_in.weight", ("flux-2-klein-base-4b.safetensors", _DIFFUSERS_FLUX2),
                    "bfl_nothink", 4.0, 50, preview_px=512),
    # BFL's flux2 util.py, "flux.2-klein-4b": "defaults": {"guidance": 1.0, "num_steps": 4} (guidance and timesteps
    # distilled into the weights).  With 4 steps the preview decodes cost as much as the trunk saves at small budgets,
    # so the layout is built from the noise at step 0 and rebuilt once, at step 1, from a 256 px preview (the latent
    # pooled 4x before the decode).  On the 16 video prompts at 100 / 50 / 25 / 10 % this is as close (LPIPS) to a
    # refresh at every step from full previews (the training validation's schedule) as the earlier default (steps 0
    # and 2, a full 1024 px preview), and as close to the full-budget image, for a much cheaper preview
    "flux2_distilled": Family("flux2_distilled", "FLUX.2 [klein] 4B", "black-forest-labs/FLUX.2-klein-4B", "flux2",
                              Flux2Trunk, BudgetPixFlux2, "img_in.weight", ("flux-2-klein-4b.safetensors", _DIFFUSERS_FLUX2),
                              "bfl_nothink", 1.0, 4, relayout_every=1, relayout_until=1, preview_px=256),
    # guidance 3.5, 50 steps, shift 6.93: the release's config.json "sampler" (and diffusers 0.42.1's pipeline)
    "flux3": Family("flux3", "FLUX 3 Image Klein Base", "black-forest-labs/flux-3-image-klein-base", "flux3", Flux3Trunk,
                    BudgetPixFlux3, "emb_in.image.weight",
                    ("model.safetensors", "transformer/diffusion_pytorch_model.safetensors.index.json"), "flux3", 3.5, 50,
                    preview_px=512),
    # the release's config.json "sampler": 16 steps, guidance null, alpha 6.93 (diffusers 0.42.1: Flux3DistilledAutoBlocks);
    # BFL's flux3/sampling.py denoise_flow_map: each step passes the time it jumps to and moves by (t_next - t) v.  The
    # layout comes from the noise at step 0 and once more, at step 8, from a 256 px preview (the latent pooled 4x before
    # the decode): on the 16 validation prompts at 50 / 25 / 10 % it is as close to the full-budget image (PSNR / LPIPS
    # 20.42 / 0.236, 18.76 / 0.342, 17.86 / 0.479) as the training validation's refreshes at steps 0, 4, 8 and 12 from
    # full previews (20.52 / 0.240, 18.76 / 0.343, 17.83 / 0.477), with one small preview decode per image instead of
    # three full ones: bench.py --fast, batch 1, 3.53x at 20 % and 4.66x at 10 % (the trained schedule 2.76x and 3.38x);
    # since --fast fuses the projections and times the matmul library per shape: 4.00x and 6.01x (0.52 s and 0.35 s)
    "flux3_distilled": Family("flux3_distilled", "FLUX 3 Image Klein", "black-forest-labs/flux-3-image-klein", "flux3",
                              Flux3Trunk, BudgetPixFlux3, "emb_in.image.weight",
                              ("model.safetensors", "transformer/diffusion_pytorch_model.safetensors.index.json"), "flux3",
                              1.0, 16, relayout_every=8, relayout_until=8, preview_px=256, flow_map=True),
}
DEFAULT_FAMILY = "flux2"


def family_of(name=None) -> Family:
    name = name or DEFAULT_FAMILY
    if name not in FAMILIES:
        raise ValueError(f"model family must be one of {sorted(FAMILIES)}, not {name!r}")
    return FAMILIES[name]


FLOW_MAP_KEYS = ("ref_time_in.in_layer.weight", "target_timestep_embedder.linear_1.weight")   # BFL's, diffusers' name


def family_from_keys(keys):
    """The family of a state dict from its tensor names (this code's, diffusers' or the dit/ export's): "flux2",
    "flux3", or "flux3_distilled" for a FLUX 3 trunk with the flow map's ref_time_in (a file without it is never
    called that); None when the names do not say (a file of added modules only).  FLUX.2 [klein] 4B (flux2_distilled)
    has FLUX.2's names: only a file's metadata tells the two apart (weights.checkpoint_family)."""
    keys = set(keys)
    if any(k in keys for k in FLOW_MAP_KEYS):
        return "flux3_distilled"
    if "emb_in.image.weight" in keys or "x_embedders.image.weight" in keys:
        return "flux3"
    if "img_in.weight" in keys or "x_embedder.weight" in keys or any(k.startswith("dit/") for k in keys):
        return "flux2"
    return None
