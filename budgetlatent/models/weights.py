"""Weights: the released snapshots, BudgetLatent checkpoints, loading the model and the frozen teacher, saving.

    ckpt = resolve_checkpoint("BudgetAny/BudgetLatent-FLUX.2-klein-4B")   # a local file, or a Hub repo id -> its file
    family = checkpoint_family(ckpt)                                       # what the file says, else its tensor names
    model = load_model(ckpt, device="cuda", dtype=torch.bfloat16, family=family)

Checkpoint files: the released single files (FLUX.2 [klein] 4B Base: flux-2-klein-base-4b.safetensors, FLUX.2 [klein]
4B: flux-2-klein-4b.safetensors, FLUX 3 Image Klein Base and Klein: model.safetensors), the diffusers transformers
(FLUX 3's sharded), the dit/ export (FLUX.2), and the trainers' ema-*.safetensors (bf16, this code's names, with
metadata: family, chat_template, resolution, step).  A checkpoint of the added modules only is loaded on top of the
released trunk.
"""
import json
import os
import re

import torch
import torch.nn as nn

from ..tokenization.layout import CELL_PX, GRID
from .budgetpix import FP32_ONLY, BPConfig, add_modules
from .convert import is_adapter, is_flux3_cond, to_bfl, torch_to_dit
from .families import BUDGETLATENT_FILE, DEFAULT_FAMILY, FAMILIES, FLOW_MAP_KEYS, family_from_keys, family_of

BFL_FILE = "flux-2-klein-base-4b.safetensors"           # FLUX.2 [klein] 4B Base's single file (families.Family.trunk_files)


# ---------------------------------------------------------------------------------------------------------
# released snapshots
# ---------------------------------------------------------------------------------------------------------
def klein_dir(repo=None):
    """The local snapshot of a released model (scripts/fetch_weights.py downloads it), or `repo` if it is a path.
    Default: FLUX.2 [klein] 4B Base."""
    repo = repo or family_of(DEFAULT_FAMILY).repo
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception as e:
        raise SystemExit(f"{repo} is not in the Hugging Face cache; run scripts/fetch_weights.py first ({e})")


def trunk_file(root, family=DEFAULT_FAMILY):
    """The released trunk weights inside a snapshot: BFL's single file, else the diffusers transformer (a file, or
    the index of a sharded one)."""
    rels = family_of(family).trunk_files
    for rel in rels:
        if os.path.exists(os.path.join(root, rel)):
            return os.path.join(root, rel)
    raise SystemExit(f"no {family} transformer weights under {root} (run scripts/fetch_weights.py)")


# ---------------------------------------------------------------------------------------------------------
# reading checkpoints
# ---------------------------------------------------------------------------------------------------------
def _load_safetensors(path):
    from safetensors import safe_open
    from safetensors.torch import load_file
    if path.endswith(".index.json"):                                 # a sharded diffusers file
        shards = sorted(set(json.load(open(path))["weight_map"].values()))
        sd = {}
        for s in shards:
            sd.update(load_file(os.path.join(os.path.dirname(path), s)))
        return sd, {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    return load_file(path), meta


_RELEASED = {}          # path -> (state dict, metadata) of the released trunk, while load_model and load_teacher share it
_RELEASED_KEEP = []     # non-empty: keep what is read (keep_released())


def keep_released(on=True):
    """Between keep_released() and keep_released(False), the released trunk file is read once per process."""
    _RELEASED_KEEP.clear()
    if on:
        _RELEASED_KEEP.append(True)
    else:
        _RELEASED.clear()


def released_state(path, keep=False):
    """read_state of the released trunk file, read once per process while keep=True (the trainer builds the student
    and the teacher from the same file: one read instead of two)."""
    if path in _RELEASED:
        return _RELEASED[path]
    out = read_state(path)
    if keep:
        _RELEASED[path] = out
    return out


def read_state(path):
    """A checkpoint file -> (state dict in this code's names, metadata).

    Accepts the released single files, a diffusers transformer file (or its shard index), the dit/ export (FLUX.2)
    and the trainers' `ema-*.safetensors`; also a torch file holding {"model": ..., "ema": ...} (the EMA is used)."""
    if path.endswith((".safetensors", ".index.json")):
        sd, meta = _load_safetensors(path)
        return to_bfl(sd), meta
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = dict(ck["model"])
    sd.update(ck.get("ema", {}))
    return sd, ck.get("cfg") or {}


def checkpoint_meta(path):
    """The metadata of a safetensors file ({} for anything else)."""
    if not path or not path.endswith(".safetensors"):
        return {}
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return f.metadata() or {}


def _names(path):
    """The tensor names of a safetensors file (its header only)."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return list(f.keys())


def checkpoint_family(path, default=DEFAULT_FAMILY):
    """The family of a checkpoint: what its metadata records (the trainers write it; the only way to tell
    flux2_distilled from flux2, whose tensors have the same names), else what its tensor names say (the file's header
    only), else `default` (also for no path).  A FLUX 3 file recorded as "flux3" whose tensors include the flow map's
    ref_time_in is flux3_distilled (the trainer of a FLUX 3 Image Klein run records the trunk, "flux3"); one without
    ref_time_in is never flux3_distilled by its names."""
    meta = checkpoint_meta(path)
    recorded = meta.get("family") or ("flux3" if str(meta.get("format", "")).startswith("budgetpix_flux3") else None)
    if recorded == "flux3":
        return "flux3_distilled" if any(k in FLOW_MAP_KEYS for k in _names(path)) else "flux3"
    if recorded:
        return recorded
    if path and path.endswith(".safetensors"):
        found = family_from_keys(_names(path))
        if found:
            return found
    return default


def checkpoint_template(path, default=None, family=DEFAULT_FAMILY):
    """The chat template a checkpoint was trained with (the trainers record it in the file's metadata); the released
    weights, and files that do not say, get the family's own."""
    from ..encoders.text import template_name
    name = checkpoint_meta(path).get("chat_template")
    return template_name(name) if name else (default or family_of(family).template)


_REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def resolve_checkpoint(ckpt):
    """--ckpt: a local file as it is, or a Hugging Face model repository id (e.g. families.BUDGETLATENT_REPOS) ->
    its BUDGETLATENT_FILE, downloaded into the Hugging Face cache.  None stays None (the released base model)."""
    if not ckpt or os.path.exists(ckpt):
        return ckpt
    if ckpt.endswith((".safetensors", ".pt", ".pth")) or not _REPO_ID.match(ckpt):
        raise SystemExit(f"--ckpt {ckpt}: no such file (and not a Hugging Face repository id)")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=ckpt, filename=BUDGETLATENT_FILE)


# ---------------------------------------------------------------------------------------------------------
# building the model
# ---------------------------------------------------------------------------------------------------------
def load_model(ckpt=None, weights=None, device="cpu", dtype=None, bp: BPConfig = None, family=DEFAULT_FAMILY,
               trunk_cfg=None):
    """BudgetPix from the released trunk (`weights`: a snapshot dir / repo id, default the HF cache) and, if `ckpt`
    is given, a BudgetPix checkpoint on top of it or instead of it.  With no ckpt the added modules are at their zero
    initialisation.  dtype=torch.bfloat16 stores the weights in bf16 (run under autocast).  trunk_cfg: overrides of
    the trunk config (FLUX 3: e.g. {"image_cond": True})."""
    fam = family_of(family)
    bp = bp or BPConfig()
    root = klein_dir(weights or fam.repo)
    cfg = fam.config(root, trunk_cfg)
    with torch.device("meta"):                                       # no memory until the weights are assigned
        model = fam.model_cls(cfg, bp)
    names = list(model.state_dict())
    src = ckpt or trunk_file(root, fam.name)
    sd, meta = read_state(src) if ckpt else released_state(src, keep=bool(_RELEASED_KEEP))
    sd = dict(sd)
    if isinstance(meta, dict) and meta.get("family") and meta["family"] != fam.name:
        known = meta["family"] in FAMILIES
        if not known or FAMILIES[meta["family"]].arch != fam.arch:
            raise SystemExit(f"{src} is a {meta['family']} checkpoint, not {fam.name}")
        # (same trunk: e.g. a FLUX.2 [klein] 4B run whose trainer recorded "flux2"; the caller's family is sampled)
        print(f"[model] note: {os.path.basename(src)} records family {meta['family']}; loading it as {fam.name}", flush=True)
    res = meta.get("resolution") if isinstance(meta, dict) else None
    if res and str(res).isdigit() and int(res) != GRID * CELL_PX:
        print(f"[model] note: {os.path.basename(src)} was trained at {res} x {res}; this process works on a "
              f"{GRID * CELL_PX} x {GRID * CELL_PX} grid (BP_GRID={GRID})", flush=True)
    has_added = any(is_adapter(k) for k in sd)
    if fam.probe not in sd:                                          # added modules only: take the released trunk
        sd = {**released_state(trunk_file(root, fam.name), keep=bool(_RELEASED_KEEP))[0], **sd}
    sd = {k: v for k, v in sd.items() if fam.keeps(k, cfg)}
    if fam.arch == "flux3" and cfg.image_cond and fam.probe in sd and any(is_flux3_cond(k) and k not in sd for k in names):
        # a BudgetLatent checkpoint holds the text-to-image trunk it trained; the reference-image branch (editing) is
        # the released one, as BudgetLatent never trains it
        rel = released_state(trunk_file(root, fam.name), keep=bool(_RELEASED_KEEP))[0]
        sd.update({k: v for k, v in rel.items() if is_flux3_cond(k) and k in names and k not in sd})
    missing, unexpected = [k for k in names if k not in sd], [k for k in sd if k not in names]
    if fam.arch == "flux3" and (unexpected or missing):              # FLUX 3 Image Klein Base vs Klein (the flow map)
        is_ref = lambda k: k.startswith("ref_time_in.")
        if unexpected and all(is_ref(k) for k in unexpected):
            raise SystemExit(f"{src} is a flow map (FLUX 3 Image Klein, ref_time_in), not {fam.label}: use --family flux3_distilled")
        if any(is_ref(k) for k in missing) and not any(is_ref(k) for k in sd) and fam.probe in sd:
            raise SystemExit(f"{src} has no ref_time_in: FLUX 3 Image Klein Base, not the flow map {fam.label}: use --family flux3")
    if unexpected:
        raise SystemExit(f"{src}: {len(unexpected)} unexpected keys, e.g. {unexpected[:4]}")
    if any(not is_adapter(k) for k in missing):
        raise SystemExit(f"{src}: trunk weights missing, e.g. {[k for k in missing if not is_adapter(k)][:4]}")
    if missing and has_added:
        raise SystemExit(f"{src}: {len(missing)} BudgetPix weights missing, e.g. {missing[:4]}")
    if missing:                                                      # no checkpoint: the zero initialisation
        sd.update(add_modules(nn.Module(), cfg, bp).state_dict())
    target = dtype or torch.float32
    sd = {k: sd[k].detach().to(device, torch.float32 if k.startswith(FP32_ONLY) else target) for k in names}
    model.load_state_dict(sd, strict=True, assign=True)
    model.family = fam.name                                          # the sampler's defaults (flux2 and flux2_distilled share the class)
    return model.requires_grad_(True)


def load_teacher(weights=None, device="cpu", dtype=torch.bfloat16, family=DEFAULT_FAMILY, trunk_cfg=None):
    """The released trunk, frozen."""
    fam = family_of(family)
    root = klein_dir(weights or fam.repo)
    cfg = fam.config(root, trunk_cfg)
    with torch.device("meta"):
        teacher = fam.trunk_cls(cfg)
    sd = released_state(trunk_file(root, fam.name), keep=bool(_RELEASED_KEEP))[0]
    sd = {k: v for k, v in sd.items() if fam.keeps(k, cfg)}
    teacher.load_state_dict({k: v.to(device, dtype or torch.float32) for k, v in sd.items()}, strict=True, assign=True)
    return teacher.eval().requires_grad_(False)


def expected_counts(family=DEFAULT_FAMILY, root=None, bp: BPConfig = None, trunk_cfg=None):
    """(released trunk parameters the model loads, added parameters), from the released file itself (its header)
    and the added modules' definition: what trainlib.preflight checks the built model against."""
    fam = family_of(family)
    bp = bp or BPConfig()
    if fam.arch == "flux2":
        return 3_875_544_576, 28_136_192
    import math
    import struct
    root = klein_dir(root or fam.repo)
    cfg = fam.config(root, trunk_cfg)
    path = trunk_file(root, fam.name)
    if path.endswith(".index.json"):
        files = [os.path.join(os.path.dirname(path), s) for s in sorted(set(json.load(open(path))["weight_map"].values()))]
    else:
        files = [path]
    head = {}
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head.update({k: v for k, v in json.loads(fh.read(n)).items() if k != "__metadata__"})
    if path.endswith(".index.json"):                                 # diffusers names -> this code's
        from .convert import flux3_key
        name = lambda k: "final_layer.image.adaLN_modulation.1.weight" if k == "norm_out.linear.weight" else flux3_key(k)
    else:
        name = lambda k: k
    trunk = sum(math.prod(v["shape"]) for k, v in head.items() if fam.keeps(name(k), cfg))
    with torch.device("meta"):
        added = sum(p.numel() for p in add_modules(nn.Module(), cfg, bp).parameters())
    return trunk, added


# ---------------------------------------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------------------------------------
def save_weights(state: dict, path: str, dtype=torch.bfloat16, layout="bfl", metadata=None):
    """Write a state dict as one safetensors file: this code's names (layout="bfl") or the dit/ export
    (layout="dit", kernel-layout names; FLUX.2 only)."""
    from safetensors.torch import save_file
    sd = {k: v.detach().to("cpu", dtype if v.is_floating_point() else v.dtype).contiguous() for k, v in state.items()}
    if layout == "dit":
        sd = torch_to_dit(sd)
    save_file(sd, path + ".tmp", metadata={k: str(v) for k, v in (metadata or {}).items()})
    os.replace(path + ".tmp", path)
