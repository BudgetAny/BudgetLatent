"""CPU checks of FLUX.2 [klein] 4B, BFL's step-distilled release ("flux2_distilled"), that need no weights (a tiny
random FLUX.2 trunk stands in for it: the distilled release has FLUX.2 [klein] 4B Base's architecture and names).

    python tests/test_units_flux2_distilled.py
    FLUX2_ROOT=/path/to/black-forest-labs/flux2 python tests/test_units_flux2_distilled.py   # + BFL's get_schedule itself

  family     the family table: the release's repository, the shared FLUX.2 trunk and classes, the sampler defaults
             (4 steps, guidance 1.0, layouts at steps 0 and 1, the second from a 256 px preview), the BudgetLatent
             repositories of every family
  detection  a file's metadata says flux2_distilled, its tensor names only "flux2"; load_model gives the model its
             family, takes a FLUX.2 file recorded as "flux2" and refuses a FLUX 3 one
  schedule   schedule(4, "flux2_distilled") = 1 - BFL's get_schedule(4, tokens); at 1024 x 1024 get_schedule(4, 4096)
  sampler    the no-guidance path: no empty prompt encoded, no fused batch, one row per image through the trunk, the
             layout from the noise at step 0 and from a decoded preview at step 1 (the latent pooled to a 256 px
             decode, upsampled for the entropy; a refresh at every step on request); the defaults equal the explicit settings; at 100 % it is plain unguided Euler; fast.warmup and the validation suite take the
             same path
"""
import ast
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch

from budgetlatent.inference import sampling as S
from budgetlatent.models import families as F
from budgetlatent.models.budgetpix import BPConfig, BudgetPixFlux2, BudgetPixFlux3
from budgetlatent.models.flux2_trunk import Flux2Trunk
from budgetlatent.tokenization import layout as L
from test_units import OK, TINY, StubText, check, randomize, tiny_inputs

DISTILLED = "flux2_distilled"
# get_schedule(4, 4096) of BFL's flux2/sampling.py as sigmas (1 = noise): the 1024 x 1024 times of the distilled run
SIGMAS_4096 = (1.0, 0.967384, 0.908144, 0.767200, 0.0)


# ---------------------------------------------------------------------------------------------------------
def test_family():
    print("[family]")
    fam, base, f3 = F.family_of(DISTILLED), F.family_of("flux2"), F.family_of("flux3")
    check("table", fam.name == DISTILLED and fam.label == "FLUX.2 [klein] 4B" and fam.repo == "black-forest-labs/FLUX.2-klein-4B"
          and fam.arch == "flux2" and fam.trunk_cls is Flux2Trunk and fam.model_cls is BudgetPixFlux2 and fam.probe == "img_in.weight"
          and fam.template == "bfl_nothink", f"{fam.label} ({fam.repo}): FLUX.2's trunk, classes and chat template")
    check("sampler defaults", (fam.steps, fam.guidance, fam.guided, fam.relayout_every, fam.relayout_until, fam.preview_px)
          == (4, 1.0, False, 1, 1, 256), "4 steps, guidance 1.0 (no unconditional pass), layouts at steps 0 and 1, a 256 px preview")
    check("the other families unchanged", (base.steps, base.guidance, base.relayout_every, base.relayout_until, base.arch,
                                           base.preview_px) == (50, 4.0, 4, 20, "flux2", 512)
          and (f3.steps, f3.guidance, f3.relayout_every, f3.relayout_until, f3.arch, f3.preview_px) == (50, 3.5, 4, 20, "flux3", 512)
          and base.repo == "black-forest-labs/FLUX.2-klein-base-4B" and f3.model_cls is BudgetPixFlux3)
    check("trunk config and tensors", fam.config() == base.config() and fam.keeps("img_in.weight", fam.config())
          and fam.trunk_files[0] == "flux-2-klein-4b.safetensors" and base.trunk_files[0] == "flux-2-klein-base-4b.safetensors",
          "the base model's TrunkConfig; BFL's single file flux-2-klein-4b.safetensors")
    check("BudgetLatent repositories", {k: F.BUDGETLATENT_REPOS[k] for k in ("flux2", "flux2_distilled", "flux3")} == {
        "flux2": "BudgetAny/BudgetLatent-FLUX.2-klein-base-4B", "flux2_distilled": "BudgetAny/BudgetLatent-FLUX.2-klein-4B",
        "flux3": "BudgetAny/BudgetLatent-FLUX-3-Image-Klein-Base"} and set(F.BUDGETLATENT_REPOS) == set(F.FAMILIES),
          "BFL's naming: klein-base-4B the base model, klein-4B the distilled one")
    from budgetlatent.models.weights import expected_counts, trunk_file
    with tempfile.TemporaryDirectory() as tmp:
        for rel in ("flux-2-klein-base-4b.safetensors", "flux-2-klein-4b.safetensors"):
            open(os.path.join(tmp, rel), "wb").close()
        picked = os.path.basename(trunk_file(tmp, DISTILLED)), os.path.basename(trunk_file(tmp, "flux2"))
    check("released trunk file", picked == ("flux-2-klein-4b.safetensors", "flux-2-klein-base-4b.safetensors"), str(picked))
    check("parameter counts", expected_counts(DISTILLED) == expected_counts("flux2") == (3_875_544_576, 28_136_192))
    try:
        F.family_of("flux2-distilled")
        refused = False
    except ValueError:
        refused = True
    check("unknown family refused", refused)


# ---------------------------------------------------------------------------------------------------------
def tiny_file(path, family=None, keys="flux2"):
    """A full tiny BudgetPix FLUX.2 checkpoint (trunk + added modules) with optional metadata."""
    from safetensors.torch import save_file
    torch.manual_seed(0)
    model = BudgetPixFlux2(TINY, BPConfig())
    sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    if keys == "flux3":
        sd = {"emb_in.image.weight": torch.zeros(2, 2)}
    meta = {"family": family, "chat_template": "bfl_nothink", "format": "budgetpix_flux2_s8"} if family else None
    save_file(sd, path, metadata=meta)
    return model


def test_detection():
    print("[detection]")
    from budgetlatent.models.weights import checkpoint_family, checkpoint_template, load_model
    with tempfile.TemporaryDirectory() as tmp:
        p_dist, p_base, p_keys, p_f3 = (os.path.join(tmp, f"{n}.safetensors") for n in ("dist", "base", "keys", "f3meta"))
        ref = tiny_file(p_dist, DISTILLED)
        tiny_file(p_base, "flux2")
        tiny_file(p_keys, None)
        tiny_file(p_f3, "flux3")
        from safetensors import safe_open
        with safe_open(p_dist, "pt") as f:
            names = list(f.keys())
        check("tensor names say flux2", F.family_from_keys(names) == "flux2" and checkpoint_family(p_keys) == "flux2",
              "family_from_keys cannot tell the distilled release from the base model")
        check("metadata decides", checkpoint_family(p_dist) == DISTILLED and checkpoint_family(p_base) == "flux2"
              and checkpoint_family(None) == "flux2" and checkpoint_template(p_dist, family=DISTILLED) == "bfl_nothink",
              "family: flux2_distilled in the header overrides the names")
        cfg = asdict(TINY)
        model = load_model(p_dist, weights=tmp, family=checkpoint_family(p_dist), trunk_cfg=cfg)
        d = max(float((a.float() - b.float().to(torch.bfloat16).float()).abs().max())
                for a, b in zip(model.state_dict().values(), ref.state_dict().values()))
        check("load_model gives the model its family", model.family == DISTILLED and type(model) is BudgetPixFlux2
              and S.sampler_defaults(model) == (4, 1.0, 1, 1) and S.preview_side(model) == 256 and d == 0.0,
              f"model.family {model.family!r}; the sampler's defaults {S.sampler_defaults(model)}")
        base = load_model(p_base, weights=tmp, family="flux2", trunk_cfg=cfg)
        check("the base model keeps its own", base.family == "flux2" and S.sampler_defaults(base) == (50, 4.0, 4, 20))
        again = load_model(p_base, weights=tmp, family=DISTILLED, trunk_cfg=cfg)
        check("a FLUX.2 file recorded as flux2 loads as flux2_distilled when asked", again.family == DISTILLED,
              "(the trainers of the distilled run wrote family: flux2; same trunk)")
        try:
            load_model(p_f3, weights=tmp, family=DISTILLED, trunk_cfg=cfg)
            msg = ""
        except SystemExit as e:
            msg = str(e)
        check("a FLUX 3 file is refused", "is a flux3 checkpoint, not flux2_distilled" in msg, msg[-60:])


# ---------------------------------------------------------------------------------------------------------
def bfl_get_schedule():
    """BFL's get_schedule: from FLUX2_ROOT's src/flux2/sampling.py (its three functions, exec'd as written), else the
    same code (below: copied from github.com/black-forest-labs/flux2, src/flux2/sampling.py, without the type hints and
    blank lines; Copyright Black Forest Labs, Apache License 2.0, see LICENSE)."""
    root = os.environ.get("FLUX2_ROOT") or os.path.join(ROOT, "third_party", "flux2")
    path = os.path.join(root, "src", "flux2", "sampling.py")
    names = ("generalized_time_snr_shift", "get_schedule", "compute_empirical_mu")
    if os.path.exists(path):
        src = open(path).read()
        tree = ast.parse(src)
        code = "\n\n".join(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)
        where = path
    else:
        code = '''
def generalized_time_snr_shift(t, mu, sigma):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(num_steps, image_seq_len):
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def compute_empirical_mu(image_seq_len, num_steps):
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b
    return float(mu)
'''
        where = "a verbatim copy of BFL's flux2/sampling.py"
    ns = {"math": math, "torch": torch, "Tensor": torch.Tensor}
    exec(compile(code, where, "exec"), ns)
    return ns["get_schedule"], where


def test_schedule():
    print("[schedule]")
    get_schedule, where = bfl_get_schedule()
    for tokens in sorted({L.N_DENSE, 1024, 4096}):
        ours = L.flux2_klein_timesteps(4, tokens)
        bfl = [1.0 - s for s in get_schedule(4, tokens)]
        d = max(abs(a - b) for a, b in zip(ours, bfl))
        check(f"4 steps at {tokens} tokens == 1 - BFL get_schedule(4, {tokens})", len(ours) == 5 and d < 1e-6,
              f"max |diff| {d:.1e} (BFL computes in float32; {where})")
    ts = L.schedule(4, DISTILLED)
    check("schedule(4, flux2_distilled) is FLUX.2's", ts == L.flux2_klein_timesteps(4, L.N_DENSE) and ts[0] == 0.0 and ts[-1] == 1.0,
          f"t = {[round(v, 4) for v in ts]} at {L.N_DENSE} tokens")
    t4096 = L.flux2_klein_timesteps(4, 4096)
    d = max(abs((1.0 - a) - b) for a, b in zip(t4096, SIGMAS_4096))
    check("get_schedule(4, 4096): the times of the distilled run", d < 1e-6 and abs(L.compute_empirical_mu(4096, 4) - 2.29118) < 1e-5,
          f"t = {[round(v, 4) for v in t4096]} (config: 0, 0.0326, 0.0919, 0.2328), mu {L.compute_empirical_mu(4096, 4):.5f}")


# ---------------------------------------------------------------------------------------------------------
class SpyText(StubText):
    def __init__(self, dim):
        super().__init__(dim)
        self.seen = []

    def encode(self, prompts):
        self.seen.extend(prompts)
        return super().encode(prompts)


class PixelVAE:
    """A fixed map from cells to 16 x 16 px (previews and the final decode), counting its calls."""

    def __init__(self, c, gen):
        self.w, self.calls, self.downscales = 0.3 * torch.randn(c, 3 * 16 * 16, generator=gen), 0, []

    def decode(self, zz, bf16=False, downscale=1):
        self.calls += 1
        self.downscales.append(downscale)
        b, g = zz.shape[0], zz.shape[1]
        x = (zz.float() @ self.w).view(b, g, g, 3, 16, 16).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, 16 * g, 16 * g)
        x = torch.tanh(x)
        return torch.nn.functional.avg_pool2d(x, downscale) if downscale > 1 else x


def spy(model):
    """Record every trunk pass (batch, layout kind), every prepare(repeat) and every layout build; refuse forward_cfg."""
    log = {"forward": [], "prepare": [], "builds": 0, "build_steps": []}
    fwd, prep = model.forward, model.prepare

    def forward(z, t, ctx, layout=None, **kw):
        log["forward"].append((z.shape[0], ctx.shape[0], type(layout).__name__))
        return fwd(z, t, ctx, layout, **kw)

    def prepare(layout, text_len, repeat=1):
        log["prepare"].append(repeat)
        return prep(layout, text_len, repeat=repeat)

    def forward_cfg(*a, **kw):
        raise AssertionError("forward_cfg called on the no-guidance path")
    model.forward, model.prepare, model.forward_cfg = forward, prepare, forward_cfg
    build = S.build_exact_budget_layout_s8

    def counted(pyr, budget):
        log["builds"] += 1
        log["build_steps"].append(len(log["forward"]))              # (the step about to run)
        return build(pyr, budget)
    S.build_exact_budget_layout_s8 = counted
    return log, lambda: setattr(S, "build_exact_budget_layout_s8", build)


def test_sampler():
    print("[sampler without guidance]")
    gen = torch.Generator().manual_seed(3)
    torch.manual_seed(0)
    model = randomize(BudgetPixFlux2(TINY, BPConfig()).eval(), gen)
    model.family = DISTILLED                                        # (weights.load_model sets it)
    _, _, ctx = tiny_inputs(b=2, gen=gen)
    text = SpyText(TINY.context_in_dim)
    prompts = ["a red fox", "a corgi on the moon"]
    cond, uncond, lens = S.encode_prompts(text, model, prompts, amp=False)
    check("no empty prompt encoded", uncond is None and lens == {"len_cond": None, "len_uncond": None} and text.seen == prompts
          and cond.shape[0] == 2, f"the text encoder saw {text.seen}")
    text.seen.clear()
    _, u2, _ = S.encode_prompts(text, model, prompts, amp=False, uncond=True)
    check("uncond=True still encodes it", u2 is not None and text.seen == prompts + ["", ""])

    vae = PixelVAE(TINY.in_channels, gen)
    noise = torch.randn(2, L.GRID, L.GRID, TINY.in_channels, generator=gen)
    log, restore = spy(model)
    try:
        with torch.no_grad():
            res = S.sample(model, vae, cond, uncond, budget=0.3, noise=noise, amp=False, **lens)
            n_fwd, kinds, calls = len(log["forward"]), set(log["forward"]), vae.calls
            side = 16 * L.GRID
            check("defaults: 4 steps, layouts at steps 0 and 1, a 256 px preview", n_fwd == 4 and log["build_steps"] == [0, 1]
                  and calls == 1 + 1 and vae.downscales[0] == side // 256 and vae.downscales[-1] == 1
                  and bool((L.count_tokens(res["layout"]) == L.snap_tokens(0.3)).all()) and abs(res["tokens"] - L.snap_tokens(0.3)) < 1e-6,
                  f"{n_fwd} trunk passes, layouts before steps {log['build_steps']} (step 0 from the noise, step 1 from "
                  f"{calls - 1} preview decoded at 1/{vae.downscales[0]} of {side} px), {L.snap_tokens(0.3)} tokens")
            check("one row per image", kinds == {(2, 2, "PreparedLayout")} and log["prepare"] == [1, 1],
                  f"passes {sorted(kinds)}, prepare(repeat=) {log['prepare']}")
            explicit = S.sample(model, vae, cond, uncond, budget=0.3, noise=noise, amp=False, steps=4, cfg=1.0, relayout_every=1,
                                relayout_until=1, preview_px=256, **lens)
            check("defaults == the explicit settings (4 steps, cfg 1, relayout_every 1, relayout_until 1, preview_px 256)",
                  torch.equal(res["latent"], explicit["latent"]) and torch.equal(res["image"], explicit["image"])
                  and all(torch.equal(res["layout"][s], explicit["layout"][s]) for s in res["layout"]))
            full_px = S.sample(model, vae, cond, uncond, budget=0.3, noise=noise, amp=False, preview_px=side, **lens)
            prev = S.decode_preview(vae, noise, 256)
            check("preview_px: the full-size preview on request, and the small one upsampled to full size",
                  vae.downscales[-2] == 1 and tuple(prev.shape) == (2, 3, side, side) and vae.downscales[-1] == side // 256
                  and torch.isfinite(full_px["latent"]).all().item(), f"preview {tuple(prev.shape)}")
            log["build_steps"].clear()
            log["forward"].clear()
            every = S.sample(model, vae, cond, uncond, budget=0.3, noise=noise, amp=False, relayout_every=1, relayout_until=None, **lens)
            check("a refresh at every step on request", log["build_steps"] == [0, 1, 2, 3]
                  and not torch.equal(every["latent"], res["latent"]), "relayout_every=1, relayout_until=None: steps 0, 1, 2, 3")
            with_null = S.sample(model, vae, cond, model.project_text(torch.zeros_like(ctx)), budget=0.3, noise=noise, amp=False, **lens)
            check("cfg 1.0 ignores an unconditional text", torch.equal(res["latent"], with_null["latent"]))
            per_step = S.sample(model, vae, cond, uncond, budget=0.3, noise=noise, amp=False, reuse_layout=False, **lens)
            check("prepared layouts == per-step path", torch.equal(res["latent"], per_step["latent"]))

            log["forward"].clear()
            full = S.sample(model, None, cond, uncond, budget=1.0, noise=noise, amp=False, decode=False)
            z, ts = noise.clone(), L.schedule(4, DISTILLED)
            for i in range(4):
                tt = torch.full((2,), ts[i])
                z = z + (ts[i + 1] - ts[i]) * model.forward_dense(z, tt, cond, ctx_projected=True)
            d = float((full["latent"] - z).abs().max())
            check("100 % == plain unguided Euler on get_schedule(4, tokens)", d < 1e-5 and full["layout"] is None
                  and [k[:2] for k in log["forward"]] == [(2, 2)] * 4, f"max |diff| {d:.1e}")

            from budgetlatent.inference.fast import warmup
            log["forward"].clear()
            log["builds"] = 0
            log["build_steps"].clear()
            warmup(model, vae, cond, uncond, [0.5, 0.1], noise=noise, amp=False, **lens)
            check("fast.warmup: the family's cadence, one row", len(log["forward"]) == 2 * 3 and log["build_steps"] == [0, 1, 3, 4]
                  and {k[:2] for k in log["forward"]} == {(2, 2)}, "3 steps (relayout_every 1 + 2) per budget, layouts at steps 0 and 1")

            from budgetlatent.inference import validate as V
            text.seen.clear()
            with tempfile.TemporaryDirectory() as tmp:
                perf = V.render(model, vae, text, V.PROMPTS[:3], tmp, budgets=(1.0, 0.1), batch=2, thumb=32, amp=False)
                strips = sorted(os.listdir(tmp))
            check("validation suite: the family's sampler, no empty prompt", "" not in text.seen and len(text.seen) == 3
                  and strips == ["000.jpg", "001.jpg", "002.jpg"] and perf["0.1"]["tokens"] == L.snap_tokens(0.1),
                  f"{len(text.seen)} prompts encoded, strips {strips}")
    finally:
        restore()


def test_scripts():
    print("[scripts]")
    for rel in ("scripts/inference/sample.py", "scripts/inference/gallery.py", "scripts/inference/bench.py", "scripts/fetch_weights.py"):
        out = subprocess.run([sys.executable, os.path.join(ROOT, rel), "--help"], capture_output=True, text=True)
        check(f"{rel} --family flux2_distilled", out.returncode == 0 and "flux2_distilled" in out.stdout)


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    which = sys.argv[1:] or ["family", "detection", "schedule", "sampler", "scripts"]
    for name in which:
        globals()[f"test_{name}"]()
    print(f"\n{len(OK)} checks passed")
