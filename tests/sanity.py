"""GPU checks with the real weights -- run before any training (the start-up checks, and a few more).

    python tests/sanity.py --out runs/flux2_sanity [--ckpt budgetpix_flux2_s8_step64000_ema1_bf16.safetensors]

  1  weights     the diffusers transformer, key-converted, is tensor-for-tensor the BFL single file
  2  trunk       this code == BFL's reference model.py == diffusers' Flux2Transformer2DModel (fp32, real weights)
  3  student     dense layout == the released model (fp32), and == the bf16 teacher under autocast
  4  schedule    end points of the 50-step schedule; the Wiener gain curve
  5  vae         latent statistics and the round-trip PSNR of a photograph (--image) at 512 x 512 (the grid size: BP_GRID)
  6  text        27-layer encoder == the full Qwen3-4B at layers 9 / 18 / 27; the two chat templates
  7  sampler     this sampler at 100 % == BFL's guided Euler loop (fp32, a few steps, same noise)
  8  images      50-step samples at a few budgets with the token layout drawn over them (bf16), with timings
  9  checkpoint  (--ckpt) a BudgetPix checkpoint, e.g. the dit/ export, has every key with the right shape

FLUX2_ROOT (default third_party/flux2) is a checkout of github.com/black-forest-labs/flux2, the reference of 2 and 7.
One GPU with 48 GB is enough.  --device cpu runs the fp32 checks (1, 2, 4-7, 9) without a GPU: slow (the 4B model
on CPU threads), but it needs no GPU slot; the bf16 checks (3's teacher half, 8) are GPU only.
"""
import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch

from budgetlatent.encoders.text import EMPTY_PROMPT_IDS, PAD_ID, Qwen3Text, tokenize, wrap
from budgetlatent.encoders.vae import Flux2VAE
from budgetlatent.inference.sampling import initial_noise, sample, to_uint8
from budgetlatent.models import convert
from budgetlatent.models.added_modules import hf_skip_gain
from budgetlatent.models.budgetpix import BPConfig, BudgetPixFlux2, count_parameters
from budgetlatent.models.flux2_trunk import TrunkConfig
from budgetlatent.models.weights import BFL_FILE, klein_dir, load_model, load_teacher, read_state, resolve_checkpoint
from budgetlatent.tokenization import layout as L

REPORT = {}


def note(name, ok, detail):
    REPORT[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def bfl_module():
    root = os.environ.get("FLUX2_ROOT") or os.path.join(ROOT, "third_party", "flux2")
    path = os.path.join(root, "src", "flux2", "model.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("bfl_flux2_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bfl_flux2_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def ids_for(b, g, lt, device):
    """BFL's position ids: image (0, h, w, 0) row-major, text (0, 0, 0, l)."""
    x_ids = torch.cartesian_prod(torch.arange(1), torch.arange(g), torch.arange(g), torch.arange(1))[None].expand(b, -1, -1)
    c_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(lt))[None].expand(b, -1, -1)
    return x_ids.to(device), c_ids.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/flux2_sanity")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--ckpt", default=None, help="a BudgetPix checkpoint (a local file, e.g. the dit/ export, or a Hugging Face "
                                                 "repo id) to verify and sample from")
    ap.add_argument("--image", default=None, help="a natural photograph for the VAE round trip (5; skipped without one)")
    ap.add_argument("--prompts", nargs="+", default=["a red fox standing in a snowy forest at golden hour, detailed fur",
                                                     "a corgi wearing a spacesuit on the moon, the earth in the black sky"])
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--sampler_steps", type=int, default=6, help="steps of the sampler comparison (7)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--skip", nargs="*", default=[], help="stages to skip, by number")
    a = ap.parse_args()
    a.ckpt = resolve_checkpoint(a.ckpt)                             # a Hub repo id -> its downloaded file
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    gpu = dev.type == "cuda"
    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file
    root = klein_dir(a.weights) if a.weights else klein_dir()
    print(f"[sanity] weights {root}; {torch.cuda.get_device_name(0) if gpu else f'cpu, {torch.get_num_threads()} threads'}, "
          f"torch {torch.__version__}", flush=True)
    skip = set(a.skip) | (set() if gpu else {"8"})
    enc_dtype = torch.bfloat16 if gpu else torch.float32
    prompts, b = list(a.prompts), len(a.prompts)

    # ---- 1. the two released layouts hold the same tensors ----------------------------------------------
    dif_path = os.path.join(root, "transformer", "diffusion_pytorch_model.safetensors")
    if "1" not in skip and os.path.exists(dif_path):
        sd_bfl, conv = load_file(os.path.join(root, BFL_FILE)), convert.diffusers_to_bfl(load_file(dif_path))
        diff = [k for k in conv if k in sd_bfl and not torch.equal(conv[k], sd_bfl[k])]
        note("1 weights", set(conv) == set(sd_bfl) and not diff, f"{len(sd_bfl)} tensors, {sum(v.numel() for v in sd_bfl.values()):,} parameters, "
             f"stored {next(iter(sd_bfl.values())).dtype}; same keys {set(conv) == set(sd_bfl)}, tensors that differ {len(diff)} {diff[:3]}")
        del sd_bfl, conv
        gc.collect()

    # ---- 6. text encoder (its states are the test input of everything below) ----------------------------
    text = Qwen3Text(root, dev, dtype=enc_dtype)                       # BFL's template ("bfl_nothink"), the default
    with torch.no_grad():
        ctx, ctx_empty = text(prompts).float(), text([""] * b).float()
        ctx_nothink = text.encode_ids(*tokenize(text.tokenizer, prompts, "no_think")).float()
        ctx_empty_nothink = text.encode_ids(*tokenize(text.tokenizer, [""] * b, "no_think")).float()
    if "6" not in skip:
        ids_s, mask_s = tokenize(text.tokenizer, ["", prompts[0]], "bfl_nothink")
        ids_o, _ = tokenize(text.tokenizer, ["", prompts[0]], "no_think")
        n_s, n_o = int((ids_s[0] != PAD_ID).sum()), int((ids_o[0] != PAD_ID).sum())
        official = text.tokenizer.apply_chat_template([{"role": "user", "content": prompts[0]}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
        same_text = official == wrap(prompts[0], "bfl_nothink")
        empty_ok = ids_s[0, :n_s].tolist() == list(EMPTY_PROMPT_IDS["bfl_nothink"]) and ids_o[0, :n_o].tolist() == list(EMPTY_PROMPT_IDS["no_think"])
        with torch.no_grad():
            got = text.encode_ids(ids_s, mask_s).float()
        from transformers import AutoModelForCausalLM
        full = AutoModelForCausalLM.from_pretrained(os.path.join(root, "text_encoder"), torch_dtype=enc_dtype).to(dev).eval()
        with torch.no_grad():
            hs = full(input_ids=ids_s.to(dev), attention_mask=mask_s.to(dev), output_hidden_states=True, use_cache=False).hidden_states
            want = torch.cat([hs[k] for k in (9, 18, 27)], dim=-1).float()
        d = float((got - want).abs().max())
        note("6 text encoder", d <= 1e-3 * float(want.abs().max()) and same_text and empty_ok,
             f"27-layer vs full Qwen3-4B ({len(hs) - 1} layers): max |diff| {d:.1e} over {tuple(got.shape)} (scale {float(want.abs().max()):.0f}); "
             f"empty prompt = {n_s} tokens with 'bfl_nothink' ({ids_s[0, :n_s].tolist()[8:]} after the 8 of 'no_think'), {n_o} with 'no_think', "
             f"ids as expected: {empty_ok}; 'bfl_nothink' == apply_chat_template(enable_thinking=False): {same_text}")
        del full, hs, want, got
    del text
    flush()

    z = initial_noise(b, seed=7, device=dev)
    t = torch.tensor(([0.15, 0.65] + [0.4] * b)[:b], device=dev)

    # ---- 2 / 3 / 7. the model, fp32 ----------------------------------------------------------------------
    student = load_model(None, root, dev).eval().requires_grad_(False)
    n_trunk, n_added = count_parameters(student)
    with torch.no_grad():
        mine = student.forward_dense(z, t, ctx)
        d32 = float((student(z, t, ctx, L.pack(L.dense_layout(b, dev))) - mine).abs().max())
    scale = float(mine.abs().max())
    with torch.no_grad():
        other = student.forward_dense(z, t, ctx_nothink)
    rel = float((other - mine).pow(2).mean().sqrt() / mine.pow(2).mean().sqrt())
    note("6b chat template", True, f"released model, same latent and time: velocity with the 'no_think' template differs from 'bfl_nothink' by "
         f"{100 * rel:.2f} % rms (max |diff| {float((other - mine).abs().max()):.3f} at scale {scale:.2f})")
    del other
    bfl = bfl_module()
    if bfl is None:
        print("[--] no BFL checkout (FLUX2_ROOT): 2a and 7 skipped", flush=True)
    elif not {"2", "7"} <= skip:
        ref = bfl.Flux2(bfl.Klein4BParams())
        ref.load_state_dict({k: v.float() for k, v in load_file(os.path.join(root, BFL_FILE)).items()}, strict=True)
        ref = ref.to(dev).eval().requires_grad_(False)
        if "2" not in skip:
            x_ids, c_ids = ids_for(b, L.GRID, ctx.shape[1], dev)
            with torch.no_grad():
                raw = ref(x=z.flatten(1, 2), x_ids=x_ids, timesteps=1.0 - t, ctx=ctx, ctx_ids=c_ids, guidance=None)
            d = float((mine + raw.view_as(mine)).abs().max())
            note("2a trunk == BFL model.py", d < 1e-3 * scale, f"max |diff| {d:.2e} at output scale {scale:.2f} (fp32, strict state dict)")
        if "7" not in skip:
            steps, guidance = a.sampler_steps, 4.0
            e_mu = math.exp(L.compute_empirical_mu(L.N_DENSE, steps))     # BFL get_schedule: shifted linspace(1, 0, steps + 1)
            sig = [1.0] + [e_mu / (e_mu + (1.0 / (1.0 - i / steps) - 1.0)) for i in range(1, steps)] + [0.0]
            sched = max(abs((1.0 - p) - q) for p, q in zip(L.flux2_klein_timesteps(steps, L.N_DENSE), sig))
            noise = initial_noise(b, seed=11, device=dev)
            x_ids, c_ids = ids_for(2 * b, L.GRID, ctx.shape[1], dev)
            ts = L.flux2_klein_timesteps(steps, L.N_DENSE)
            pc, pu = student.project_text(ctx), student.project_text(ctx_empty)
            per_step = 0.0
            with torch.no_grad():
                img, txt = torch.cat([noise, noise]).flatten(1, 2), torch.cat([ctx_empty, ctx])     # denoise_cfg: [uncond; cond]
                for i, (t_curr, t_prev) in enumerate(zip(sig[:-1], sig[1:])):
                    pred = ref(x=img, x_ids=x_ids, timesteps=torch.full((2 * b,), t_curr, device=dev), ctx=txt, ctx_ids=c_ids, guidance=None)
                    p_u, p_c = pred.chunk(2)
                    pred = p_u + guidance * (p_c - p_u)
                    # our guided step on the same state: the implementations compared without the path in between
                    state = img.chunk(2)[0].view(b, L.GRID, L.GRID, -1)
                    mine_g = -student.forward_cfg(state, torch.full((b,), ts[i], device=dev), pc, pu, None, guidance).flatten(1, 2)
                    per_step = max(per_step, float((mine_g - pred).abs().max()) / float(pred.abs().max()))
                    img = img + (t_prev - t_curr) * torch.cat([pred, pred])
                want = img.chunk(2)[0].view(b, L.GRID, L.GRID, -1)
                got = sample(student, None, pc, pu, budget=1.0, steps=steps, cfg=guidance, noise=noise, amp=False, decode=False)["latent"]
            d, s = float((got - want).abs().max()), float(want.abs().max())
            # fp32 round-off grows along a 6-step guided path (at 1024, ~0.2 in one region by the last, large step):
            # the steps must agree closely, the two whole paths only roughly (a wrong update would be far off)
            note("7 sampler == BFL denoise_cfg", per_step < 5e-3 and d < 0.1 * s and sched < 1e-6,
                 f"{steps} steps, guidance {guidance}: each step on BFL's state, max rel |diff| {per_step:.1e}; schedule diff {sched:.1e}; "
                 f"the two whole paths end {d:.2e} apart at scale {s:.2f}")
        del ref
        flush()
    if "2" not in skip and os.path.exists(dif_path):
        from diffusers import Flux2Transformer2DModel
        cfg = json.load(open(os.path.join(root, "transformer", "config.json")))
        dm = Flux2Transformer2DModel(**{k: v for k, v in cfg.items() if not k.startswith("_") and k != "guidance_embeds"})
        missing, unexpected = dm.load_state_dict({k: v.float() for k, v in load_file(dif_path).items()}, strict=False)
        assert not unexpected and all("guidance_embedder" in k for k in missing), (missing[:4], unexpected[:4])
        for n, p in dm.named_parameters():                           # klein has no guidance embedding: silence it
            if "guidance_embedder" in n:
                p.data.zero_()
        dm = dm.to(dev).eval().requires_grad_(False)
        x_ids, c_ids = ids_for(b, L.GRID, ctx.shape[1], dev)
        with torch.no_grad():
            raw = dm(hidden_states=z.flatten(1, 2), encoder_hidden_states=ctx, timestep=1.0 - t, img_ids=x_ids[0].float(),
                     txt_ids=c_ids[0].float(), guidance=torch.zeros(b, device=dev), return_dict=False)[0]
        d = float((mine + raw.view_as(mine)).abs().max())
        note("2b trunk == diffusers Flux2Transformer2DModel", d < 1e-3 * scale, f"max |diff| {d:.2e} at output scale {scale:.2f} (fp32, converted keys)")
        del dm
        flush()
    counts = n_trunk == 3_875_544_576 and n_added == 28_136_192
    if gpu:
        teacher = load_teacher(root, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            v_t = teacher.forward_dense(z, t, ctx.to(torch.bfloat16))
            dbf = float((student(z, t, ctx.to(torch.bfloat16), L.pack(L.dense_layout(b, dev))) - v_t).abs().max())
        d_amp = float((v_t - mine).abs().max())
        note("3 student dense == released", d32 <= 1e-5 and dbf <= 0.02 * scale and counts,
             f"max |diff| {d32:.1e} (fp32, general path vs dense path); {dbf:.1e} vs the bf16 teacher under autocast; the bf16 error of "
             f"the model itself is {d_amp:.3f} ({100 * d_amp / scale:.1f} % of the output scale); parameters {n_trunk:,} + {n_added:,}")
        del teacher, v_t
    else:
        note("3 student dense == released", d32 <= 1e-5 and counts, f"max |diff| {d32:.1e} (fp32, general path vs dense path); "
             f"parameters {n_trunk:,} + {n_added:,}; the bf16 teacher comparison needs a GPU")
    del student, mine
    flush()

    # ---- 4. schedule and Wiener gain ---------------------------------------------------------------------
    ts = L.flux2_klein_timesteps(50, L.N_DENSE)
    mu = L.compute_empirical_mu(L.N_DENSE, 50)
    if L.N_DENSE == 1024:                                            # the reference values (512 x 512)
        ok = all(abs(p - q) < 1e-6 for p, q in zip(ts[:4] + ts[-4:], [0.0, 0.003707, 0.007540, 0.011504, 0.740694, 0.813982, 0.899336, 1.0]))
    else:                                                            # other grids: 7 compares it with BFL's get_schedule
        ok = ts[0] == 0.0 and ts[-1] == 1.0 and all(q > p for p, q in zip(ts, ts[1:]))
    note("4a schedule", ok, f"mu {mu:.6f}, exp(mu) {math.exp(mu):.6f}; first {[round(v, 6) for v in ts[:4]]}, last {[round(v, 6) for v in ts[-4:]]}")
    w = hf_skip_gain(torch.tensor([1.0, 0.5, 1.0 / 17.0, 0.0], dtype=torch.float64))
    grid = torch.linspace(0, 1, 100001, dtype=torch.float64)
    peak = hf_skip_gain(grid)
    note("4b wiener gain", abs(float(w[0]) - 1) < 1e-6 and abs(float(w[2])) < 1e-6 and abs(float(w[3]) + 1) < 1e-6,
         f"w(sigma = 1) = {float(w[0]):+.4f}, w(0.5) = {float(w[1]):+.4f}, w(1/17) = {float(w[2]):+.1e}, w(0) = {float(w[3]):+.4f}; "
         f"maximum {float(peak.max()):.4f} at sigma = {float(grid[peak.argmax()]):.4f}")

    # ---- 5. VAE ------------------------------------------------------------------------------------------
    vae = Flux2VAE(root, dev)
    if "5" not in skip and not a.image:
        print("[--] no --image (a natural photograph): 5 skipped", flush=True)
    elif "5" not in skip:
        path = a.image
        im = Image.open(path).convert("RGB")
        side = min(im.size)
        box = ((im.width - side) // 2, (im.height - side) // 2, (im.width + side) // 2, (im.height + side) // 2)
        im = im.crop(box).resize((L.GRID * L.CELL_PX,) * 2, Image.BICUBIC)
        x = (torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)[None].float() / 127.5 - 1.0).to(dev)
        lat = vae.encode(x)
        rec, rec16 = vae.decode(lat), vae.decode(lat, bf16=True)
        psnr = lambda p: float(10 * torch.log10(4.0 / (p.clamp(-1, 1) - x).pow(2).mean()))
        Image.fromarray(torch.cat([to_uint8(x)[0], to_uint8(rec)[0]], dim=1).numpy()).save(os.path.join(a.out, "vae_roundtrip.png"))
        note("5 vae", lat.shape == (1, L.GRID, L.GRID, 128) and psnr(rec) > 30.0 and abs(float(lat.mean())) < 0.15 and 0.75 < float(lat.std()) < 1.25,
             f"latent {tuple(lat.shape)} mean {float(lat.mean()):+.3f} std {float(lat.std()):.3f}; "
             f"round trip {psnr(rec):.2f} dB (fp32 decode), {psnr(rec16):.2f} dB (bf16 decode) on {os.path.basename(path)}")

    # ---- 9. a checkpoint ---------------------------------------------------------------------------------
    if a.ckpt:
        sd, _ = read_state(a.ckpt)
        with torch.device("meta"):
            shapes = {n: tuple(p.shape) for n, p in BudgetPixFlux2(TrunkConfig(), BPConfig()).named_parameters()}
        missing, unexpected = [k for k in shapes if k not in sd], [k for k in sd if k not in shapes]
        wrong = [k for k in shapes if k in sd and tuple(sd[k].shape) != shapes[k]]
        n_ad = sum(v.numel() for k, v in sd.items() if convert.is_adapter(k))
        live = float(sd["refiners.2.out.weight"].float().abs().max()) if "refiners.2.out.weight" in sd else float("nan")
        row0 = float(sd["scale_emb"][0].float().abs().max()) if "scale_emb" in sd else float("nan")
        note("9 checkpoint", not missing and not unexpected and not wrong and n_ad == 28_136_192,
             f"{os.path.basename(a.ckpt)}: {len(sd)} tensors; missing {len(missing)} {missing[:3]}, unexpected {len(unexpected)} {unexpected[:3]}, "
             f"wrong shape {len(wrong)} {wrong[:3]}; added parameters {n_ad:,}; max |refiners.2.out| {live:.3g}, max |scale_emb[0]| {row0:.3g}")
        del sd
        gc.collect()

    # ---- 8. images ---------------------------------------------------------------------------------------
    if "8" not in skip:
        model = load_model(a.ckpt, root, dev, dtype=torch.bfloat16).eval().requires_grad_(False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cond, uncond = model.project_text(ctx.to(torch.bfloat16)), model.project_text(ctx_empty.to(torch.bfloat16))
        rows = []
        for budget in a.budgets:
            sample(model, vae, cond, uncond, budget=budget, steps=2, seed=42)             # kernels warm
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tic = time.time()
            res = sample(model, vae, cond, uncond, budget=budget, steps=a.steps, seed=42)
            torch.cuda.synchronize()
            sec, peak = time.time() - tic, torch.cuda.max_memory_allocated() / 2 ** 30
            lay = res["layout"] or L.dense_layout(b, dev)
            rows.append(torch.cat([torch.cat(list(to_uint8(res["image"])), dim=1), torch.cat(list(L.overlay(res["image"], lay).cpu()), dim=1)], dim=1))
            REPORT[f"8 budget {budget:g}"] = {"ok": bool(torch.isfinite(res["latent"]).all()), "detail":
                                              f"{L.snap_tokens(budget)} tokens; {sec:.1f} s for {b} images x {a.steps} steps; peak {peak:.1f} GiB"}
            print(f"[sanity] budget {budget:.0%}: {REPORT[f'8 budget {budget:g}']['detail']}", flush=True)
        if not a.ckpt:                                              # the released model without the think block, to compare
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                c_o, u_o = model.project_text(ctx_nothink.to(torch.bfloat16)), model.project_text(ctx_empty_nothink.to(torch.bfloat16))
            bfl_row = torch.cat(list(to_uint8(sample(model, vae, cond, uncond, budget=1.0, steps=a.steps, seed=42)["image"])), dim=1)
            alt_row = torch.cat(list(to_uint8(sample(model, vae, c_o, u_o, budget=1.0, steps=a.steps, seed=42)["image"])), dim=1)
            Image.fromarray(torch.cat([bfl_row, alt_row], dim=0).numpy()).save(os.path.join(a.out, "templates.png"))
            print(f"[sanity] {os.path.join(a.out, 'templates.png')}: 100 % with the 'bfl_nothink' template (top) and 'no_think' (bottom)", flush=True)
        Image.fromarray(torch.cat(rows, dim=0).numpy()).save(os.path.join(a.out, "samples.png"))
        note("8 images", all(v["ok"] for k, v in REPORT.items() if k.startswith("8 budget")),
             f"{os.path.join(a.out, 'samples.png')}: one row per budget {a.budgets}; {b} samples, then their token layouts"
             + ("" if a.ckpt else "; no checkpoint, so below 100 % this is the untrained start (mean-pooled in, upsampled out)"))

    json.dump(REPORT, open(os.path.join(a.out, "sanity.json"), "w"), indent=1)
    bad = [k for k, v in REPORT.items() if not v["ok"]]
    print(f"\n[sanity] {len(REPORT) - len(bad)} / {len(REPORT)} checks passed" + (f"; FAILED: {bad}" if bad else ""), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
