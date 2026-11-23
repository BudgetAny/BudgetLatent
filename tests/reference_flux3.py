"""OPTIONAL: the FLUX 3 release through the official code (diffusers 0.42.1), against this code -- one GPU, real weights.

    PYTHONPATH=<an environment with diffusers 0.42.1 + transformers 5.x> python tests/reference_flux3.py --out runs/flux3_ref_512 --res 512

Not part of tests/run_cpu.sh or tests/run_gpu.sh: it needs an environment of its own (diffusers 0.42.1 and
transformers 5.x first on PYTHONPATH, which the training environment does not use); from this code it imports only
what needs no transformers (the trunk, the model, the sampler, the layouts).

  A  trunk     flux3_trunk.Flux3Trunk == diffusers' Flux3Transformer2DModel on the released weights, fp32 and bf16,
               the same inputs (the official Qwen3-VL text states of real prompts, two text windows), at three times;
               and our padded two-prompt batch (per-row windows) == the official one-prompt calls
  B  sampler   the official modular pipeline (prompt upsampler removed: the prompts are used as given), text to
               image, 50 steps, guidance 3.5, shift 6.93, from given noise -- fp32 and bf16 -- against sample.sample
               at 100 % on BudgetPixFlux3 with the official text states and the same noise
  C  saved     the official text states (cond and empty prompt), latents and images, for tests/sanity_flux3.py to
               compare this code's text encoder and its end-to-end run with
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch

PROMPTS = [
    "a red fox standing in a snowy forest at golden hour, detailed fur",
    "A cozy reading nook beside a tall rain-streaked window at dusk: a deep green velvet armchair with a knitted cream "
    "blanket draped over one arm, a small walnut side table holding a steaming ceramic mug and a stack of worn "
    "hardcover books, a brass floor lamp casting warm amber light across the wooden floorboards, a sleeping tabby cat "
    "curled on a round wool rug, potted ferns on the windowsill, soft focus city lights glowing through the glass, "
    "photographed with a 35 mm lens, shallow depth of field, gentle film grain.",
]
REPORT = {}


def note(name, ok, detail):
    REPORT[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def stats(a, b):
    """max |a - b|, that relative to max |b|, and the rms of the difference relative to the rms of b."""
    a, b = a.double(), b.double()
    d = (a - b).abs()
    return float(d.max()), float(d.max() / b.abs().max()), float(d.pow(2).mean().sqrt() / b.pow(2).mean().sqrt())


def psnr(a, b):
    """Images in [-1, 1]."""
    mse = float((a.float().clamp(-1, 1) - b.float().clamp(-1, 1)).pow(2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(4.0 / mse)


def build_pipeline(root, dtype, device):
    from diffusers.modular_pipelines.flux3 import Flux3AutoBlocks
    blocks = Flux3AutoBlocks()
    blocks.sub_blocks.pop("prompt_upsampler")
    pipe = blocks.init_pipeline(root)
    pipe.load_components(dtype={"vae": torch.float32, "default": dtype})
    pipe.to(device)
    return pipe


def official_embeds(pipe, prompt, device):
    from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
    return Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(pipe.text_encoder, pipe.tokenizer, prompt, device=device)


def ids(g, lt, device, scale=0.5):
    img = torch.cartesian_prod(torch.arange(1), torch.arange(g), torch.arange(g), torch.arange(1)).float()
    img[:, 1:3] *= scale
    txt = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(lt)).float()
    return img.to(device), txt.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--prompts", type=int, default=2, help="how many of PROMPTS")
    ap.add_argument("--skip", nargs="*", default=[], help="parts to skip: A B")
    ap.add_argument("--precisions", nargs="+", default=["fp32", "bf16"], choices=["fp32", "bf16"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import diffusers
    import transformers
    from budgetlatent.inference.sampling import sample
    from budgetlatent.models.families import FAMILIES
    from budgetlatent.models.weights import klein_dir, load_model, load_teacher
    from budgetlatent.tokenization.layout import CELL_PX, GRID
    if GRID * CELL_PX != a.res:
        raise SystemExit(f"--res {a.res} needs BP_GRID={a.res // CELL_PX}")
    dev = torch.device("cuda")
    root = klein_dir(FAMILIES["flux3"].repo)
    prompts = PROMPTS[:a.prompts]
    g = a.res // 16
    print(f"[ref] diffusers {diffusers.__version__}, transformers {transformers.__version__}, torch {torch.__version__}, "
          f"{torch.cuda.get_device_name(0)}; {a.res} x {a.res}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prev = os.path.join(a.out, "reference.pt")                      # a rerun of one precision keeps the other's results
    saved = torch.load(prev, weights_only=False) if os.path.exists(prev) else {}
    for k in [k for k in saved if k.startswith("ours_images") or k.endswith(tuple(a.precisions))]:
        saved.pop(k)
    saved.update({"prompts": prompts, "res": a.res})
    noise = [torch.randn(1, 128, g, g, generator=torch.Generator().manual_seed(100 + k)) for k in range(len(prompts))]
    saved["noise"] = torch.cat(noise)

    for prec in a.precisions:
        dtype = torch.float32 if prec == "fp32" else torch.bfloat16
        tic = time.time()
        pipe = build_pipeline(root, dtype, dev)
        print(f"[ref] {prec} pipeline loaded in {time.time() - tic:.0f} s: {type(pipe.transformer).__name__}, "
              f"{type(pipe.text_encoder).__name__}, scheduler shift {pipe.scheduler.config.shift}, guider "
              f"{type(pipe.guider).__name__}({pipe.guider.guidance_scale})", flush=True)
        with torch.no_grad():
            emb = [official_embeds(pipe, p, dev) for p in prompts]
            emb_u = official_embeds(pipe, "", dev)
        saved[f"embeds_{prec}"] = [e.float().cpu() for e in emb]
        saved[f"embeds_empty_{prec}"] = emb_u.float().cpu()
        lens = [e.shape[1] for e in emb]
        print(f"[ref] text windows {lens} (empty prompt {emb_u.shape[1]})", flush=True)
        cast = dict(dtype=None) if prec == "fp32" else dict(dtype=torch.bfloat16)
        ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec == "bf16")

        # ---- A: the trunk --------------------------------------------------------------------------------
        if "A" not in a.skip:
            teacher = load_teacher(root, dev, cast["dtype"] or torch.float32, family="flux3")
            gen = torch.Generator().manual_seed(5)
            worst, rows = (0.0, 0.0, 0.0), []
            zs = [torch.randn(1, g, g, 128, generator=gen).to(dev) for _ in prompts]
            for sigma in (0.9, 0.5, 0.1):
                outs = []
                for k, e in enumerate(emb):
                    img_ids, txt_ids = ids(g, e.shape[1], dev)
                    with torch.no_grad():
                        ref = pipe.transformer(hidden_states=zs[k].flatten(1, 2).to(dtype), encoder_hidden_states=e.to(dtype),
                                               timestep=torch.full((1,), sigma, device=dev).to(dtype), img_ids=img_ids, txt_ids=txt_ids,
                                               return_dict=False)[0].float()
                        with ac:
                            mine = -teacher.forward_dense(zs[k], torch.full((1,), 1.0 - sigma, device=dev), e.to(dtype)).flatten(1, 2)
                    st = stats(mine, ref)
                    worst = tuple(max(x, y) for x, y in zip(worst, st))
                    rows.append(f"sigma {sigma} prompt {k} ({e.shape[1]} text tokens): max {st[0]:.2e} (rel {st[1]:.1e}), rms rel {st[2]:.1e}")
                    outs.append(mine)
                # the two prompts as one padded batch with their own windows
                width = max(lens)
                ctx = torch.cat([torch.nn.functional.pad(e.to(dtype), (0, 0, 0, width - e.shape[1])) for e in emb])
                with torch.no_grad(), ac:
                    both = -teacher.forward_dense(torch.cat(zs), torch.full((len(zs),), 1.0 - sigma, device=dev), ctx,
                                                  txt_len=torch.tensor(lens)).flatten(1, 2)
                st_b = stats(both, torch.cat(outs))
                rows.append(f"sigma {sigma}: padded 2-prompt batch vs one prompt at a time: max {st_b[0]:.2e} (rel {st_b[1]:.1e})")
                REPORT.setdefault(f"A {prec} batch", []).append(st_b[1])
            tol = 1e-4 if prec == "fp32" else 0.08
            note(f"A trunk {prec} == diffusers Flux3Transformer2DModel", worst[1] < tol,
                 f"worst over 3 times x {len(prompts)} prompts: max |diff| {worst[0]:.2e}, relative {worst[1]:.1e}, rms rel {worst[2]:.1e}; "
                 + " | ".join(rows))
            del teacher
            torch.cuda.empty_cache()

        # ---- B: the sampler --------------------------------------------------------------------------------
        if "B" not in a.skip:
            tic = time.time()
            lat_ref, img_ref = [], []
            for k, p in enumerate(prompts):
                out = pipe(prompt=p, height=a.res, width=a.res, latents=noise[k].to(dev), num_inference_steps=a.steps,
                           output=["latents", "images"], output_type="pt")
                lat_ref.append(out["latents"].float().cpu())
                im = out["images"]
                im = im if torch.is_tensor(im) else torch.stack([torch.as_tensor(x) for x in im])
                img_ref.append((im.float().cpu() * 2 - 1) if float(im.min()) >= 0 else im.float().cpu())
            t_ref = (time.time() - tic) / len(prompts)
            saved[f"latents_{prec}"], saved[f"images_{prec}"] = torch.cat(lat_ref), torch.cat(img_ref)
            sched = [float(s) for s in pipe.scheduler.sigmas]
            from budgetlatent.tokenization.layout import schedule
            ours_s = [1.0 - v for v in schedule(a.steps, "flux3")]
            d_s = max(abs(x - y) for x, y in zip(sched, ours_s))
            note(f"B0 schedule {prec}", d_s < 1e-6 and len(sched) == a.steps + 1,
                 f"diffusers sigmas {sched[:3]} ... {sched[-3:]}; max |diff| to layout.schedule(50, 'flux3') {d_s:.1e}")
            del pipe
            torch.cuda.empty_cache()
            model = load_model(None, root, dev, family="flux3", **cast).eval().requires_grad_(False)
            from budgetlatent.encoders.vae import Flux2VAE
            vae = Flux2VAE(root, dev)
            res_rows, worst = [], (0.0, 0.0, 0.0)
            tic = time.time()
            for k, p in enumerate(prompts):
                with torch.no_grad(), ac:
                    c = model.project_text(emb[k].to(dtype))
                    u = model.project_text(emb_u.to(dtype))
                out = sample(model, vae, c, u, budget=1.0, steps=a.steps, noise=noise[k].permute(0, 2, 3, 1).to(dev),
                             amp=prec == "bf16", decode=True)
                lat = out["latent"].permute(0, 3, 1, 2).float().cpu()
                r = lat_ref[k]
                if r.shape != lat.shape:                             # tokens (1, N, C) instead of the unpacked grid
                    r = r.reshape(1, g, g, -1).permute(0, 3, 1, 2)
                    lat_ref[k] = r
                st = stats(lat, r)
                worst = tuple(max(x, y) for x, y in zip(worst, st))
                pp = psnr(out["image"].cpu(), img_ref[k])
                res_rows.append(f"prompt {k}: latent max |diff| {st[0]:.2e} (rel {st[1]:.1e}, rms rel {st[2]:.1e}), image PSNR {pp:.1f} dB")
                REPORT.setdefault(f"B {prec} psnr", []).append(pp)
                saved.setdefault(f"ours_images_{prec}", []).append(out["image"].cpu())
            t_ours = (time.time() - tic) / len(prompts)
            if prec == "bf16":                                       # the official bf16 conventions, on this code's model
                from budgetlatent.models import flux3_trunk as trunk_flux3
                from budgetlatent.tokenization.layout import schedule as sched_fn
                emu = []
                trunk_flux3.TIME_BF16 = True                         # 1000 sigma on the bf16 grid
                try:
                    ts = sched_fn(a.steps, "flux3")
                    for k in range(len(prompts)):
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            c = model.project_text(emb[k].to(dtype))
                            u = model.project_text(emb_u.to(dtype))
                        z = noise[k].permute(0, 2, 3, 1).to(dev).to(torch.bfloat16)          # the state in bf16
                        for i in range(a.steps):
                            tt = torch.full((1,), ts[i], device=dev)
                            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                                v_c = model(z.float(), tt, c, ctx_projected=True).to(torch.bfloat16)
                                v_u = model(z.float(), tt, u, ctx_projected=True).to(torch.bfloat16)
                            v = v_u + 3.5 * (v_c - v_u)                                   # the guider, in bf16
                            z = (z.float() + (ts[i + 1] - ts[i]) * v.float()).to(torch.bfloat16)
                        img = vae.decode(z.float()).clamp(-1, 1).cpu()
                        lat = z.float().permute(0, 3, 1, 2).cpu()
                        st = stats(lat, lat_ref[k])
                        pe = psnr(img, img_ref[k])
                        emu.append(f"prompt {k}: latent rms rel {st[2]:.1e}, image PSNR {pe:.1f} dB")
                        REPORT.setdefault("B bf16 emulated psnr", []).append(pe)
                finally:
                    trunk_flux3.TIME_BF16 = False
                res_rows.append("with the official bf16 conventions (time on the bf16 grid, guidance and state in bf16): " + "; ".join(emu))
            if prec == "fp32":
                ok_b = worst[2] < 1e-3
            else:                                                    # bf16 trajectories drift: judge against the exact (fp32) one
                ok_b, rows32 = True, []
                if "images_fp32" in saved:
                    for k in range(len(prompts)):
                        p_ours = psnr(saved["ours_images_bf16"][k], saved["images_fp32"][k:k + 1])
                        p_off = psnr(img_ref[k], saved["images_fp32"][k:k + 1])
                        rows32.append(f"prompt {k}: vs the official fp32 image, ours bf16 {p_ours:.1f} dB, official bf16 {p_off:.1f} dB")
                        ok_b = ok_b and p_ours >= p_off - 1.0
                    res_rows.append("; ".join(rows32))
                else:
                    ok_b = worst[2] < 0.25
            note(f"B sampler {prec} (official text states) == official pipeline", ok_b,
                 f"{a.steps} steps, guidance 3.5: " + " | ".join(res_rows) + f"; {t_ref:.1f} s / image official, {t_ours:.1f} s ours")
            del model, vae
            torch.cuda.empty_cache()
        else:
            del pipe
            torch.cuda.empty_cache()

    for k in list(saved):
        if isinstance(saved[k], list) and saved[k] and torch.is_tensor(saved[k][0]) and k.startswith("ours_images") and saved[k][0].dim() == 4:
            saved[k] = torch.cat(saved[k])
    torch.save(saved, os.path.join(a.out, "reference.pt"))
    try:
        from PIL import Image
        for prec in ("fp32", "bf16"):
            for key in (f"images_{prec}", f"ours_images_{prec}"):
                if key in saved:
                    for k, im in enumerate(saved[key]):
                        arr = ((im.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
                        Image.fromarray(arr).save(os.path.join(a.out, f"{key}_{k}.png"))
    except Exception as e:                                           # noqa: BLE001
        print(f"[ref] could not write the PNGs: {e}")
    json.dump(REPORT, open(os.path.join(a.out, "report_reference.json"), "w"), indent=1, default=str)
    oks = [v["ok"] for v in REPORT.values() if isinstance(v, dict)]
    print(f"[ref] {sum(oks)} / {len(oks)} checks passed -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
