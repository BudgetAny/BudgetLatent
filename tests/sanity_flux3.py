"""GPU checks of FLUX 3 Image Klein Base with the real weights, in the training environment -- run before any training.

    python tests/sanity_flux3.py --out runs/flux3_sanity_512 [--ref runs/flux3_ref_512]
    BP_GRID=64 python tests/sanity_flux3.py --out runs/flux3_sanity_1024 [--ref runs/flux3_ref_1024]

tests/reference_flux3.py (optional: the official code in an environment of its own) compares the trunk and the sampler
with diffusers 0.42.1 and saves the official text states, latents and images; this script, in the training
environment (transformers 4.57), compares the rest:

  1  text      text.Qwen3VLText (32 of 36 layers kept, no final norm) == the full Qwen3-VL-4B at layers 4 .. 32 ==
               BFL's text_encoder.py recipe (4-row position ids) == the official states saved by tests/reference_flux3.py
               (diffusers 0.42.1 under transformers 5.x), fp32 and bf16; windows, empty prompt
  2  student   BudgetPixFlux3 on the dense layout == the released trunk (fp32, exactly), and == the bf16 teacher under
               autocast; every added output layer zero; the Wiener gain
  3  e2e       this code end to end (its text encoder, its sampler, 100 %) against the official pipeline's images
  4  latents   (--pixels) FLUX 3's VAE round trip on training images; how far the latents encoded for FLUX.2
               (--flux2_latents) are from FLUX 3's (its VAE has its own weights, so they are not reusable); FLUX 3
               latents (--latents) == its encoding; skipped without --pixels
  5  images    50-step samples at 100 / 50 / 25 % from the untrained adaptive model (the added modules at zero), the
               token layout drawn over them, with timings; 100 % is the released model
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

from budgetlatent.encoders.text import F3_EMPTY_PROMPT_IDS, F3_LAYERS, Qwen3VLText
from budgetlatent.inference.sampling import encode_prompts, sample, to_uint8
from budgetlatent.models.budgetpix import count_parameters
from budgetlatent.models.convert import is_adapter
from budgetlatent.models.families import FAMILIES
from budgetlatent.models.weights import klein_dir, load_model, load_teacher
from budgetlatent.tokenization import layout as L
from budgetlatent.training.trainlib import ZERO_AT_START

REPORT = {}


def note(name, ok, detail):
    REPORT[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).abs().max()), float((a - b).abs().max() / b.abs().max()), float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt())


def psnr(a, b):
    mse = float((a.float().clamp(-1, 1) - b.float().clamp(-1, 1)).pow(2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(4.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="the output directory of tests/reference_flux3.py (same resolution)")
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--pixels", default=None, help="pixel chunks of the training mix at this resolution (check 4)")
    ap.add_argument("--flux2_latents", default=None, help="the same chunks encoded by FLUX.2's VAE (check 4, optional)")
    ap.add_argument("--latents", default=None, help="the same chunks encoded by FLUX 3's VAE (check 4, optional)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    res = L.GRID * L.CELL_PX
    root = klein_dir(FAMILIES["flux3"].repo)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import transformers
    print(f"[sanity] FLUX 3 at {res} x {res}; transformers {transformers.__version__}, torch {torch.__version__}, "
          f"{torch.cuda.get_device_name(0)}", flush=True)
    ref = torch.load(os.path.join(a.ref, "reference.pt"), weights_only=False) if a.ref and os.path.exists(os.path.join(a.ref, "reference.pt")) else None
    from reference_flux3 import PROMPTS
    prompts = ref["prompts"] if ref else PROMPTS

    # ---- 1. text ---------------------------------------------------------------------------------------------
    if "1" not in a.skip:
        for prec, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            text = Qwen3VLText(root, dev, dtype)
            ctx, lens = text.encode(prompts + [""])
            ctx = ctx.float()
            rows, ok = [], True
            # (a) the full model, its own hidden states
            from transformers import Qwen3VLForConditionalGeneration
            full = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(root, "text_encoder"), dtype=dtype).to(dev).eval()
            # (b) BFL's recipe, one prompt at a time: its window, (4, B, L) position ids
            worst_full = worst_bfl = 0.0
            for k, p in enumerate(prompts + [""]):
                ids, mask = text.tokenize([p])
                n = int(lens[k])
                with torch.no_grad():
                    pos = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1).to(dev)
                    o = full(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos[None].expand(4, -1, -1),
                             output_hidden_states=True, use_cache=False)
                    bfl = torch.cat([o.hidden_states[j] for j in F3_LAYERS], dim=-1).float()
                    o2 = full.model.language_model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos,
                                                   output_hidden_states=True, use_cache=False)
                    lm = torch.cat([o2.hidden_states[j] for j in F3_LAYERS], dim=-1).float()
                worst_bfl = max(worst_bfl, rel(ctx[k:k + 1, :n], bfl[:, :n])[1])
                worst_full = max(worst_full, rel(ctx[k:k + 1, :n], lm[:, :n])[1])
                if ids.shape[1] != n:
                    ok = False
            del full
            torch.cuda.empty_cache()
            tol = 1e-5 if prec == "fp32" else 2e-2
            rows.append(f"vs the full 36-layer language model: max rel {worst_full:.1e}; vs BFL's recipe (4-row position ids): {worst_bfl:.1e}")
            ok = ok and worst_full < tol and worst_bfl < tol
            pad_same = float((ctx[0, int(lens[0]) - 1] - ctx[0, -1]).abs().max()) if ctx.shape[1] > int(lens[0]) else 0.0
            # (c) the official states (diffusers 0.42.1, transformers 5.x)
            if ref is not None:
                worst = 0.0
                for k, e in enumerate(ref[f"embeds_{prec}"] + [ref[f"embeds_empty_{prec}"]]):
                    if e.shape[1] != int(lens[k]):
                        ok = False
                        rows.append(f"prompt {k}: official window {e.shape[1]}, ours {int(lens[k])}")
                        continue
                    worst = max(worst, rel(ctx[k:k + 1, :e.shape[1]].cpu(), e)[1])
                rows.append(f"vs the official states (diffusers 0.42.1, transformers {ref.get('transformers', '5.x')}): max rel {worst:.1e}")
                ok = ok and worst < (1e-3 if prec == "fp32" else 5e-2)
            ids_e, _ = text.tokenize([""])
            note(f"1 text {prec}", ok and tuple(ids_e[0, :8].tolist()) == F3_EMPTY_PROMPT_IDS and int(lens[-1]) == 80,
                 f"windows {lens.tolist()} (empty prompt {int(lens[-1])}); " + "; ".join(rows)
                 + f"; padding beyond the window repeats the pad state (max |diff| {pad_same:.1e})")
            if prec == "bf16":
                keep_text = text
            else:
                del text
                torch.cuda.empty_cache()
    else:
        keep_text = Qwen3VLText(root, dev, torch.bfloat16)

    # ---- 2. the student on the dense layout ------------------------------------------------------------------
    if "2" not in a.skip:
        g = torch.Generator().manual_seed(0)
        z = torch.randn(2, L.GRID, L.GRID, 128, generator=g).to(dev)
        t = torch.tensor([0.2, 0.7], device=dev)
        ctx, lens = keep_text.encode(prompts[:2])
        ctx, lens = ctx.to(dev), lens.cpu()
        student = load_model(None, root, dev, family="flux3").eval()
        n_trunk, n_add = count_parameters(student)
        teacher32 = load_teacher(root, dev, torch.float32, family="flux3")
        with torch.no_grad():
            d32 = float((student(z, t, ctx.float(), L.pack(L.dense_layout(2, dev)), txt_len=lens)
                         - teacher32.forward_dense(z, t, ctx.float(), txt_len=lens)).abs().max())
        del teacher32
        torch.cuda.empty_cache()
        teacher = load_teacher(root, dev, torch.bfloat16, family="flux3")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            v_t = teacher.forward_dense(z, t, ctx, txt_len=lens)
            v_s = student(z, t, ctx, L.pack(L.dense_layout(2, dev)), txt_len=lens)
        dbf = rel(v_s, v_t)
        nonzero = [n for n, p in student.named_parameters() if is_adapter(n) and n.endswith(ZERO_AT_START) and bool(p.any())]
        note("2 student", d32 < 1e-5 and dbf[1] < 0.02 and not nonzero,          # (exactly 0 on a CPU; GPU kernels pick per instance)
             f"{n_trunk:,} trunk + {n_add:,} added parameters; dense layout vs the released trunk: fp32 max |diff| {d32:.1e}; fp32 "
             f"student under bf16 autocast vs the bf16 teacher: max rel {dbf[1]:.1e}, rms rel {dbf[2]:.1e}; added output layers zero: {not nonzero}")
        del student, teacher
        torch.cuda.empty_cache()

    # ---- 3 + 5: end to end, and samples at budgets --------------------------------------------------------------
    from budgetlatent.encoders.vae import Flux2VAE
    vae = Flux2VAE(root, dev)
    model = load_model(None, root, dev, dtype=torch.bfloat16, family="flux3").eval().requires_grad_(False)
    cond, uncond, lens = encode_prompts(keep_text, model, prompts)
    from PIL import Image
    if ref is not None and "3" not in a.skip:
        rows, worst = [], 99.0
        for k in range(len(prompts)):
            out = sample(model, vae, cond[k:k + 1, :int(lens["len_cond"][k])], uncond[k:k + 1], budget=1.0, steps=a.steps,
                         noise=ref["noise"][k:k + 1].permute(0, 2, 3, 1).to(dev), amp=True)
            pp = psnr(out["image"].cpu(), ref["images_bf16"][k:k + 1])
            pq = psnr(ref["images_fp32"][k:k + 1], ref["images_bf16"][k:k + 1]) if "images_fp32" in ref else float("nan")
            p32 = psnr(out["image"].cpu(), ref["images_fp32"][k:k + 1]) if "images_fp32" in ref else float("nan")
            # bf16 trajectories drift from the exact one: this code passes if it is at least as close to the official fp32
            # image as the official bf16 pipeline is (within 1 dB)
            worst = min(worst, p32 - pq) if p32 == p32 and pq == pq else min(worst, pp - 25.0)
            rows.append(f"prompt {k}: PSNR vs official bf16 {pp:.1f} dB, vs official fp32 {p32:.1f} dB (official bf16 vs fp32: {pq:.1f} dB)")
            Image.fromarray(to_uint8(out["image"])[0].numpy()).save(os.path.join(a.out, f"e2e_{k}.png"))
        note("3 end to end (bf16) vs the official pipeline", worst > -1.0,
             " | ".join(rows) + " (criterion: ours bf16 at least as close to the official fp32 image as the official bf16 pipeline, -1 dB)")

    if "5" not in a.skip:
        rows = []
        noise = torch.cat([torch.randn(1, L.GRID, L.GRID, 128, generator=torch.Generator().manual_seed(42 + k)) for k in range(len(prompts))]).to(dev)
        n_cond = cond.shape[1]
        base_img = None
        for b in a.budgets:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tic = time.time()
            out = sample(model, vae, cond, uncond, budget=b, steps=a.steps, noise=noise, amp=True, **lens)
            torch.cuda.synchronize()
            sec = (time.time() - tic) / len(prompts)
            imgs = to_uint8(out["image"])
            over = L.overlay(out["image"], out["layout"] or L.dense_layout(len(prompts), dev)).cpu()
            for k in range(len(prompts)):
                Image.fromarray(imgs[k].numpy()).save(os.path.join(a.out, f"p{k}_b{round(100 * b):03d}.png"))
                Image.fromarray(over[k].numpy()).save(os.path.join(a.out, f"p{k}_b{round(100 * b):03d}_layout.png"))
            if b == 1.0:
                base_img = out["image"]
            pv = psnr(out["image"], base_img) if base_img is not None and b != 1.0 else float("nan")
            rows.append(f"{round(100 * b)} % ({L.snap_tokens(b)} tokens): {sec:.1f} s / image, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB"
                        + ("" if b == 1.0 else f", PSNR vs 100 % {pv:.1f} dB"))
        # 100 % through the plain released trunk, the same sampler arithmetic: must be the same image
        if 1.0 in a.budgets:
            teacher = load_teacher(root, dev, torch.bfloat16, family="flux3")
            z = noise.clone()
            ts = L.schedule(a.steps, "flux3")
            with torch.no_grad():
                for i in range(a.steps):
                    tt = torch.full((len(prompts),), ts[i], device=dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        width = max(cond.shape[1], uncond.shape[1])
                        pad = lambda x: torch.nn.functional.pad(x, (0, 0, 0, width - x.shape[1]))
                        v = teacher.forward_dense(torch.cat([z, z]), torch.cat([tt, tt]), torch.cat([pad(cond), pad(uncond)]),
                                                  ctx_projected=True, txt_len=torch.cat([lens["len_cond"], lens["len_uncond"]]))
                    v_c, v_u = v.chunk(2)
                    z = z + (ts[i + 1] - ts[i]) * (v_u + 3.5 * (v_c - v_u))
            # (the trunk was built with the released projections; project_text above used the student's identical ones)
            d = psnr(vae.decode(z).clamp(-1, 1), base_img)
            rows.append(f"100 % == the plain released trunk through the same steps: PSNR {d:.1f} dB")
            del teacher
            ok100 = d >= 60.0
        else:
            ok100 = True
        note("5 samples at budgets (untrained adaptive model)", ok100, " | ".join(rows) + f"; images in {a.out}")

    # ---- 4. latents ---------------------------------------------------------------------------------------------
    if "4" not in a.skip and a.pixels:
        flux2_dir = a.flux2_latents or ""                                     # encoded with FLUX.2's VAE
        lp, pp_ = os.path.join(flux2_dir, "chunk_000000.pt"), os.path.join(a.pixels, "chunk_000000.pt")
        if os.path.exists(pp_):
            from budgetlatent.training.data import to_unit
            px = torch.load(pp_, map_location="cpu", mmap=True, weights_only=False)["pixel_values"][:8]
            x = to_unit(px, dev)
            with torch.no_grad():
                enc = vae.encode(x)
                p3 = psnr(vae.decode(enc).clamp(-1, 1), x)
            rows = [f"FLUX 3's VAE on 8 training images: round trip {p3:.1f} dB, latent mean {float(enc.mean()):+.3f} std {float(enc.std()):.3f}"]
            ok4 = p3 > 25.0
            if flux2_dir and os.path.exists(lp):
                old = torch.load(lp, map_location="cpu", mmap=True, weights_only=False)["latent"][:8].float()
                d = rel(old, enc.cpu())
                with torch.no_grad():
                    p_old = psnr(vae.decode(old.to(dev)).clamp(-1, 1), x)
                rows.append(f"FLUX.2-encoded latents ({flux2_dir}) vs FLUX 3's: rms rel diff {d[2]:.1e}; decoded by FLUX 3's VAE {p_old:.1f} dB "
                            f"(FLUX.2's VAE is FLUX 3's rounded to bf16; the run re-encodes: latents_flux3_<res>)")
            lat3 = a.latents
            if lat3 and os.path.exists(os.path.join(lat3, "chunk_000000.pt")):
                new3 = torch.load(os.path.join(lat3, "chunk_000000.pt"), map_location="cpu", mmap=True, weights_only=False)["latent"][:8].float()
                d3 = rel(new3, enc.cpu())
                rows.append(f"{lat3}: == FLUX 3's encoding, rms rel diff {d3[2]:.1e} (max rel {d3[1]:.1e}; "
                            f"scripts/data/encode_latents.py runs cuDNN's default TF32 convolutions, this check fp32)")
                ok4 = ok4 and d3[2] < 5e-3
            note("4 latents", ok4, "; ".join(rows))
        else:
            note("4 latents", False, f"{pp_} missing")

    ok = all(r["ok"] for r in REPORT.values())
    print(f"[sanity] {sum(r['ok'] for r in REPORT.values())} / {len(REPORT)} checks passed", flush=True)
    json.dump(REPORT, open(os.path.join(a.out, "report_sanity.json"), "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
