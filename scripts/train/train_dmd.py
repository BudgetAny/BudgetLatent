"""Stage 2 (FLUX.2 [klein] only): data-free DMD2 + GAN sharpening of a BudgetLatent student (the pieces are in
budgetlatent/training/dmd.py).

    torchrun --nproc_per_node=4 scripts/train/train_dmd.py --config configs/flux2_512.yaml --data data/mix120k/latents_512 \
        --init runs/bpflux2_s8/ema-final.safetensors --out runs/bpflux2_s8_dmd --max_steps 8000 --resume

--init: stage-1 EMA weights -> warm start (lower rates, two fixed EMAs, a short critic warm-up; the AdamW moments
start from zero).  Without it the student starts as the released model with the added modules at zero -> cold
start (the stage-1 rates and warm-up EMA, 60,000 steps).  The two sets of numbers are `warm_start` / `cold_start`
under dmd2_gan in configs/flux2_512.yaml.  Only the captions of --data are read (pixel or latent chunks, whichever exist).

One step, on every GPU (batch_per_gpu x grad_accum examples):
  1. captions -> text states (10 % replaced by the empty prompt);
  2. a 5-step rollout from noise without guidance -- the student's own on layouts, or the dense teacher's; one coin
     per step, the same on all GPUs -- gives "clean" latents, which are noised to t; the dense teacher's velocity
     there is the paired target, and the layouts come from the image it implies, exactly as in stage 1;
  3. the student update: velocity distillation, plus -- once the critics have warmed up -- the DMD term and the
     generator GAN term; spike guard, clip, EMAs and the warm-up counter move only on accepted updates;
  4. `ttur_updates` critic updates on the detached estimates: the LoRA learns to denoise the student's outputs,
     the head to tell them from the teacher's.

The critics are not wrapped in DDP: their gradients are averaged over the GPUs by hand (they are 11M numbers).
Per step this runs ~15 dense teacher passes (3 of them with a backward) next to the student's, so a step costs
several stage-1 steps.  Memory is as in stage 1 plus the second EMA and the teacher's checkpointed backward.
"""
import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))    # the repository root
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from budgetlatent.encoders.text import PAD_ID, Qwen3Text, template_name
from budgetlatent.encoders.vae import Flux2VAE
from budgetlatent.models.convert import is_adapter
from budgetlatent.models.budgetpix import BPConfig
from budgetlatent.models.weights import checkpoint_family, checkpoint_meta, klein_dir, load_model, load_teacher
from budgetlatent.tokenization.layout import MODES, N_DENSE, LayoutMix, count_tokens, pack, select
from budgetlatent.training.data import make_loader
from budgetlatent.training.dmd import (DMD2DiscriminatorHead, DMDConfig, FakeScoreLoRA, coarse_weights, critic_loss,
                                       effective_scale, generator_loss, pooled_text, score_weights, set_trainable, synthesize)
from budgetlatent.training.trainlib import (N_PARAMS, Checkpoint, Clock, EmaShards, SpikeGuard, add_run_arguments, average_grads,
                                            broadcast_parameters, check_cadence, grad_norm, init_runtime, log, lr_scale,
                                            param_groups, plan_micro_batches, preflight, run_validation)

PHASES = ("data", "text", "rollout", "student", "critic", "optim")
TERMS = ("distill", "dmd", "gan", "score", "d_gan")


def main():
    ap = add_run_arguments(argparse.ArgumentParser())
    ap.add_argument("--init", default=None, help="stage-1 EMA weights to start from (default: dmd2_gan.warm_start_ema_ckpt; "
                                                 "neither = cold start from the released model)")
    ap.add_argument("--critic_batch", type=int, default=None, help="examples per critic pass (default: batch_per_gpu)")
    ap.add_argument("--teacher_ckpt_blocks", type=int, default=None, help="teacher blocks under gradient checkpointing in the "
                                                                          "critics' fake-score pass (default: all 25; 0 is faster "
                                                                          "and keeps ~7 GiB per example)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if cfg.get("model", {}).get("family", "flux2") != "flux2":
        raise SystemExit("stage 2 (DMD2 + GAN) is implemented for FLUX.2 [klein] only; FLUX 3 trains with scripts/train/train.py (stage 1)")
    dcfg, ocfg, gcfg = cfg["distillation"], cfg["optim"], cfg["dmd2_gan"]
    init = a.init or gcfg.get("warm_start_ema_ckpt") or None
    if init and os.path.isfile(init) and checkpoint_family(init) != "flux2":
        raise SystemExit(f"{init} is a {checkpoint_family(init)} checkpoint; stage 2 is implemented for FLUX.2 [klein] only")
    start = gcfg["warm_start" if init else "cold_start"]
    max_steps = a.max_steps or start.get("max_steps")
    if not max_steps:
        raise SystemExit("a warm start needs --max_steps (the recipe gives this mode no length)")
    dmd = DMDConfig.from_config(gcfg)
    check_cadence(a, cfg)

    rt = init_runtime()
    device, cuda, amp = rt.device, rt.cuda, rt.cuda
    torch.manual_seed(a.seed)                                      # random initialisations: the same on every rank
    os.makedirs(a.out, exist_ok=True)
    n_rank = a.batch_per_gpu * a.grad_accum
    gbs = n_rank * rt.world
    if gbs != int(ocfg["global_batch_size"]) and not a.allow_batch_mismatch:
        raise SystemExit(f"batch_per_gpu x grad_accum x GPUs = {gbs}, config global_batch_size = {ocfg['global_batch_size']} "
                         f"(pass --allow_batch_mismatch to run anyway)")

    # ---- models ------------------------------------------------------------------------------------------
    root = klein_dir(a.weights or cfg["model"]["flux2_base_repo"])
    student = load_model(init, root, device, bp=BPConfig.from_config(cfg.get("budgetpix_s8")))
    student.checkpoint_blocks = student.num_blocks if a.ckpt_blocks is None else max(0, min(a.ckpt_blocks, student.num_blocks))
    teacher = load_teacher(root, device, rt.frozen_dtype)
    # The teacher has three passes with a backward per step.  The generator's GAN pass runs while the student's
    # gradients are held, so it always checkpoints every block; the critics' two fake-score passes run after those
    # 14.5 GiB are freed and may keep their activations instead (--teacher_ckpt_blocks).
    ckpt_gan = teacher.num_blocks
    ckpt_score = ckpt_gan if a.teacher_ckpt_blocks is None else max(0, min(a.teacher_ckpt_blocks, ckpt_gan))
    vae = Flux2VAE(root, device).optimize()
    if init:
        trained_with = checkpoint_meta(init).get("chat_template") if os.path.isfile(init) else None
        template = template_name(a.template or trained_with or cfg["model"].get("chat_template"))
        if trained_with and template != template_name(trained_with):
            raise SystemExit(f"{init} was trained with the '{trained_with}' chat template, this run asks for '{template}'")
    else:
        template = template_name(a.template or cfg["model"].get("chat_template"))
    text = Qwen3Text(root, device, rt.frozen_dtype, template=template)
    empty_ids, empty_mask = (v.to(device) for v in text.empty_ids(1))
    critics = nn.ModuleDict({"lora": FakeScoreLoRA.for_trunk(teacher, dmd.lora_rank, dmd.lora_alpha),
                             "head": DMD2DiscriminatorHead.for_trunk(teacher, dmd)}).to(device)
    lora, head = critics["lora"], critics["head"]
    broadcast_parameters(rt, critics)
    critic_params = list(critics.parameters())

    # ---- optimizers, EMAs --------------------------------------------------------------------------------
    lr, lr_added = float(start["trunk_lr"]), float(start["adapter_lr"])
    groups, added, params = param_groups(student, lr, lr_added, is_adapter)
    opt = torch.optim.AdamW(groups, betas=tuple(ocfg["betas"]), eps=float(ocfg["eps"]),
                            weight_decay=float(ocfg["weight_decay"]), fused=cuda)
    opt_c = torch.optim.AdamW(critic_params, lr=dmd.critic_lr, betas=dmd.critic_betas, weight_decay=dmd.critic_weight_decay, fused=cuda)
    base_lrs = [g["lr"] for g in opt.param_groups]
    if rt.main and not a.skip_preflight:
        for line in preflight(student, teacher, text, rt, groups, N_PARAMS, released=init is None):
            log(line)
    rt.barrier()
    decays = tuple(float(d) for d in start["ema_decays"])
    ema = EmaShards([(n, p) for n, p in student.named_parameters() if p.requires_grad], rt.rank, rt.world,
                    decays, warmup=bool(start["ema_warmup"]))
    guard = SpikeGuard.from_config(ocfg)

    step, critic_step, ckpt = 0, 0, Checkpoint(rt, a.out)
    if a.resume and ckpt.exists():
        meta, files = ckpt.load(student, opt, ema, extra=("critic.pt",))
        critics.load_state_dict(files["critic.pt"]["critics"])
        opt_c.load_state_dict(files["critic.pt"]["optimizer"])
        step, critic_step, guard.state = int(meta["step"]), int(meta["critic_step"]), meta["guard"]
        if rt.main:
            log(f"[dmd] resumed from {ckpt.last} at step {step} (critic step {critic_step})")

    net = DDP(student, device_ids=[rt.local] if cuda else None, find_unused_parameters=False) if rt.ddp else student
    student.train()
    mix = LayoutMix.from_config(dcfg)
    num_steps, first_p = int(dcfg.get("data_num_steps", 50)), float(dcfg["rollout_first_p"])
    boost = float(dcfg["coarse_cell_boost"])
    prep_batch = min(a.prep_batch, n_rank)
    critic_batch = min(a.critic_batch or a.batch_per_gpu, n_rank)
    loader = iter(make_loader(a.data, root, prep_batch, a.num_workers, template, seed=a.seed + 7919 * step, rank=rt.rank,
                              world_size=rt.world, source_weights=cfg.get("data", {}).get("source_weights"), captions_only=True))
    gen = torch.Generator().manual_seed(a.seed + 1000 * rt.rank + 7919 * step)
    cap_tokens = a.batch_per_gpu * (int(cfg["model"]["text_seq_len"]) + N_DENSE)
    # a student pass here is followed by dense teacher passes on the same examples (two for the DMD term, one with
    # a backward for the GAN term), so it holds no more examples than the dense grid allows unless --max_micro says so
    cap_batch = a.max_micro or a.batch_per_gpu
    clock = Clock(PHASES, sync=a.profile and cuda)
    if rt.main:
        n_c = sum(p.numel() for p in critic_params)
        log(f"[dmd] {'warm start from ' + init if init else 'cold start from the released model'}; global batch {gbs} "
            f"({a.batch_per_gpu} x {a.grad_accum} x {rt.world}), steps {step} -> {max_steps}; lr {lr:.1e} trunk, {lr_added:.1e} added, "
            f"{dmd.critic_lr:.1e} critics ({n_c / 1e6:.2f}M parameters, {dmd.ttur_updates} updates per step); EMA decays {decays}"
            f"{' with warm-up' if start['ema_warmup'] else ''}; critic warm-up {start['critic_warmup_steps']} + ramp {start['dmd_ramp_steps']} "
            f"critic updates; template '{template}'")

    @torch.no_grad()
    def prepare(use_student):
        """This rank's training state of one step, from captions alone."""
        parts = []
        for _ in range(math.ceil(n_rank / prep_batch)):
            batch = next(loader)
            clock.lap("data")
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            b = ids.shape[0]
            drop = (torch.rand(b, generator=gen) < dcfg["class_dropout_prob"]).to(device)
            ids = torch.where(drop[:, None], empty_ids.expand(b, -1), ids)
            mask = torch.where(drop[:, None], empty_mask.expand(b, -1), mask)
            with rt.autocast():
                ctx = text.encode_ids(ids, mask)
            pooled = pooled_text(ctx, ids, PAD_ID)
            clock.lap("text")
            z_t, t, v_real, x0_real, active, mode = synthesize(student, teacher, vae, ctx, use_student, mix, dmd, gen, first_p, num_steps, amp)
            clock.lap("rollout")
            parts.append((z_t, t, ctx, pooled, v_real, x0_real, mode, active))
        out = [torch.cat([p[k] for p in parts])[:n_rank] for k in range(7)]
        return (*out, {s: torch.cat([p[7][s] for p in parts])[:n_rank] for s in parts[0][7]})

    def save(at_step):
        ckpt.save(student, opt, ema, {"step": at_step, "critic_step": critic_step, "guard": guard.state, "world": rt.world,
                                      "init": init, "args": vars(a), "cfg": cfg},
                  extra={"critic.pt": {"critics": critics.state_dict(), "optimizer": opt_c.state_dict()}})

    stats = torch.zeros(3 + len(TERMS), device=device)           # examples, tokens, student passes | sums of TERMS
    t0, step0, eff = time.time(), step, 0.0
    t_log, s_log = t0, step                                          # for the rate since the previous log line
    clock.last = t0
    while step < max_steps:
        scale = lr_scale(guard.state["accepted"], int(ocfg["warmup_steps"]))     # the warm-up counts accepted updates
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * scale
        # one coin per step for every rank: the student's own rollout, or the dense teacher's
        use_student = random.Random(1_000_003 * a.seed + step).random() < dmd.dmd_backward_sim_prob
        z_t, t, ctx, pooled, v_real, x0_real, mode, active = prepare(use_student)
        eff = effective_scale(critic_step, int(start["critic_warmup_steps"]), int(start["dmd_ramp_steps"]))

        # ---- the student ---------------------------------------------------------------------------------
        set_trainable(False, critics)
        teacher.checkpoint_blocks = ckpt_gan
        plan = plan_micro_batches(count_tokens(active).tolist(), ctx.shape[1], cap_tokens, cap_batch, not a.no_bucket, a.batch_per_gpu)
        x0_fake = torch.empty_like(z_t)
        opt.zero_grad(set_to_none=True)
        for j, grp in enumerate(plan):
            idx = torch.tensor(grp, device=device)
            lay = pack(select(active, idx))
            with (net.no_sync() if rt.ddp and j < len(plan) - 1 else nullcontext()):
                with rt.autocast():
                    v = net(z_t[idx], t[idx], ctx[idx], lay)
                terms, x0_fake[idx] = generator_loss(v.float(), z_t[idx], t[idx], v_real[idx], coarse_weights(lay.cell_scale, boost),
                                                     ctx[idx], pooled[idx], teacher, lora, head, dmd, eff, gen, amp)
                per = terms["distill"]
                if eff > 0.0:
                    per = per + eff * (dmd.dmd_weight * terms["dmd"] + dmd.gan_weight * terms["gan"])
                # every added tensor stays in the graph whatever the layout, so DDP sees the same set on every rank
                (per.sum() / n_rank + 0.0 * sum(p.reshape(-1)[0] for p in added)).backward()
            with torch.no_grad():
                stats[0] += len(grp)
                stats[1] += lay.n.sum()
                stats[2] += 1
                for k, name in enumerate(TERMS[:3]):
                    if name in terms:
                        stats[3 + k] += terms[name].sum()
        clock.lap("student")
        gnorm = rt.reduce(grad_norm(params), "max")
        gn = float(gnorm)
        if guard.reject(gn):
            if rt.main:
                log(f"[guard] step {step}: gradient norm {gn:.3g} (running mean {guard.state['ema']:.3g}) -- step skipped")
        else:
            torch.nn.utils.clip_grads_with_norm_(params, float(ocfg["grad_clip_norm"]), gnorm[0])
            opt.step()
            ema.update(step)
        opt.zero_grad(set_to_none=True)                              # 14.5 GiB back before the critics' passes
        clock.lap("optim")

        # ---- the critics ---------------------------------------------------------------------------------
        set_trainable(True, critics)
        teacher.checkpoint_blocks = ckpt_score
        w_score = score_weights(t, dmd.fake_score_match_dmd)
        for _ in range(dmd.ttur_updates):
            opt_c.zero_grad(set_to_none=True)
            for lo in range(0, n_rank, critic_batch):
                sl = slice(lo, lo + critic_batch)
                terms = critic_loss(x0_fake[sl], x0_real[sl], t[sl], w_score[sl], ctx[sl], pooled[sl], teacher, lora, head, dmd, gen, amp)
                ((terms["score"] + dmd.d_gan_loss_weight * terms["gan"]).sum() / n_rank).backward()
                with torch.no_grad():
                    stats[6] += terms["score"].sum() / dmd.ttur_updates
                    stats[7] += terms["gan"].sum() / dmd.ttur_updates
            average_grads(rt, critic_params)
            torch.nn.utils.clip_grad_norm_(critic_params, dmd.critic_grad_clip_norm)
            opt_c.step()
            critic_step += 1
        opt_c.zero_grad(set_to_none=True)
        clock.lap("critic")
        step += 1

        if step % a.log_every == 0 or step == step0 + 1 or step == max_steps:
            s = rt.reduce(stats.clone())
            if rt.main:
                n, done = max(1.0, float(s[0])), step - step0
                sec, recent = (time.time() - t0) / done, (time.time() - t_log) / max(1, step - s_log)
                t_log, s_log = time.time(), step
                losses = "  ".join(f"{name} {float(s[3 + k]) / n:.4f}" for k, name in enumerate(TERMS))
                log(f"step {step:6d}  {losses}  scale {eff:.3f}  rollout {'student' if use_student else 'teacher'}  "
                    f"tokens {float(s[1]) / n:5.0f}  gnorm {gn:.3f}  lr {base_lrs[-1] * scale:.2e}  skipped {guard.state['skipped']}  "
                    f"critic step {critic_step}  {sec:.2f} s/step (last {recent:.2f})  {gbs / sec:.1f} img/s  mem {rt.peak_gib():.1f}G"
                    + (f"  eta {(max_steps - step) * sec / 3600:.1f} h" if done >= 3 else ""))
                if a.profile:
                    log(f"         seconds per step on rank 0: {clock.report(done)}")
            stats.zero_()
        if not a.no_save and (step % a.save_every == 0 or step == max_steps):
            save(step)
            if step % a.snapshot_every == 0 or step == max_steps:
                tag = "final" if step == max_steps else f"s{step}"
                weights = os.path.join(a.out, f"ema-{tag}.safetensors")
                if rt.main:
                    for k, decay in enumerate(decays):               # the first decay is "the" EMA; the others are named by theirs
                        ckpt.export_ema(student, weights if k == 0 else os.path.join(a.out, f"ema{decay:g}-{tag}.safetensors"), k,
                                        metadata={"step": step, "stage": 2, "ema_decay": decay, "init": init or "released",
                                                  "base": cfg["model"]["flux2_base_repo"], "format": "budgetpix_flux2_s8",
                                                  "chat_template": template})
                rt.barrier()
                if a.val_every and (step % a.val_every == 0 or step == max_steps):
                    run_validation(rt, a, cfg, weights, root, vae, text, step)
            clock.last = t_log = time.time()                         # saving and validating are not part of a step's time
            s_log = step

    if rt.main:
        log("[dmd] done" + ("" if a.no_save else f": {os.path.join(a.out, 'ema-final.safetensors')}"))
    rt.close()


if __name__ == "__main__":
    main()
