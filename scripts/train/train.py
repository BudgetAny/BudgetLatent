"""Stage 1: distil a released FLUX model into BudgetLatent -- real-image paired teacher-velocity distillation.

    torchrun --nproc_per_node=4 scripts/train/train.py --config configs/flux2_512.yaml --data data/mix120k/latents_512 \
        --out runs/bpflux2_s8 --batch_per_gpu 2 --grad_accum 16 --resume
    BP_GRID=64 torchrun --nproc_per_node=4 scripts/train/train.py --config configs/flux3_1024.yaml \
        --data data/mix120k/latents_flux3_1024 --out runs/bpflux3_s8_1024 --batch_per_gpu 2 --grad_accum 16 --resume

The model family comes from the config (model.family): FLUX.2 [klein] 4B Base (configs/flux2_512.yaml,
configs/flux2_1024.yaml) or FLUX 3 Image Klein Base (configs/flux3_512.yaml, configs/flux3_1024.yaml).  Both predict
the flow velocity eps - x, so the target is the dense teacher's velocity for either; FLUX 3 differs in the schedule
(constant shift 6.93), the text features (Qwen3-VL, a text window per caption, carried per example) and the trunk
(models/flux3_trunk.py).

One training example: a clean latent is noised to a time t, the frozen dense teacher predicts the velocity, the
image the teacher implies is decoded and its entropy decides a token layout (one of four modes), and the student
is run on that layout and regressed onto the teacher's velocity (training/stage1.py prepares them).  Every weight is
trained, the added modules 8x faster; with the dense layout the student starts as the released model, which is
asserted before the first step (trainlib.preflight).  Every optim.ckpt_every steps the EMA is written as weights and
the validation suite (32 prompts x 10 budgets) is rendered with it into <out>/val-s<step>/.

What a step costs, and what is done about it:
  * the teacher, the text encoder and the VAE run without gradients, in batches of --prep_batch;
  * the VAE encoder is skipped when --data holds latents (scripts/data/encode_latents.py; the preprocessing has no
    augmentation, so they are exact), and the VAE decoder only runs for the examples whose layout reads the image;
  * the student's examples are sorted by token count and run in groups of similar length, so little is padded and
    short sequences run more at a time; gradients accumulate and are synchronised once per step;
  * gradient checkpointing covers the first --ckpt_blocks blocks (all by default; fewer is faster and takes more
    memory).

Resolution: 512 x 512 by default; BP_GRID=64 with a *_1024.yaml config for 1024 x 1024, best started from a 512 run
with --init <its checkpoint-last directory> (the fp32 EMA).

Memory (per GPU, DDP, FLUX.2): fp32 student + gradients + AdamW moments = 4 x 14.5 GiB, bf16 teacher 7.2 GiB, Qwen3
5.8 GiB, plus this rank's share of the EMA; --shard_optim, --bucket_view and --offload_frozen bring that down.
Checkpoints are ~62 GB at 4B (model, optimizer, EMA): point --out at a large disk.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))    # the repository root
import time
from contextlib import contextmanager, nullcontext

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from budgetlatent.encoders.text import F3_MAX_TOKENS, template_name, text_encoder
from budgetlatent.encoders.vae import Flux2VAE
from budgetlatent.models import flux2_trunk
from budgetlatent.models.convert import is_adapter
from budgetlatent.models.budgetpix import BPConfig
from budgetlatent.models.families import family_of
from budgetlatent.models.weights import expected_counts, keep_released, klein_dir, load_model, load_teacher
from budgetlatent.tokenization.layout import CELL_PX, GRID, MODES, N_DENSE, LayoutMix, count_tokens, pack, select, time_shift
from budgetlatent.training.data import make_loader
from budgetlatent.training.stage1 import make_prepare
from budgetlatent.training.trainlib import (N_PARAMS, Checkpoint, Clock, EmaShards, Parked, ShardedOptimizer, SpikeGuard,
                                            add_run_arguments, check_cadence, grad_norm, init_runtime, load_init, log,
                                            lr_scale, param_groups, plan_micro_batches, prebuild_ddp_buckets, preflight,
                                            run_validation)

PHASES = ("data", "encode", "text", "teacher", "layout", "student", "optim")


def main():
    ap = add_run_arguments(argparse.ArgumentParser())
    ap.add_argument("--compile_teacher", action="store_true", help="torch.compile the frozen teacher's blocks")
    ap.add_argument("--teacher_attention", default=None, choices=["cudnn"],
                    help="the teacher's attention kernel: PyTorch's choice (FlashAttention, default) or cuDNN's "
                         "(flux2_trunk.SDPA_BACKEND during the teacher's forward only)")
    ap.add_argument("--compile_student", action="store_true",
                    help="torch.compile the student's blocks with a symbolic batch / sequence length (fast.compile_blocks)")
    ap.add_argument("--varlen_attention", action="store_true",
                    help="padded student passes: one unmasked FlashAttention call per image instead of one masked call "
                         "(flux2_trunk.VARLEN_ATTENTION)")
    ap.add_argument("--rope_complex", action="store_true",
                    help="eager (uncompiled) blocks: the rotary products as one complex multiply (flux2_trunk.ROPE_COMPLEX)")
    ap.add_argument("--freeze_trunk", action="store_true", help="train only the 28M added parameters")
    ap.add_argument("--unfreeze", default="", help="with --freeze_trunk: comma-separated name prefixes of trunk parameters "
                                                    "to train anyway (memory-limited tests of trunk gradients)")
    ap.add_argument("--fast_prep", action="store_true",
                    help="no host synchronisation while a step is prepared (the same targets, layouts and random "
                         "stream; make_prepare)")
    ap.add_argument("--vae_chunk", type=int, default=None, help="images per VAE decoder pass (default 4)")
    ap.add_argument("--vae_nchw", action="store_true",
                    help="VAE convolutions in NCHW instead of channels-last (eager channels-last group norms copy the "
                         "activations between layouts: 2x slower at 1024 x 1024)")
    ap.add_argument("--vae_bf16", action="store_true",
                    help="detail maps from a bf16 copy of the VAE decoder, group norms included (vae.bf16_preview), "
                         "instead of bf16 autocast over the fp32 decoder")
    ap.add_argument("--compile_vae", action="store_true",
                    help="torch.compile the VAE decoder of the detail maps (with --vae_bf16: its bf16 copy only)")
    ap.add_argument("--bucket_view", action="store_true",
                    help="DDP: the gradients live in its all-reduce buckets (gradient_as_bucket_view) and are zeroed in "
                         "place, instead of a second copy (-14.5 GiB per GPU at 4B; the same numbers)")
    ap.add_argument("--shard_optim", action="store_true",
                    help="ZeRO-1: each GPU keeps the AdamW moments of its share of the tensors (the EMA's split) and "
                         "broadcasts them after the step (-29 GiB x (1 - 1 / GPUs) per GPU at 4B; the same numbers); "
                         "checkpoints hold optimizer-<rank>.pt, and either layout resumes into either")
    ap.add_argument("--no_cast_cache", action="store_true",
                    help="the student's forward without autocast's cache of bf16 weight copies, which otherwise keeps a "
                         "copy of every weight -- checkpointed blocks too -- until the forward ends (up to 7.2 GiB at 4B); "
                         "each weight is cast once per pass either way (the same numbers, the same time)")
    ap.add_argument("--offload_frozen", action="store_true",
                    help="keep the frozen teacher and text encoder in pinned host memory outside the step's preparation "
                         "(trainlib.Parked): their GPU memory goes to the student's passes, for one host-to-device copy of "
                         "their weights per step (FLUX 3: 9.1 + 6.7 GiB)")
    ap.add_argument("--prebuild_buckets", action="store_true",
                    help="DDP: do its one-time gradient-bucket rebuild at start-up (trainlib.prebuild_ddp_buckets) instead of "
                         "in the second step, where it costs one more fp32 gradient copy at once")
    ap.add_argument("--probe_world", type=int, default=None,
                    help="MEASUREMENT ONLY (with --shard_optim --no_save): shard the AdamW moments and the EMA as if over this "
                         "many GPUs, so a 2-GPU run holds what one GPU of a probe_world-GPU run holds; the shards of the missing "
                         "ranks are never updated, so the training itself is wrong")
    ap.add_argument("--accept_latents", action="store_true",
                    help="train on latents another family's VAE encoded (FLUX.2's VAE file is FLUX 3's rounded to bf16; "
                         "tests/sanity_flux3.py check 4 measures how far its latents are)")
    ap.add_argument("--init", default=None,
                    help="start from these weights instead of the released model: a stage-1 checkpoint directory (its "
                         "fp32 EMA shards, else model.pt), a .pt state dict or a .safetensors file; ignored on --resume "
                         "when <out> has a checkpoint")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    res = int(cfg["model"].get("resolution", 512))
    if res != GRID * CELL_PX:
        raise SystemExit(f"{a.config}: resolution {res}, but the layout grid is {GRID} cells = {GRID * CELL_PX} px: "
                         f"run with BP_GRID={res // CELL_PX}")
    dcfg, ocfg = cfg["distillation"], cfg["optim"]
    max_steps = a.max_steps or int(ocfg["max_steps"])
    family = cfg["model"].get("family", "flux2")
    fam = family_of(family)
    template = template_name(a.template or cfg["model"].get("chat_template") or fam.template)
    max_tokens = int(cfg["model"].get("text_max_tokens", F3_MAX_TOKENS))
    check_cadence(a, cfg)

    first = sorted(glob.glob(os.path.join(a.data, "chunk_*.pt")))[:1]
    if first and "latent" in torch.load(first[0], map_location="cpu", mmap=True, weights_only=False):
        tag = os.path.join(a.data, ".family")                       # scripts/data/encode_latents.py writes it
        data_family = open(tag).read().strip() if os.path.exists(tag) else "flux2"
        if data_family != family and not a.accept_latents:
            raise SystemExit(f"{a.data}: latents of {data_family}'s VAE, the model is {family} (FLUX 3 has a VAE of its own: "
                             f"scripts/data/encode_latents.py --family {family})")
    rt = init_runtime()
    device, cuda, amp = rt.device, rt.cuda, rt.cuda                # without a GPU everything runs fp32 on a CPU (dry runs)
    torch.manual_seed(a.seed)                                      # the added modules' random parts: the same on every rank
    os.makedirs(a.out, exist_ok=True)
    n_rank = a.batch_per_gpu * a.grad_accum
    gbs = n_rank * rt.world
    if gbs != int(ocfg["global_batch_size"]) and not a.allow_batch_mismatch:
        raise SystemExit(f"batch_per_gpu x grad_accum x GPUs = {gbs}, config global_batch_size = {ocfg['global_batch_size']} "
                         f"(pass --allow_batch_mismatch to run anyway)")

    # ---- models ------------------------------------------------------------------------------------------
    root = klein_dir(a.weights or cfg["model"].get("base_repo") or cfg["model"].get("flux2_base_repo") or fam.repo)
    bp = BPConfig.from_config(cfg.get("budgetpix_s8"))
    keep_released()                                                  # the student and the teacher from one read of the file
    student = load_model(None, root, device, bp=bp, family=family)
    resuming = a.resume and os.path.exists(os.path.join(a.out, "checkpoint-last", "meta.json"))
    init = a.init if a.init and not resuming else None
    if init:
        n = load_init(student, init)
        if rt.main:
            log(f"[train] {n} tensors initialised from {init}")
    student.checkpoint_blocks = student.num_blocks if a.ckpt_blocks is None else max(0, min(a.ckpt_blocks, student.num_blocks))
    teacher = load_teacher(root, device, rt.frozen_dtype, family=family)
    keep_released(False)
    vae = Flux2VAE(root, device).optimize(compile_decoder=a.compile_vae and not a.vae_bf16, channels_last=not a.vae_nchw)
    if a.vae_bf16 and cuda:
        vae.bf16_preview(compile=a.compile_vae)
    if a.vae_chunk:
        vae.chunk = a.vae_chunk
    text = text_encoder(family, root, device, rt.frozen_dtype, template=template, max_tokens=max_tokens)
    empty_ids, empty_mask = text.empty_ids(1, pad_to=max_tokens if family == "flux3" else None)       # on the host
    if a.freeze_trunk:
        keep = tuple(x for x in a.unfreeze.split(",") if x)
        for n, p in student.trunk_named_parameters():
            p.requires_grad_(bool(keep) and n.startswith(keep))
        if keep:
            n_un = sum(p.numel() for _, p in student.trunk_named_parameters() if p.requires_grad)
            if not n_un:
                raise SystemExit(f"--unfreeze {a.unfreeze}: no trunk parameter has those prefixes")
            log(f"[train] trunk frozen except {a.unfreeze} ({n_un / 1e6:.1f}M parameters)")

    # ---- optimizer, EMA -----------------------------------------------------------------------------------
    lr = float(ocfg["trunk_lr"])
    lr_added = float(ocfg["adapter_lr"]) if "adapter_lr" in ocfg else lr * float(ocfg["adapter_lr_mult"])
    groups, added, params = param_groups(student, lr, lr_added, is_adapter)
    trained = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    adamw = lambda gs: torch.optim.AdamW(gs, betas=tuple(ocfg["betas"]), eps=float(ocfg["eps"]),
                                         weight_decay=float(ocfg["weight_decay"]), fused=cuda)
    probe = a.probe_world if a.probe_world and a.probe_world > rt.world else None
    if probe and not (a.shard_optim and a.no_save and rt.ddp):
        raise SystemExit("--probe_world is a memory measurement: it needs --shard_optim, --no_save and more than one GPU")
    opt = ShardedOptimizer(groups, trained, rt.rank, probe or rt.world, adamw, comm_world=rt.world) if a.shard_optim and rt.ddp \
        else adamw(groups)
    base_lrs = [g["lr"] for g in opt.param_groups]
    if rt.main and not a.skip_preflight:
        expect = N_PARAMS if family == "flux2" else expected_counts(family, root, bp)
        for line in preflight(student, teacher, text, rt, None if a.freeze_trunk else groups, expect, released=init is None):
            log(line)
    rt.barrier()
    if a.compile_teacher or a.compile_student:
        from budgetlatent.inference.fast import compile_blocks
        if a.compile_teacher:
            compile_blocks(teacher)
        if a.compile_student:
            compile_blocks(student, dynamic=True)
    trunk = flux2_trunk                                              # the attention switches of both trunks
    trunk.VARLEN_ATTENTION = trunk.VARLEN_ATTENTION or a.varlen_attention
    trunk.ROPE_COMPLEX = trunk.ROPE_COMPLEX or a.rope_complex
    if a.teacher_attention:

        @contextmanager
        def teacher_attention():
            """trunk.SDPA_BACKEND for the teacher's forward only (it is never masked and has one shape; the student's
            shapes change every pass, and cuDNN builds a plan per shape).  Compiled code guards on the value, so the
            teacher's and the student's graphs each keep their own."""
            old, trunk.SDPA_BACKEND = trunk.SDPA_BACKEND, a.teacher_attention
            try:
                yield
            finally:
                trunk.SDPA_BACKEND = old
    else:
        teacher_attention = nullcontext
    ema = EmaShards(trained, rt.rank, probe or rt.world, (float(ocfg["ema_decay"]),), warmup=True)
    if probe and rt.main:
        log(f"[train] MEMORY PROBE: moments and EMA sharded as over {probe} GPUs on {rt.world}; 1 - {rt.world}/{probe} of the "
            f"tensors are never updated -- for measurements only")
    guard = SpikeGuard.from_config(ocfg)

    step, ckpt = 0, Checkpoint(rt, a.out)
    if a.resume and ckpt.exists():
        meta, _ = ckpt.load(student, opt, ema)
        step, guard.state = int(meta["step"]), meta["guard"]
        if rt.main:
            log(f"[train] resumed from {ckpt.last} at step {step}")

    # parked before DDP is built: its start-up bucket rebuild (--prebuild_buckets) then has their memory too
    parked = [Parked(teacher, device), Parked(text.model, device)] if a.offload_frozen and cuda else []
    if parked and rt.main:
        log(f"[train] teacher and text encoder parked in pinned host memory between preparations "
            f"({sum(m.bytes for m in parked) / 2 ** 30:.1f} GiB per step host -> device)")
    keep_grads = rt.ddp and a.bucket_view                          # the gradients are DDP's buckets: zeroed, never freed
    if keep_grads:                                                 # gradients that exist when DDP is built become views
        for p in params:                                           # of its buckets right away (not after a first step,
            p.grad = torch.zeros_like(p)                           # which would need room for both copies once)
    net = DDP(student, device_ids=[rt.local] if cuda else None, find_unused_parameters=False,
              gradient_as_bucket_view=keep_grads) if rt.ddp else student
    if rt.ddp and a.prebuild_buckets:
        before = torch.cuda.memory_allocated(device) / 2 ** 30 if cuda else 0.0
        done = prebuild_ddp_buckets(net)
        if rt.main:
            log(f"[train] DDP buckets rebuilt at start-up ({done}); allocated {before:.1f} -> "
                f"{(torch.cuda.memory_allocated(device) / 2 ** 30) if cuda else 0.0:.1f} GiB, peak so far {rt.peak_gib():.1f} GiB")
    student.train()
    mix = LayoutMix.from_config(dcfg)
    num_steps = int(dcfg.get("data_num_steps", dcfg.get("time_shift_steps", 50)))
    shift = float(cfg["model"].get("time_shift", time_shift(family, num_steps))) if family != "flux2" else None
    boost = float(dcfg["coarse_cell_boost"])
    # "mean": mean squared error over cells and channels; "sum": squared L2 norm per cell (128x larger).  AdamW does not
    # care about the factor, the gradient clip and the spike guard (absolute thresholds) do.  The logged loss is the mean.
    loss_gain = float(cfg["model"]["latent_channels"]) if str(dcfg.get("loss_channel_reduction", "mean")) == "sum" else 1.0
    prep_batch = min(a.prep_batch, n_rank)
    loader = iter(make_loader(a.data, root, prep_batch, a.num_workers, template, seed=a.seed + 7919 * step, rank=rt.rank,
                              world_size=rt.world, source_weights=cfg.get("data", {}).get("source_weights"), max_tokens=max_tokens))
    gen = torch.Generator().manual_seed(a.seed + 1000 * rt.rank + 7919 * step)
    cap_tokens = a.batch_per_gpu * (int(cfg["model"]["text_seq_len"]) + N_DENSE)
    cap_batch = a.max_micro or 4 * a.batch_per_gpu
    clock = Clock(PHASES, sync=a.profile and cuda)
    student_autocast = (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cuda, cache_enabled=False)) \
        if a.no_cast_cache else rt.autocast
    if rt.main:
        n_add, n_all = sum(p.numel() for p in added), sum(p.numel() for p in student.parameters())
        log(f"[train] global batch {gbs} ({a.batch_per_gpu} x {a.grad_accum} x {rt.world}), steps {step} -> {max_steps}; "
            f"{family}, {n_all / 1e6:.1f}M parameters, {n_add / 1e6:.2f}M added; lr {lr:.1e} trunk{' (frozen)' if a.freeze_trunk else ''}, "
            f"{lr_added:.1e} added; template '{template}'; checkpointing {student.checkpoint_blocks} / {student.num_blocks} blocks; "
            f"EMA{' and AdamW moments' if isinstance(opt, ShardedOptimizer) else ''} on {rt.world} rank(s); "
            f"{'gradients in the DDP buckets; ' if keep_grads else ''}snapshots every {a.snapshot_every}, validation every {a.val_every or 'never'}")

    prepare = make_prepare(loader, text, teacher, vae, gen, mix, dcfg, empty_ids, empty_mask, n_rank, prep_batch, num_steps,
                           rt, clock, data_dir=a.data, fast=a.fast_prep, decode_bf16="weights" if a.vae_bf16 else None,
                           teacher_context=teacher_attention, shift=shift, parked=parked)

    def save(at_step):
        ckpt.save(student, opt, ema, {"step": at_step, "guard": guard.state, "world": rt.world, "args": vars(a), "cfg": cfg})

    stats = torch.zeros(4 + 3 * len(MODES), device=device)       # loss sum, examples, tokens, passes | per mode: loss, n, tokens
    t0, step0 = time.time(), step
    t_log, s_log = t0, step                                          # for the rate since the previous log line
    clock.last = t0
    while step < max_steps:
        scale = lr_scale(step, int(ocfg["warmup_steps"]))
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * scale
        z_t, t, ctx, v_t, active, mode, lens = prepare()
        plan = plan_micro_batches(count_tokens(active).tolist(), ctx.shape[1] if lens is None else lens.tolist(), cap_tokens, cap_batch,
                                  not a.no_bucket, a.batch_per_gpu)
        opt.zero_grad(set_to_none=not keep_grads)
        for j, grp in enumerate(plan):
            idx = torch.tensor(grp, device=device)
            lay = pack(select(active, idx))
            lg = None if lens is None else lens[torch.tensor(grp)]
            cg = ctx[idx] if lg is None else ctx[idx][:, :int(lg.max())]
            with (net.no_sync() if rt.ddp and j < len(plan) - 1 else nullcontext()):
                with student_autocast():
                    v = net(z_t[idx], t[idx], cg, lay, txt_len=None if lg is None else tuple(lg.tolist()))   # host ints: DDP moves tensors
                w = 1.0 + boost * (lay.cell_scale > 1).float()
                w = w / w.mean(dim=(1, 2), keepdim=True)
                per = (w[..., None] * (v.float() - v_t[idx]).pow(2)).mean(dim=(1, 2, 3))
                # every added tensor stays in the graph whatever the layout, so DDP sees the same set on every rank
                (loss_gain * per.sum() / n_rank + 0.0 * sum(p.reshape(-1)[0] for p in added)).backward()
            with torch.no_grad():
                stats[0] += per.sum()
                stats[1] += len(grp)
                stats[2] += lay.n.sum()
                stats[3] += 1
                for k in range(len(MODES)):
                    m = mode[idx] == k
                    stats[4 + 3 * k] += per[m].sum()
                    stats[5 + 3 * k] += m.sum()
                    stats[6 + 3 * k] += lay.n[m].sum()
        clock.lap("student")
        gnorm = rt.reduce(grad_norm(params), "max")                 # one number for every rank: the same decision everywhere
        gn = float(gnorm)
        if guard.reject(gn):
            if rt.main:
                log(f"[guard] step {step}: gradient norm {gn:.3g} (running mean {guard.state['ema']:.3g}) -- step skipped")
        else:
            torch.nn.utils.clip_grads_with_norm_(params, float(ocfg["grad_clip_norm"]), gnorm[0])
            opt.step()
        ema.update(step)
        clock.lap("optim")
        step += 1
        if step == step0 + 1 or step % a.save_every == 0:          # the replicas must stay identical (DDP)
            if isinstance(opt, ShardedOptimizer):
                opt.wait()                                          # its broadcasts finish the step's update
            if not rt.in_sync(params):
                raise SystemExit(f"step {step}: the weights differ across ranks -- gradients are not being synchronised")
            if rt.main and step == step0 + 1:
                log(f"[train] step {step}: weights identical on all {rt.world} rank(s)")

        if step % a.log_every == 0 or step == step0 + 1 or step == max_steps:
            s = rt.reduce(stats.clone())
            if rt.main:
                n, done = max(1.0, float(s[1])), step - step0
                per_mode = "  ".join(f"{MODES[k]} {float(s[4 + 3 * k]) / max(1.0, float(s[5 + 3 * k])):.4f}" for k in range(len(MODES)))
                sec, recent = (time.time() - t0) / done, (time.time() - t_log) / max(1, step - s_log)
                t_log, s_log = time.time(), step
                log(f"step {step:6d}  loss {float(s[0]) / n:.4f}  [{per_mode}]  tokens {float(s[2]) / n:5.0f}  "
                    f"passes/GPU {float(s[3]) / rt.world / max(1, min(a.log_every, done)):.1f}  gnorm {gn:.3f}  "
                    f"lr {base_lrs[-1] * scale:.2e}  skipped {guard.state['skipped']}  {sec:.2f} s/step (last {recent:.2f})  {gbs / sec:.1f} img/s  "
                    f"mem {rt.peak_gib():.1f}G (reserved {rt.peak_reserved_gib():.1f}G)" + (f"  eta {(max_steps - step) * sec / 3600:.1f} h" if done >= 3 else ""))
                if a.profile:
                    log(f"         seconds per step on rank 0: {clock.report(done)}")
            stats.zero_()
        if not a.no_save and (step % a.save_every == 0 or step == max_steps):
            save(step)
            if step % a.snapshot_every == 0 or step == max_steps:
                weights = os.path.join(a.out, "ema-final.safetensors" if step == max_steps else f"ema-s{step}.safetensors")
                if rt.main:
                    ckpt.export_ema(student, weights, metadata={"step": step, "stage": 1, "base": fam.repo, "family": family,
                                                                 "format": f"budgetpix_{family}_s8", "chat_template": template,
                                                                 "resolution": GRID * CELL_PX, "init": a.init or ""})
                rt.barrier()
                if a.val_every and (step % a.val_every == 0 or step == max_steps):
                    if not keep_grads:                               # (bucket gradients would only come back as a copy)
                        opt.zero_grad(set_to_none=True)              # the gradients' 14.5 GiB make room for the EMA model
                    for m in parked[1:]:
                        m.load()                                     # the text encoder renders the prompts
                    run_validation(rt, a, cfg, weights, root, vae, text, step, family)
                    for m in parked[1:]:
                        m.park()
            clock.last = t_log = time.time()                         # saving and validating are not part of a step's time
            s_log = step

    if isinstance(opt, ShardedOptimizer):
        opt.wait()                                                   # the last step's broadcasts
    if rt.main:
        log("[train] done" + ("" if a.no_save else f": {os.path.join(a.out, 'ema-final.safetensors')}"))
    rt.close()


if __name__ == "__main__":
    main()
