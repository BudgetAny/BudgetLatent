"""What the two trainers share (scripts/train/train.py: stage 1, scripts/train/train_dmd.py: stage 2).

  Runtime, init_runtime     the process group, this rank's device, bf16 autocast (everything is fp32 on a CPU)
  lr_scale                  linear warm-up, then constant
  SpikeGuard                skip an update whose gradient norm is an outlier
  EmaShards                 the EMA of the trained tensors, each rank keeping 1 / world of it
  Checkpoint                checkpoint-last/ (model, optimizer, EMA shards, meta), written atomically; EMA export
  average_grads             gradient mean over the ranks for modules outside DDP (the stage-2 critics)
  plan_micro_batches        a step's examples grouped into student passes of similar token count
  Clock                     wall time per phase of a step
  ShardedOptimizer          ZeRO stage 1 beside DDP: each rank steps the AdamW state of its share of the tensors
  Parked                    a frozen model's weights in pinned host memory while the student trains (--offload_frozen)
  prebuild_ddp_buckets      DDP's one-time bucket rebuild done at start-up, before the big allocations
  preflight                 the start-up assertions that need no data
  add_run_arguments, check_cadence, run_validation, load_init
                            the command line, the snapshot cadence, the validation call and --init
"""
import datetime
import glob
import json
import math
import os
import shutil
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Runtime:
    world: int
    rank: int
    local: int
    device: torch.device
    cuda: bool

    @property
    def main(self):
        return self.rank == 0

    @property
    def ddp(self):
        return self.world > 1

    @property
    def frozen_dtype(self):             # the frozen models' weights: bf16 on a GPU, fp32 on a CPU (dry runs)
        return torch.bfloat16 if self.cuda else torch.float32

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.cuda)

    def barrier(self):
        if self.ddp:
            dist.barrier()

    def reduce(self, x, op="sum"):
        """In place over the ranks: "sum", "max" or "mean"."""
        if self.ddp:
            dist.all_reduce(x, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if op == "mean":
                x /= self.world
        return x

    def in_sync(self, params, n=48):
        """Whether every rank holds the same weights: per-tensor sums of up to n spread-out trained tensors, compared
        exactly across ranks (DDP keeps the replicas bit-identical, whatever the kernels).  True on one rank."""
        if not self.ddp:
            return True
        pick = params[:: max(1, len(params) // n)]
        fp = torch.stack([p.detach().double().sum() for p in pick])
        hi, lo = fp.clone(), -fp
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        dist.all_reduce(lo, op=dist.ReduceOp.MAX)
        return bool(torch.equal(hi, -lo))

    def peak_gib(self):
        return torch.cuda.max_memory_allocated(self.device) / 2 ** 30 if self.cuda else 0.0

    def peak_reserved_gib(self):
        """The caching allocator's peak: what the card has to hold (allocated + cached + fragmentation)."""
        return torch.cuda.max_memory_reserved(self.device) / 2 ** 30 if self.cuda else 0.0

    def close(self):
        if self.ddp:
            dist.destroy_process_group()


def init_runtime() -> Runtime:
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available()
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo", timeout=datetime.timedelta(hours=2))
        if cuda:
            torch.cuda.set_device(local)
    return Runtime(world, rank, local, torch.device(f"cuda:{local}" if cuda else "cpu"), cuda)


# ---------------------------------------------------------------------------------------------------------
# optimisation
# ---------------------------------------------------------------------------------------------------------
def lr_scale(step, warmup, init=1e-4):
    """Linear warm-up from `init` x the rate, then constant."""
    return 1.0 if step >= warmup else init + (1.0 - init) * step / max(1, warmup)


def param_groups(student, trunk_lr, adapter_lr, is_adapter):
    """AdamW groups: the added modules at adapter_lr, the released trunk at trunk_lr (absent if it is frozen).
    Returns (groups, added parameters, all trained parameters)."""
    trainable = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    added = [p for n, p in trainable if is_adapter(n)]
    trunk = [p for n, p in trainable if not is_adapter(n)]
    groups = [{"params": added, "lr": adapter_lr}] + ([{"params": trunk, "lr": trunk_lr}] if trunk else [])
    return groups, added, [p for _, p in trainable]


def grad_norm(params):
    return torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None]).reshape(1)


def average_grads(rt, params):
    """Mean over the ranks of the gradients of modules that are not wrapped in DDP (the stage-2 critics)."""
    if not rt.ddp:
        return
    flat = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1) for p in params])
    rt.reduce(flat, "mean")
    at = 0
    for p in params:
        p.grad = flat[at:at + p.numel()].view_as(p).clone()
        at += p.numel()


def broadcast_parameters(rt, module):
    """Rank 0's values on every rank (modules created with random weights outside DDP)."""
    if rt.ddp:
        for p in module.parameters():
            dist.broadcast(p.data, src=0)


class SpikeGuard:
    """Reject an update whose gradient norm is not finite, above max_norm, or -- after `warmup` accepted updates --
    above factor x the running mean (decay 0.98) of the accepted norms.  GradSpikeGuard, as a decision
    on a number (the caller clips and steps); `state` goes into the checkpoint."""

    def __init__(self, factor=10.0, max_norm=50.0, warmup=50, decay=0.98, state=None):
        self.factor, self.max_norm, self.warmup, self.decay = float(factor), float(max_norm), int(warmup), float(decay)
        self.state = state or {"ema": 0.0, "accepted": 0, "skipped": 0}

    @classmethod
    def from_config(cls, ocfg, state=None):
        return cls(ocfg["spike_guard_factor"], ocfg["spike_guard_max_norm"], ocfg.get("spike_guard_warmup", 50), state=state)

    def reject(self, gnorm: float) -> bool:
        s = self.state
        bad = not math.isfinite(gnorm) or gnorm > self.max_norm \
            or (s["accepted"] >= self.warmup and gnorm > self.factor * max(s["ema"], 1e-6))
        if bad:
            s["skipped"] += 1
        else:
            s["ema"] = gnorm if s["accepted"] == 0 else self.decay * s["ema"] + (1.0 - self.decay) * gnorm
            s["accepted"] += 1
        return bad


def split_owners(named, world):
    """Which rank keeps the EMA of which tensor: largest first, each to the rank holding the least so far."""
    load, owner = [0] * world, {}
    for n, p in sorted(named, key=lambda kv: -kv[1].numel()):
        r = min(range(world), key=lambda k: load[k])
        owner[n] = r
        load[r] += p.numel()
    return owner


class EmaShards:
    """This rank's share of the EMA of the trained tensors -- one copy per decay, fp32, on the GPU.  The tensors
    are dealt out by split_owners, so a rank holds 1 / world of each EMA and the full copies only ever meet on disk.

    warmup=True: decay_n = min(decay, (1 + n) / (10 + n)) at update n (0-indexed), which follows the fast early
    changes of the added modules; False: the fixed decay."""

    def __init__(self, named, rank, world, decays=(0.9999,), warmup=True):
        owner = split_owners(named, world)
        self.names = [n for n, _ in named if owner[n] == rank]
        self.src = [p for n, p in named if owner[n] == rank]
        self.decays, self.warmup = tuple(float(d) for d in decays), bool(warmup)
        self.avg = [[p.detach().clone() for p in self.src] for _ in self.decays]

    @torch.no_grad()
    def update(self, n):
        for decay, avg in zip(self.decays, self.avg):
            d = min(decay, (1.0 + n) / (10.0 + n)) if self.warmup else decay
            torch._foreach_mul_(avg, d)
            torch._foreach_add_(avg, self.src, alpha=1.0 - d)

    @staticmethod
    def stem(k):                        # shard files: ema-<rank>.pt for the first decay, ema<k>-<rank>.pt for the others
        return "ema" if k == 0 else f"ema{k}"

    def save(self, directory, rank):
        for k, avg in enumerate(self.avg):
            torch.save(dict(zip(self.names, avg)), os.path.join(directory, f"{self.stem(k)}-{rank:02d}.pt"))

    @torch.no_grad()
    def load(self, directory):
        """From the shards of any number of ranks (a run may resume on a different number of GPUs)."""
        for k, avg in enumerate(self.avg):
            mine, found = dict(zip(self.names, avg)), 0
            for f in sorted(glob.glob(os.path.join(directory, f"{self.stem(k)}-*.pt"))):
                shard = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
                for n in mine.keys() & shard.keys():
                    mine[n].copy_(shard[n])
                    found += 1
                del shard
            if found != len(mine):
                raise SystemExit(f"{directory}: EMA {k} of {len(mine) - found} tensors not found")


class ShardedOptimizer:
    """ZeRO stage 1 beside DDP (train.py --shard_optim): rank r keeps the optimizer state of the tensors split_owners
    deals it -- the same split as EmaShards -- steps only those, and then every tensor is broadcast from its owner.
    AdamW works element by element, so the update is the unsharded optimizer's bit for bit; the moments (29 GiB at
    4B) cost 1 / world per rank, for one broadcast of the trained weights per step.

    `make(groups)` builds the local optimizer (e.g. a fused AdamW) on this rank's share of each group; the groups keep
    their order and hyper-parameters, so `param_groups` can be driven like the unsharded optimizer's.  zero_grad()
    covers every trained tensor (DDP fills all their gradients).  state_dict() is this rank's share, keyed by name
    (Checkpoint writes it as optimizer-<rank>.pt); load_optimizer() reads it back from any number of ranks.

    The broadcasts of step() are left running (they need nothing but the weights, so they overlap whatever the GPU
    does next: train.py's teacher, VAE and text passes); wait() -- called by zero_grad() and Checkpoint.save -- makes
    the current stream wait for them, before anything reads the weights again."""

    def __init__(self, groups, named, rank, world, make, comm_world=None):
        """named: the trained (name, tensor) pairs, in the order EmaShards gets them (so the two splits agree).
        comm_world < world: a memory probe (train.py --probe_world) -- the state is split as over `world` ranks but only
        comm_world exist, so the tensors of the missing owners are never updated (measurements only)."""
        name = {id(p): n for n, p in named}
        owner = split_owners(named, world)
        self.rank, self.world = rank, world
        self.comm_world = world if comm_world is None else comm_world
        self.full_groups = [{"params": list(g["params"])} for g in groups]
        self.all_params = [p for g in groups for p in g["params"]]
        self.owner = [owner[name[id(p)]] for p in self.all_params]
        self.opt = make([{**g, "params": [p for p in g["params"] if owner[name[id(p)]] == rank]} for g in groups])
        self.group_names = [[name[id(p)] for p in g["params"]] for g in self.opt.param_groups]
        self.pending = []

    @property
    def param_groups(self):
        return self.opt.param_groups

    def zero_grad(self, set_to_none=True):
        self.wait()
        if set_to_none:
            for p in self.all_params:
                p.grad = None
        else:
            grads = [p.grad for p in self.all_params if p.grad is not None]
            if grads:
                torch._foreach_zero_(grads)

    @torch.no_grad()
    def step(self):
        self.opt.step()
        self.sync(wait=False)

    @torch.no_grad()
    def sync(self, wait=True):
        """Every tensor from its owner, in the same order on every rank."""
        self.wait()
        if self.comm_world > 1:
            self.pending = [dist.broadcast(p.data, src=r, async_op=True) for p, r in zip(self.all_params, self.owner)
                            if r < self.comm_world]
        if wait:
            self.wait()

    def wait(self):
        for w in self.pending:
            w.wait()
        self.pending = []

    def state_dict(self):
        return {"group_names": self.group_names, "optimizer": self.opt.state_dict()}

    def load_state_dict(self, sd):
        load_optimizer(self.opt, self.group_names, *named_optimizer_state([sd]))


def named_optimizer_state(sources, all_names=None):
    """Optimizer states -> ({tensor name: its state}, the groups' hyper-parameters).  `sources`: ShardedOptimizer
    state dicts (any number of ranks), or one plain optimizer state dict with `all_names` (per group, the names of
    its tensors in order: what the indices of that state dict stand for)."""
    by_name, hparams = {}, None
    for sd in sources:
        names, osd = (sd["group_names"], sd["optimizer"]) if "group_names" in sd else (all_names, sd)
        if names is None:
            raise ValueError("a plain optimizer state dict needs the names of its tensors")
        flat = [n for group in names for n in group]
        if len(flat) != sum(len(g["params"]) for g in osd["param_groups"]):
            raise SystemExit(f"optimizer state: {sum(len(g['params']) for g in osd['param_groups'])} tensors, the model trains {len(flat)}")
        for g, group in zip(osd["param_groups"], names):
            for i, n in zip(g["params"], group):
                if i in osd["state"]:
                    by_name[n] = osd["state"][i]
        hparams = hparams or [{k: v for k, v in g.items() if k != "params"} for g in osd["param_groups"]]
    return by_name, hparams


def load_optimizer(opt, group_names, by_name, hparams):
    """Give `opt` (whose groups hold the tensors `group_names`) the states of `by_name`; tensors without one start
    fresh.  The optimizer moves each state to its tensor's device."""
    sd, i = {"state": {}, "param_groups": []}, 0
    for names, hp in zip(group_names, hparams):
        idx = []
        for n in names:
            if n in by_name:
                sd["state"][i] = by_name[n]
            idx.append(i)
            i += 1
        sd["param_groups"].append({**hp, "params": idx})
    opt.load_state_dict(sd)


class Checkpoint:
    """<out>/checkpoint-last/: model.pt, optimizer.pt (optimizer-<rank>.pt with --shard_optim), the EMA shards of
    every rank, meta.json and any extra files.  Written to checkpoint-tmp and swapped in, so a kill at any moment
    leaves one complete checkpoint.  About 62 GB at 4B with one EMA: keep <out> on scratch.  Either optimizer layout
    resumes into either (by tensor name), on any number of ranks."""

    def __init__(self, rt: Runtime, out):
        self.rt, self.last, self.tmp = rt, os.path.join(out, "checkpoint-last"), os.path.join(out, "checkpoint-tmp")

    def exists(self):
        return os.path.exists(os.path.join(self.last, "meta.json"))

    def save(self, student, opt, ema: EmaShards, meta: dict, extra: dict = None):
        """Called on every rank: all write their EMA shards (and optimizer shards), rank 0 the rest."""
        rt = self.rt
        if rt.main:
            shutil.rmtree(self.tmp, ignore_errors=True)
            os.makedirs(self.tmp)
        rt.barrier()
        ema.save(self.tmp, rt.rank)
        sharded = isinstance(opt, ShardedOptimizer)
        if sharded:
            opt.wait()                                               # the weights rank 0 writes
                                                  # every rank its share, in parallel
            torch.save(opt.state_dict(), os.path.join(self.tmp, f"optimizer-{rt.rank:02d}.pt"))
        if rt.main:                                                  # (while the other ranks write their shards)
            torch.save(student.state_dict(), os.path.join(self.tmp, "model.pt"))
            if not sharded:
                torch.save(opt.state_dict(), os.path.join(self.tmp, "optimizer.pt"))
            for name, obj in (extra or {}).items():
                torch.save(obj, os.path.join(self.tmp, name))
        rt.barrier()
        if rt.main:
            json.dump(meta, open(os.path.join(self.tmp, "meta.json"), "w"), indent=1)
            old = self.last + ".old"
            shutil.rmtree(old, ignore_errors=True)
            if os.path.exists(self.last):
                os.replace(self.last, old)
            os.replace(self.tmp, self.last)
            shutil.rmtree(old, ignore_errors=True)
        rt.barrier()

    def load(self, student, opt, ema: EmaShards, extra=()):
        """-> (meta, {extra file name: its content}).  Called on every rank."""
        meta = json.load(open(os.path.join(self.last, "meta.json")))
        sd = torch.load(os.path.join(self.last, "model.pt"), map_location="cpu", mmap=True, weights_only=True)
        with torch.no_grad():
            for n, p in student.named_parameters():
                p.copy_(sd[n])
        del sd
        self.load_optimizer(student, opt)
        ema.load(self.last)
        files = {name: torch.load(os.path.join(self.last, name), map_location=self.rt.device, weights_only=True) for name in extra}
        if self.rt.cuda:
            torch.cuda.empty_cache()
        return meta, files

    def load_optimizer(self, student, opt):
        """optimizer.pt or the optimizer-<rank>.pt shards, into a plain optimizer or a ShardedOptimizer."""
        shards = sorted(glob.glob(os.path.join(self.last, "optimizer-*.pt")))
        full = os.path.join(self.last, "optimizer.pt")
        sharded = isinstance(opt, ShardedOptimizer)
        if not shards and not sharded:                               # the unsharded layout, as it always was
            opt.load_state_dict(torch.load(full, map_location=self.rt.device, weights_only=True))
            return
        name = {id(p): n for n, p in student.named_parameters()}
        target, names = (opt.opt, opt.group_names) if sharded else (opt, [[name[id(p)] for p in g["params"]] for g in opt.param_groups])
        if shards:
            by_name, hparams = named_optimizer_state([torch.load(f, map_location="cpu", mmap=True, weights_only=True) for f in shards])
        else:
            full_names = [[name[id(p)] for p in g["params"]] for g in opt.full_groups]
            by_name, hparams = named_optimizer_state([torch.load(full, map_location="cpu", mmap=True, weights_only=True)], full_names)
        load_optimizer(target, names, by_name, hparams)

    def export_ema(self, student, path, k=0, metadata=None):
        """Rank 0, right after save(): EMA k of every trained tensor (from the shards just written) + the frozen
        ones -> one bf16 safetensors file that scripts/inference/sample.py and gallery.py load."""
        from ..models.weights import save_weights
        state = {n: p.detach() for n, p in student.named_parameters() if not p.requires_grad}
        for f in sorted(glob.glob(os.path.join(self.last, f"{EmaShards.stem(k)}-*.pt"))):
            state.update(torch.load(f, map_location="cpu", mmap=True, weights_only=True))
        missing = [n for n, _ in student.named_parameters() if n not in state]
        if missing:
            raise RuntimeError(f"EMA export: {len(missing)} tensors missing, e.g. {missing[:3]}")
        save_weights({n: state[n] for n, _ in student.named_parameters()}, path, torch.bfloat16, metadata=metadata)
        log(f"[train] EMA weights -> {path}")


# ---------------------------------------------------------------------------------------------------------
# a step's examples -> student passes
# ---------------------------------------------------------------------------------------------------------
def plan_micro_batches(n_tok, text_len, cap_tokens, cap_batch, bucket=True, fixed=2):
    """Group the examples of a step into student passes.  Sorted by token count, so an image is padded only up to a
    neighbour of similar length, and short sequences are run more at a time: a group holds at most `cap_batch`
    examples and `cap_tokens` (text + padded image) tokens.  text_len: one length for all (FLUX.2's 512), or one per
    example (FLUX 3's text windows; a group's text is padded to its longest).  Largest groups first."""
    n = len(n_tok)
    tl = [int(v) for v in text_len] if isinstance(text_len, (list, tuple)) else [int(text_len)] * n
    if not bucket:
        return [list(range(i, min(i + fixed, n))) for i in range(0, n, fixed)]
    order = sorted(range(n), key=lambda i: (n_tok[i], tl[i]))
    groups, cur, cur_txt = [], [], 0
    for i in order:
        txt = max(cur_txt, tl[i])
        if cur and (len(cur) + 1 > cap_batch or (len(cur) + 1) * (txt + n_tok[i]) > cap_tokens):
            groups.append(cur)
            cur, txt = [], tl[i]
        cur.append(i)
        cur_txt = txt
    if cur:
        groups.append(cur)
    return sorted(groups, key=lambda g: -len(g) * (max(tl[j] for j in g) + n_tok[g[-1]]))


class Clock:
    """Wall time per phase of a step.  With sync=True the GPU is drained at every lap, which makes the split
    exact and the step slightly slower (--profile); without it only the totals are meaningful."""

    def __init__(self, phases, sync=False):
        self.sync, self.total, self.last = sync, dict.fromkeys(phases, 0.0), time.time()

    def lap(self, name):
        if self.sync:
            torch.cuda.synchronize()
        now = time.time()
        self.total[name] += now - self.last
        self.last = now

    def report(self, steps):
        return "  ".join(f"{k} {v / steps:.2f}" for k, v in self.total.items())


# ---------------------------------------------------------------------------------------------------------
# memory: frozen models off the GPU between uses, DDP's bucket rebuild at start-up
# ---------------------------------------------------------------------------------------------------------
class Parked:
    """A frozen module whose parameters live in pinned host memory and come to the GPU only while it is used
    (scripts/train/train.py --offload_frozen: the teacher and the text encoder during a step's preparation; the student's
    passes and the optimizer step then have their memory).  load() copies them in (asynchronously, on the
    current stream), park() drops the GPU copies -- nothing is copied back, the weights never change.  Buffers
    (rotary tables) stay on the GPU.  The same Parameter objects throughout, so compiled code keeps working."""

    def __init__(self, module, device):
        self.module, self.device = module, torch.device(device)
        self.params = list({id(p): p for p in module.parameters()}.values())
        self.host = [p.detach().to("cpu").pin_memory() for p in self.params]
        self.bytes = sum(h.numel() * h.element_size() for h in self.host)
        self.on_gpu = True
        self.park()

    def load(self):
        if not self.on_gpu:
            for p, h in zip(self.params, self.host):
                p.data = h.to(self.device, non_blocking=True)
            self.on_gpu = True
        return self.module

    def park(self):
        if self.on_gpu:
            for p, h in zip(self.params, self.host):
                p.data = h
            self.on_gpu = False


def prebuild_ddp_buckets(net):
    """DDP rebuilds its gradient buckets once, in the order the first synchronised backward produced the gradients,
    and allocates the new buckets before it frees the old ones: with gradient_as_bucket_view that is a one-time spike
    of one more fp32 gradient copy (19.5 GB at FLUX 3 klein) in the middle of training.  Rebuilding right after DDP is
    built, in parameter order (what DDP's join hook does), moves the spike to start-up, where nothing else is
    allocated yet, and DDP never rebuilds again.  The bucket order then follows the parameters instead of the first
    backward, which only matters for overlapping the all-reduce with the last pass of a step.  Every rank must call
    it (the bucket assignment is synchronised).  Returns whether a rebuild took place."""
    net.reducer._push_all_rebuilt_params()
    done = bool(net.reducer._rebuild_buckets())
    net._has_rebuilt_buckets = net._has_rebuilt_buckets or done
    return done


# ---------------------------------------------------------------------------------------------------------
# start-up assertions that need no data
# ---------------------------------------------------------------------------------------------------------
ZERO_AT_START = ("scale_emb", ".up.weight", ".up.bias", "ipe_2d.mlp.1.weight", "ipe_2d.mlp.1.bias", ".out.weight", ".out.bias")


def preflight(student, teacher, text, rt: Runtime, groups=None, expect=None, released=True):
    """The start-up assertions, on the student as it was just built (before a checkpoint is resumed):

      1  on the dense layout it is the plain trunk -- exactly in fp32 -- and, if `released` (it holds the released
         weights), the same arithmetic as the bf16 teacher under autocast;
      2  the Wiener gain is +1 / 0 / -1 at sigma = 1, 1/17, 0; if `released`, every added output layer is zero,
         so a coarse token is the mean of its cells going in and upsampling + skip coming out;
      4  the first and last times of the 50-step schedule (FLUX.2: the reference values at 512 x 512, at other
         grids an increasing schedule from 0 to 1 with the shift mu(N_DENSE); FLUX 3: the release's constant shift);
      5  the parameter counts of the model and of the optimizer groups.

    (3, the VAE statistics, is asserted where the latents are made: scripts/data/encode_latents.py.)
    Returns the lines to log; raises SystemExit on a failure."""
    from ..models.added_modules import hf_skip_gain
    from ..models.convert import is_adapter
    from ..tokenization.layout import GRID, N_DENSE, compute_empirical_mu, dense_layout, pack, schedule, time_shift
    family = getattr(student, "family", "flux2")
    device, lines = rt.device, []
    g = torch.Generator().manual_seed(0)
    z = torch.randn(2, GRID, GRID, student.in_channels, generator=g).to(device)
    t = torch.tensor([0.2, 0.7], device=device)
    ctx, lens = text.encode(["a photograph of a lighthouse at dusk", ""])
    ctx = ctx.to(device)
    lens = None if lens is None else lens.cpu()
    dense = pack(dense_layout(2, device))
    with torch.no_grad():
        d32 = float((student(z, t, ctx.float(), dense, txt_len=lens) - student.forward_dense(z, t, ctx.float(), txt_len=lens)).abs().max())
        with rt.autocast():
            ref = teacher.forward_dense(z, t, ctx, txt_len=lens)
            dbf = float((student(z, t, ctx, dense, txt_len=lens) - ref).abs().max()) if released else 0.0
    scale = float(ref.abs().max())
    if d32 > 1e-5 or dbf > 0.02 * scale:
        raise SystemExit(f"[preflight] the dense layout is not the plain model: fp32 {d32:.2e}, vs the bf16 teacher {dbf:.2e} "
                         f"(output scale {scale:.2f})")
    lines.append(f"[preflight] 1 dense layout == plain trunk: max |diff| {d32:.1e} (fp32)"
                 + (f", {dbf:.1e} vs the released bf16 teacher under autocast (output scale {scale:.2f})" if released else ""))

    w = hf_skip_gain(torch.tensor([1.0, 1.0 / 17.0, 0.0], dtype=torch.float64), student.bp.hf_rho).tolist()
    if student.bp.hf_rho == 0.25 and max(abs(w[0] - 1.0), abs(w[1]), abs(w[2] + 1.0)) > 1e-6:
        raise SystemExit(f"[preflight] Wiener gain at sigma = 1, 1/17, 0 is {w}, not +1, 0, -1")
    nonzero = [n for n, p in student.named_parameters() if is_adapter(n) and n.endswith(ZERO_AT_START) and bool(p.any())]
    if released and nonzero:
        raise SystemExit(f"[preflight] added output layers are not zero at the start: {nonzero[:4]}")
    lines.append(f"[preflight] 2 Wiener gain at sigma = 1, 1/17, 0: {w[0]:+.4f}, {w[1]:+.1e}, {w[2]:+.4f}"
                 + ("; every added output layer is zero" if released else ""))

    ts = schedule(50, family)
    if family == "flux3":                                            # BFL: alpha t / (1 + (alpha - 1) t) on linspace(1, 0, 51)
        a = time_shift("flux3")
        sig = torch.linspace(1.0, 0.0, 51, dtype=torch.float64)
        want = (1.0 - a * sig / (1.0 + (a - 1.0) * sig)).tolist()
        bad = a != 6.93 or max(abs(p - q) for p, q in zip(ts, want)) > 1e-9
        what = f"shift {a}"
    elif N_DENSE == 1024:
        want = [0.0, 0.003707, 0.007540, 0.011504, 0.740694, 0.813982, 0.899336, 1.0]
        bad = max(abs(a - b) for a, b in zip(ts[:4] + ts[-4:], want)) > 1e-6
        what = f"mu {compute_empirical_mu(N_DENSE, 50):.6f}"
    else:
        bad = ts[0] != 0.0 or ts[-1] != 1.0 or any(b <= a for a, b in zip(ts, ts[1:]))
        what = f"mu {compute_empirical_mu(N_DENSE, 50):.6f}"
    if bad:
        raise SystemExit(f"[preflight] schedule: {ts[:4]} ... {ts[-4:]}")
    lines.append(f"[preflight] 4 schedule(50 steps, {N_DENSE} tokens, {family} {what}): "
                 f"{', '.join(f'{v:.6f}' for v in ts[:4])} ... {', '.join(f'{v:.6f}' for v in ts[-4:])}")

    n_trunk = sum(p.numel() for n, p in student.named_parameters() if not is_adapter(n))
    n_added = sum(p.numel() for n, p in student.named_parameters() if is_adapter(n))
    if expect and (n_trunk, n_added) != tuple(expect):
        raise SystemExit(f"[preflight] parameters: trunk {n_trunk:,}, added {n_added:,}; expected {expect}")
    if groups is not None:
        sizes = [sum(p.numel() for p in grp["params"]) for grp in groups]
        if sizes[0] != n_added or (len(sizes) > 1 and sizes[1] != n_trunk):
            raise SystemExit(f"[preflight] optimizer groups hold {sizes}, model has trunk {n_trunk:,} + added {n_added:,}")
        lines.append("[preflight] 5 optimizer groups: " + ", ".join(f"{n:,} parameters at lr {grp['lr']:.1e}" for n, grp in zip(sizes, groups)))
    del z, t, ctx, ref
    if rt.cuda:
        torch.cuda.empty_cache()
    return lines


# ---------------------------------------------------------------------------------------------------------
# the command line, cadence, validation and --init of both trainers
# ---------------------------------------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # budgetlatent/training/ -> the checkout
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "flux2_512.yaml")
N_PARAMS = (3_875_544_576, 28_136_192)      # FLUX.2 [klein]: released trunk, added modules (FLUX 3: weights.expected_counts)


def add_run_arguments(ap):
    """The arguments the two trainers have in common."""
    from ..encoders.text import TEMPLATES
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="configs/{flux2,flux3}_{512,1024}.yaml (model.family picks the family)")
    ap.add_argument("--data", required=True, help="directory of chunk_*.pt: the 120K mix as pixels or as latents")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="snapshot of the released model (default: the config's repo in the HF cache)")
    ap.add_argument("--batch_per_gpu", type=int, default=2, help="examples per student pass at the dense grid")
    ap.add_argument("--grad_accum", type=int, default=16, help="examples per GPU and step = batch_per_gpu x grad_accum")
    ap.add_argument("--max_steps", type=int, default=None, help="default: the config's")
    ap.add_argument("--prep_batch", type=int, default=8, help="batch of the teacher / VAE / text-encoder passes")
    ap.add_argument("--max_micro", type=int, default=None, help="most examples in one student pass (default 4 x batch_per_gpu)")
    ap.add_argument("--no_bucket", action="store_true", help="fixed student passes of batch_per_gpu in arrival order")
    ap.add_argument("--ckpt_blocks", type=int, default=None, help="blocks under gradient checkpointing (default: all; FLUX.2 25, FLUX 3 38)")
    ap.add_argument("--template", default=None, choices=list(TEMPLATES), help="default: model.chat_template of the config")
    ap.add_argument("--allow_batch_mismatch", action="store_true", help="run with a global batch other than the config's")
    ap.add_argument("--profile", action="store_true", help="exact time per phase of a step (drains the GPU at every lap)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500, help="checkpoint-last, for resuming")
    ap.add_argument("--snapshot_every", type=int, default=None, help="EMA weights (default: optim.ckpt_every; a multiple of save_every)")
    ap.add_argument("--val_every", type=int, default=None, help="validation gallery (default: at every snapshot; 0 = never)")
    ap.add_argument("--val_prompts", type=int, default=None, help="prompts of the validation suite to render (default: all 32)")
    ap.add_argument("--val_budgets", type=float, nargs="+", default=None, help="budgets of the validation suite (default: all 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_save", action="store_true", help="write nothing (benchmark runs)")
    ap.add_argument("--skip_preflight", "--skip_parity", dest="skip_preflight", action="store_true")
    return ap


def check_cadence(a, cfg):
    """Fill in the snapshot / validation cadence from the config and check that they nest."""
    a.snapshot_every = a.snapshot_every or int(cfg["optim"].get("ckpt_every", 4000))
    a.val_every = a.snapshot_every if a.val_every is None else a.val_every
    if a.snapshot_every % a.save_every:
        raise SystemExit("--snapshot_every must be a multiple of --save_every")
    if a.val_every % a.snapshot_every:
        raise SystemExit("--val_every must be a multiple of --snapshot_every (validation reads the snapshot)")


def run_validation(rt, a, cfg, weights, root, vae, text, step, family="flux2"):
    """All ranks: render the validation suite with the EMA weights rank 0 just exported (FLUX 3: through the config's
    model.time_shift).  A failure here (memory, disk) is reported and never stops the training."""
    from ..inference.validate import validate_weights
    out, tic = os.path.join(a.out, f"val-s{step}"), time.time()
    try:
        inference = dict(cfg.get("inference") or {})
        if family != "flux2" and "time_shift" in cfg["model"]:
            inference.setdefault("time_shift", float(cfg["model"]["time_shift"]))
        validate_weights(weights, root, vae, text, out, rt.rank, rt.world, rt.device, inference, a.val_prompts, a.val_budgets,
                         family=family)
        if rt.main:
            log(f"[val] step {step}: validation gallery -> {out} ({time.time() - tic:.0f} s)")
    except Exception as e:                                           # noqa: BLE001
        log(f"[val] step {step}, rank {rt.rank}: validation failed and is skipped ({type(e).__name__}: {str(e)[:200]})")
        if rt.cuda:
            torch.cuda.empty_cache()


def load_init(student, path):
    """--init: the student's weights in place from `path` -- a stage-1 checkpoint directory (its fp32 EMA shards
    ema-XX.pt, else model.pt), a .pt state dict, or a .safetensors file (the trainers' bf16 snapshots: the trunk loses the
    precision a small learning rate builds up, prefer the fp32 shards).  Every parameter must be there; returns the
    count."""
    import glob
    from ..models.weights import read_state
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "ema-*.pt"))) or [os.path.join(path, "model.pt")]
        state = {}
        for f in files:
            state.update(torch.load(f, map_location="cpu", mmap=True, weights_only=True))
    elif path.endswith(".pt"):
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    else:
        state = read_state(path)[0]
    params = list(student.named_parameters())
    missing = [n for n, _ in params if n not in state]
    if missing:
        raise SystemExit(f"--init {path}: {len(missing)} of {len(params)} parameters missing, e.g. {missing[:3]}")
    with torch.no_grad():
        for n, p in params:
            p.copy_(state[n].to(p.device, p.dtype))
    del state
    return len(params)
