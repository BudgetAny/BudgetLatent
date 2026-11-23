"""Stage 2: data-free DMD2 + GAN sharpening of a BudgetPix student (optional; the reference model is stage 1 only).

Only captions are read.  A step makes its own "clean" latents with a 5-step rollout, noises them, and trains

  the student     on the paired teacher velocity (as in stage 1) + a distribution-matching term (the difference of
                  two denoisers at a renoised copy of its own estimate) + a generator GAN term;
  two critics     that exist only during training: FakeScoreLoRA, 26 rank-16 residuals on the frozen dense teacher
                  that turn it into a denoiser of the student's outputs (without them it is the real one, so no
                  second trunk is held), and DMD2DiscriminatorHead, real / fake logits from that teacher's features.

  FakeScoreLoRA, DMD2DiscriminatorHead      the two critics (the reference classes of the method)
  rollout, synthesize                       simulated clean latents -> a training state (z_t, t, teacher target, layouts)
  generator_loss, critic_loss               the per-example loss terms of one student pass / one critic pass
  effective_scale                           the critic warm-up and ramp of the DMD and GAN terms

Times follow the rest of the code: t = 0 noise, t = 1 clean, z_t = t x + (1 - t) eps, velocities are x - eps.
The functions take a torch.Generator for every random draw and `amp` (bf16 autocast for the trunk passes; the
critics' own arithmetic is fp32).  train_dmd.py is the trainer around them.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokenization.layout import (COARSE, N_DENSE, N_MIN, LayoutMix, calibrate_offset, draw_plan, flux2_klein_timesteps,
                                   noise_latent_pyramid_s8, pack, pixel_entropy_pyramid_s8, sample_times, threshold_layout,
                                   training_layouts)


@dataclass
class DMDConfig:                        # `dmd2_gan` of configs/flux2_512.yaml
    dmd_sim_steps: int = 5
    dmd_backward_sim_prob: float = 0.5  # share of steps whose rollout is the student's (on layouts); else the dense teacher's
    data_t_uniform_frac: float = 0.35
    dmd_weight: float = 0.25
    gan_weight: float = 0.015
    d_gan_loss_weight: float = 0.5
    logit_reg_weight: float = 1.0e-4
    lora_rank: int = 16
    lora_alpha: float = 16.0
    cls_channels: int = 256
    color_branch_weight: float = 0.5
    min_step_frac: float = 0.02
    max_step_frac: float = 0.98
    use_clean_gate: bool = True         # DMD gradient weighted by t^2 of the source state
    gan_feature_lora_grad: bool = False  # False: the discriminator's features carry no gradient to the LoRA
    fake_score_match_dmd: bool = True   # the LoRA is trained at the times and with the t^2 weights the DMD term queries it
    ttur_updates: int = 2
    critic_lr: float = 1.0e-4
    critic_betas: tuple = (0.0, 0.99)
    critic_weight_decay: float = 1.0e-4
    critic_grad_clip_norm: float = 1.0

    @classmethod
    def from_config(cls, d):
        d = dict(d or {})
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        if "critic_betas" in kw:
            kw["critic_betas"] = tuple(float(v) for v in kw["critic_betas"])
        return cls(**kw)


def _fp32(device):
    return torch.autocast(device_type=device.type, enabled=False)


def _amp(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(amp))


def _randn(shape, gen, device):
    return torch.randn(shape, generator=gen).to(device)


# ---------------------------------------------------------------------------------------------------------
# the critics
# ---------------------------------------------------------------------------------------------------------
class FakeScoreLoRA(nn.Module):
    """Rank-r residuals on the frozen dense teacher (2,508,800 parameters at 4B): one per double-stream block (image
    stream), one per single-stream block (joint stream), one on the output projection.  B = 0 at the start, so the
    fake denoiser starts as the real one and the DMD gradient at zero.  trunk.Flux2Trunk.run_blocks calls `tap`."""

    def __init__(self, dim=3072, out_channels=128, n_double=5, n_single=20, rank=16, alpha=16.0):
        super().__init__()
        self.scale = alpha / float(rank)
        std = 1.0 / math.sqrt(float(dim))
        self.a, self.b = nn.ParameterDict(), nn.ParameterDict()
        for key in [f"double_{i}" for i in range(n_double)] + [f"single_{i}" for i in range(n_single)] + ["final"]:
            self.a[key] = nn.Parameter(torch.randn(dim, rank) * std)
            self.b[key] = nn.Parameter(torch.zeros(rank, out_channels if key == "final" else dim))

    @classmethod
    def for_trunk(cls, trunk, rank=16, alpha=16.0):
        return cls(trunk.hidden_size, trunk.in_channels, len(trunk.double_blocks), len(trunk.single_blocks), rank, alpha)

    def delta(self, key, x):
        """((x A) B) alpha / rank, in fp32 whatever the surrounding autocast."""
        with _fp32(x.device):
            return (((x.float() @ self.a[key].float()) @ self.b[key].float()) * self.scale).to(x.dtype)

    def tap(self, key, x, mod):
        """What block `key` adds to its output: gate * delta of its own modulated, layer-normed input.
        x (B, L, D) the block's input stream, mod = (shift, scale, gate) of that block."""
        shift, scale, gate = mod
        with _fp32(x.device):
            h = (1.0 + scale.float()) * F.layer_norm(x.float(), x.shape[-1:], eps=1e-6) + shift.float()
            return (gate.float() * self.delta(key, h).float()).to(x.dtype)


def fourier_t_features(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
    ang = t.float()[:, None] * 100.0 * freqs[None, :]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class DMD2DiscriminatorHead(nn.Module):
    """Real / fake logits for a noised latent, from the (LoRA) teacher's features of it.

    cls: the mid and final spatial features, FiLM-conditioned on (time embedding, pooled text, t), through a
    4 x 4 stride-2 pyramid (32 -> 16 -> 8 -> 4 -> 1).  color: a norm-free branch on global and per-quadrant
    statistics of the 2 x 2-unpacked latent, which a normalised feature pyramid cannot see (overall colour and
    contrast).  total = cls + color_weight * color."""

    def __init__(self, dim=3072, txt_dim=7680, cls_ch=256, color_weight=0.5, latent_channels=128, groups=32):
        super().__init__()
        self.color_weight, self.c = color_weight, latent_channels // 4
        self.vec_ln = nn.LayerNorm(dim, eps=1e-6)
        self.txt_ln = nn.LayerNorm(txt_dim, eps=1e-6)
        self.cond_fc1 = nn.Linear(dim + txt_dim + 64, cls_ch)
        self.cond_fc2 = nn.Linear(cls_ch, cls_ch)

        self.cls_in_proj = nn.Conv2d(2 * dim, cls_ch, kernel_size=1)
        self.cls_in_gn = nn.GroupNorm(groups, cls_ch)
        self.cls_in_film = nn.Linear(cls_ch, 2 * cls_ch)
        self.cls_convs = nn.ModuleList([nn.Conv2d(cls_ch, cls_ch, kernel_size=4, stride=2, padding=0) for _ in range(3)])
        self.cls_gns = nn.ModuleList([nn.GroupNorm(groups, cls_ch) for _ in range(3)])
        self.cls_conv_final = nn.Conv2d(cls_ch, cls_ch, kernel_size=4, stride=1, padding=0)       # 4 x 4 -> 1 x 1
        self.cls_gn_final = nn.GroupNorm(groups, cls_ch)
        self.cls_out = nn.Conv2d(cls_ch, 1, kernel_size=1)

        # mean (c) + std (c) + 2 x 2 quadrant means (4 c) of the unpacked latent, the pooled final feature (64), cond
        self.dit_dc_proj = nn.Linear(dim, 64)
        self.color_fc1 = nn.Linear(6 * self.c + 64 + cls_ch, cls_ch)
        self.color_fc2 = nn.Linear(cls_ch, cls_ch // 2)
        self.color_out = nn.Linear(cls_ch // 2, 1)

    def forward(self, mid_sp, fin_sp, vec, z_gan, t_gan, pooled_txt):
        """mid_sp, fin_sp (B, g, g, D) features, vec (B, D), z_gan (B, g, g, 128) the latent they were computed from,
        t_gan (B,), pooled_txt (B, txt_dim).  Returns {"cls", "color", "total"}: (B,) logits, fp32."""
        with _fp32(fin_sp.device):
            bsz, g = fin_sp.shape[0], fin_sp.shape[1]
            cond_in = torch.cat([self.vec_ln(vec.float()), self.txt_ln(pooled_txt.float()), fourier_t_features(t_gan, 64)], dim=-1)
            cond = F.silu(self.cond_fc2(F.silu(self.cond_fc1(cond_in))))

            feat = torch.cat([mid_sp.float(), fin_sp.float()], dim=-1).permute(0, 3, 1, 2)
            f_scale, f_shift = self.cls_in_film(cond).chunk(2, dim=-1)
            h = F.silu(self.cls_in_gn(self.cls_in_proj(feat)) * (1.0 + f_scale[:, :, None, None]) + f_shift[:, :, None, None])
            for conv, gn in zip(self.cls_convs, self.cls_gns):
                h = F.silu(gn(conv(F.pad(h, (1, 1, 1, 1)))))                 # "SAME" for kernel 4, stride 2
            h = F.silu(self.cls_gn_final(self.cls_conv_final(h)))
            cls_logit = self.cls_out(h).mean(dim=(1, 2, 3))

            c = self.c
            z = z_gan.float().view(bsz, g, g, c, 2, 2).permute(0, 1, 4, 2, 5, 3).reshape(bsz, 2 * g, 2 * g, c)
            z_mean = z.mean(dim=(1, 2))
            z_std = ((z - z_mean[:, None, None, :]).pow(2).mean(dim=(1, 2)) + 1e-6).sqrt()
            z_quad = z.view(bsz, 2, g, 2, g, c).mean(dim=(2, 4)).reshape(bsz, 4 * c)
            dit_dc = self.dit_dc_proj(fin_sp.float().mean(dim=(1, 2)))
            c_in = torch.cat([z_mean, z_std, z_quad, dit_dc, cond], dim=-1)
            color_logit = self.color_out(F.silu(self.color_fc2(F.silu(self.color_fc1(c_in))))).squeeze(-1)
            return {"cls": cls_logit, "color": color_logit, "total": cls_logit + self.color_weight * color_logit}

    @classmethod
    def for_trunk(cls, trunk, cfg: DMDConfig = DMDConfig()):
        groups = 32 if cfg.cls_channels % 32 == 0 else 1
        return cls(trunk.hidden_size, trunk.cfg.context_in_dim, cfg.cls_channels, cfg.color_branch_weight, trunk.in_channels, groups)


def set_trainable(flag, *modules):
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(flag)


# ---------------------------------------------------------------------------------------------------------
# schedule of the adversarial terms, renoising times
# ---------------------------------------------------------------------------------------------------------
def effective_scale(critic_step: int, warmup: int, ramp: int) -> float:
    """Weight of the DMD and GAN terms in the student's loss: 0 while the critics warm up (`warmup` critic
    updates), then a linear ramp over `ramp` critic updates to 1."""
    if critic_step < warmup:
        return 0.0
    return min(1.0, max(0.0, float(critic_step - warmup + 1) / max(ramp, 1)))


def renoise_times(t_src, gen=None, lo=0.02, hi=0.98, capped=False):
    """Times at which a clean estimate is noised again: U[lo, hi], or with capped=True U[lo, max(t_src, lo + 0.05,
    0.15)] per example -- never (much) cleaner than the state the estimate came from."""
    u = torch.rand(t_src.shape[0], generator=gen).to(t_src.device)
    if not capped:
        return lo + u * (hi - lo)
    return lo + u * (t_src.float().clamp(min=max(lo + 0.05, 0.15), max=hi) - lo)


def pooled_text(ctx, ids, pad_id):
    """Mean of the text states over the non-padding tokens: (B, Lt, C) -> (B, C) fp32."""
    mask = (ids != pad_id).float()
    return (ctx.float() * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)


def coarse_weights(cell_scale, boost):
    """(B, g, g) loss weights: 1 + boost under coarse tokens, normalised to mean 1 per image."""
    w = 1.0 + boost * (cell_scale > 1).float()
    return w / w.mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)


# ---------------------------------------------------------------------------------------------------------
# the simulated training state
# ---------------------------------------------------------------------------------------------------------
@torch.no_grad()
def rollout(student, teacher, vae, ctx, use_student, gen=None, steps=5, budget_range=(1.0 / 64.0, 1.0), amp=True):
    """Simulated clean latents (B, g, g, 128): `steps` Euler steps from noise, without guidance.

    use_student=False: the dense teacher.  True: the student, each image on its own layout with at most
    target ~ LogUniform(budget_range) x 1024 tokens -- the first from the noise itself, the later ones from the
    entropy of the decoded running estimate, as in sampling (unaugmented, one threshold offset per image)."""
    bsz, device = ctx.shape[0], ctx.device
    net = student if use_student else teacher
    g, c = 32, net.in_channels
    ts = flux2_klein_timesteps(steps, N_DENSE)
    z = _randn((bsz, g, g, c), gen, device)
    if use_student:
        lo, hi = math.log(budget_range[0]), math.log(budget_range[1])
        target = torch.round(torch.exp(lo + (hi - lo) * torch.rand(bsz, generator=gen)) * N_DENSE).long().clamp(N_MIN, N_DENSE).to(device)
    src = z
    for i in range(steps):
        t = torch.full((bsz,), ts[i], device=device)
        if use_student:
            pyr = noise_latent_pyramid_s8(src) if i == 0 else pixel_entropy_pyramid_s8(vae.decode(src, bf16=amp))
            lay = pack(threshold_layout(pyr, calibrate_offset(pyr, target)))
            with _amp(device, amp):
                v = student(z, t, ctx, lay).float()
            src = z + (1.0 - ts[i]) * v
        else:
            with _amp(device, amp):
                v = teacher.forward_dense(z, t, ctx).float()
        z = z + (ts[i + 1] - ts[i]) * v
    return z


@torch.no_grad()
def synthesize(student, teacher, vae, ctx, use_student, mix: LayoutMix, cfg: DMDConfig, gen=None, first_p=0.10,
               num_steps=50, amp=True):
    """A training state from captions alone: rollout -> z_clean, noised to t (the stage-1 times, a share of them
    uniform instead), the dense teacher's velocity there and the layouts of the training mixture read from the image
    it implies.  Returns z_t, t, v_real, x0_real (all fp32), leaf masks, mode index per image."""
    bsz, device = ctx.shape[0], ctx.device
    z_clean = rollout(student, teacher, vae, ctx, use_student, gen, cfg.dmd_sim_steps, mix.budget_range, amp)
    t, first = sample_times(bsz, gen, device, first_p, num_steps, cfg.data_t_uniform_frac)
    tb = t[:, None, None, None]
    z_t = tb * z_clean + (1.0 - tb) * _randn(tuple(z_clean.shape), gen, device)
    with _amp(device, amp):
        v_real = teacher.forward_dense(z_t, t, ctx).float()
    x0_real = z_t + (1.0 - tb) * v_real
    plan = draw_plan(bsz, mix, gen, device)
    pyr = noise_latent_pyramid_s8(z_t)
    need = plan.needs_detail & ~first
    if bool(need.any()):
        pix = pixel_entropy_pyramid_s8(vae.decode(x0_real[need], bf16=amp))
        for s in COARSE:
            pyr[s][need] = pix[s]
    active, mode = training_layouts(pyr, mix, gen, plan)
    return z_t, t, v_real, x0_real, active, mode


# ---------------------------------------------------------------------------------------------------------
# losses (per example; the trainer sums them over its passes)
# ---------------------------------------------------------------------------------------------------------
def dmd_target(x0_fake, pred_real, pred_fake, t_src, clean_gate=True):
    """The latent the student's estimate is pulled to: x0_fake - gate * clip((pred_fake - pred_real) / norm), with
    norm = mean |x0_fake - pred_real| per example and gate = t_src^2.  All inputs detached."""
    norm = (x0_fake - pred_real).abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-4)
    grad = ((pred_fake - pred_real) / norm).clamp(-10.0, 10.0)
    if clean_gate:
        grad = t_src[:, None, None, None].pow(2) * grad
    return x0_fake - grad


def generator_loss(v_pred, z_t, t, v_real, weights, ctx, pooled, teacher, lora, head, cfg: DMDConfig, eff_scale, gen=None, amp=True):
    """One student pass.  v_pred (B, g, g, 128) fp32 with its graph; weights (B, g, g) from coarse_weights.
    Returns ({"distill", "dmd", "gan"}: (B,) each, the last two only when eff_scale > 0), x0_fake detached.
    The student's objective per example is distill + eff_scale * (dmd_weight * dmd + gan_weight * gan)."""
    device = v_pred.device
    tb = t[:, None, None, None]
    terms = {"distill": (weights[..., None] * (v_pred - v_real).pow(2)).mean(dim=(1, 2, 3))}
    x0_fake = z_t + (1.0 - tb) * v_pred
    if eff_scale > 0.0:
        with torch.no_grad():
            s = renoise_times(t, gen, cfg.min_step_frac, cfg.max_step_frac, capped=True)
            sb = s[:, None, None, None]
            z_s = sb * x0_fake.detach() + (1.0 - sb) * _randn(tuple(x0_fake.shape), gen, device)
            with _amp(device, amp):
                v_r = teacher.forward_dense(z_s, s, ctx).float()
                v_f = teacher.forward_dense(z_s, s, ctx, lora=lora).float()
            target = dmd_target(x0_fake.detach(), z_s + (1.0 - sb) * v_r, z_s + (1.0 - sb) * v_f, t, cfg.use_clean_gate)
        terms["dmd"] = 0.5 * (x0_fake - target).pow(2).mean(dim=(1, 2, 3))

        # non-saturating generator loss; the gradient reaches the student through the teacher's features of z_gan
        t_gan = renoise_times(t, gen, cfg.min_step_frac, cfg.max_step_frac)
        gb = t_gan[:, None, None, None]
        z_gan = gb * x0_fake + (1.0 - gb) * _randn(tuple(x0_fake.shape), gen, device)
        with _amp(device, amp):
            _, aux = teacher.forward_dense(z_gan, t_gan, ctx, lora=lora, return_aux=True)
        logit = head(aux["mid_spatial"], aux["final_spatial"], aux["vec"], z_gan, t_gan, pooled)["total"]
        terms["gan"] = F.softplus(-logit.float())
    return terms, x0_fake.detach()


def critic_loss(x0_fake, x0_real, t, score_weight, ctx, pooled, teacher, lora, head, cfg: DMDConfig, gen=None, amp=True):
    """One critic pass on detached clean estimates.  score_weight (B,): the weight of each example in the
    fake-score term (normalised t^2 with fake_score_match_dmd).  Returns {"score", "gan"}: (B,) each; the critics'
    objective per example is score + d_gan_loss_weight * gan."""
    device, bsz = x0_fake.device, x0_fake.shape[0]
    # discriminator: fake and real estimates at the same time and with the same noise
    t_c = renoise_times(t, gen, cfg.min_step_frac, cfg.max_step_frac)
    cb = t_c[:, None, None, None]
    eps = _randn(tuple(x0_fake.shape), gen, device)
    z_both = torch.cat((cb * x0_fake + (1.0 - cb) * eps, cb * x0_real + (1.0 - cb) * eps))
    t_both = torch.cat((t_c, t_c))
    with torch.set_grad_enabled(bool(cfg.gan_feature_lora_grad)), _amp(device, amp):
        _, aux = teacher.forward_dense(z_both, t_both, torch.cat((ctx, ctx)), lora=lora, return_aux=True)
    logit = head(aux["mid_spatial"], aux["final_spatial"], aux["vec"], z_both, t_both, torch.cat((pooled, pooled)))["total"].float()
    fake, real = logit[:bsz], logit[bsz:]
    gan = F.softplus(-real) + F.softplus(fake) + 0.5 * cfg.logit_reg_weight * (real.pow(2) + fake.pow(2))

    # fake score: the LoRA teacher denoises the student's estimates
    t_s = renoise_times(t, gen, cfg.min_step_frac, cfg.max_step_frac, capped=cfg.fake_score_match_dmd)
    sb = t_s[:, None, None, None]
    eps = _randn(tuple(x0_fake.shape), gen, device)
    with _amp(device, amp):
        v = teacher.forward_dense(sb * x0_fake + (1.0 - sb) * eps, t_s, ctx, lora=lora).float()
    score = score_weight * (v - (x0_fake - eps)).pow(2).mean(dim=(1, 2, 3))
    return {"score": score, "gan": gan}


def score_weights(t, match_dmd=True):
    """Per-example weights of the fake-score term over a rank's examples of one step: t^2 / mean(t^2), or 1."""
    if not match_dmd:
        return torch.ones_like(t)
    t2 = t.float().pow(2)
    return t2 / t2.mean().clamp(min=1e-6)
