"""The frozen FLUX.2 VAE with the latent layout the trunk uses (FLUX 3 Image Klein ships the same architecture with
its own weights and BatchNorm statistics, read from its own snapshot).

512 x 512 x 3 pixels in [-1, 1]  <->  (32, 32, 128) latents: the 8x conv autoencoder (32 channels at 64 x 64),
a 2 x 2 space-to-depth (channel = c * 4 + i * 2 + j) and the VAE's frozen BatchNorm statistics.
"""
import copy
import os

import torch
import torch.nn as nn


class Flux2VAE(nn.Module):
    def __init__(self, path, device="cpu", chunk=None):
        """path: a snapshot of black-forest-labs/FLUX.2-klein-base-4B or of black-forest-labs/flux-3-image-klein-base
        (its vae/ subfolder is read).
        chunk: images per pass through the convolutions.  None: 4 x (512 / side)^2, i.e. 4 at 512 x 512 and one
        image at a time at 1024 x 1024, where one fp32 decode alone needs about 4 GiB of activations (four at once
        took 18 GiB, more than the transformer needs at any budget, so the peak memory no longer followed the
        budget).  Each image goes through the same convolutions either way."""
        super().__init__()
        from diffusers import AutoencoderKLFlux2
        sub = os.path.join(path, "vae") if os.path.isdir(os.path.join(path, "vae")) else path
        self.vae = AutoencoderKLFlux2.from_pretrained(sub, torch_dtype=torch.float32).eval().requires_grad_(False)
        eps = self.vae.config.batch_norm_eps
        self.register_buffer("mean", self.vae.bn.running_mean.detach().float().view(1, -1, 1, 1).clone(), persistent=False)
        self.register_buffer("std", torch.sqrt(self.vae.bn.running_var.detach().float().view(1, -1, 1, 1) + eps).clone(),
                             persistent=False)
        self.chunk, self.memory_format = chunk, torch.contiguous_format
        self.preview = None                                          # bf16 copy of the decoder (bf16_preview)
        self.to(device)

    def optimize(self, compile_decoder=False, channels_last=True):
        """cuDNN autotuning (the inputs never change size), channels-last convolutions unless channels_last=False,
        optionally a compiled decoder.  Same results up to the convolution algorithm cuDNN picks.  NCHW
        (channels_last=False) can be the faster eager layout: the group norms then need no layout copies."""
        torch.backends.cudnn.benchmark = True
        if channels_last:
            self.memory_format = torch.channels_last
            self.vae.to(memory_format=torch.channels_last)
        if compile_decoder and not getattr(self, "_compiled", False):
            self.vae.decoder = torch.compile(self.vae.decoder, dynamic=False)
            self._compiled = True
        return self

    def bf16_preview(self, compile=False):
        """Make decode(z, bf16="weights") -- what the sampler's layout previews call -- run a bf16 copy of
        post_quant_conv + decoder end to end (group norms in bf16 too, no casts between the layers).  The fp32
        decoder, which makes the final images, is untouched.  compile: torch.compile the copy (a graph per batch
        size).  The preview only feeds the entropy of 16 px cells, so this changes a layout only where two
        entropies were within the bf16 rounding of each other."""
        if self.preview is None:
            dec = getattr(self.vae.decoder, "_orig_mod", self.vae.decoder)
            pq = self.vae.post_quant_conv
            mods = [m for m in (pq, dec) if m is not None]
            self.preview = nn.Sequential(*copy.deepcopy(mods)).to(torch.bfloat16).to(memory_format=self.memory_format)
            if compile:
                import torch._dynamo as dynamo
                dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 32)    # a graph per batch size
                self.preview = torch.compile(self.preview, dynamic=False)
        return self

    def _chunk(self, side_px):
        """Images per pass for images of side_px x side_px pixels."""
        return self.chunk or max(1, int(4 * (512 * 512) // (side_px * side_px)))

    @torch.no_grad()
    def encode(self, x):
        """(B, 3, H, W) in [-1, 1] -> (B, H / 16, W / 16, 128) fp32, the posterior mean (no sampling)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float().contiguous(memory_format=self.memory_format)
            z = torch.cat([self.vae.encode(part).latent_dist.mode() for part in x.split(self._chunk(x.shape[-1]))])
            b, c, h, w = z.shape
            z = z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
            return ((z - self.mean) / self.std).permute(0, 2, 3, 1).contiguous()

    @torch.no_grad()
    def decode(self, z, bf16=False, downscale=1):
        """(B, h, w, 128) -> (B, 3, 16 h, 16 w) fp32 in about [-1, 1].  bf16=True runs the convolutions in bf16
        (group norms stay fp32): the cheap preview the layouts are computed from.  bf16="weights": the bf16 copy of
        bf16_preview() if it was made, else the same as bf16=True.  downscale: decode the latent average-pooled by this
        factor, a (B, 3, 16 h / downscale, 16 w / downscale) image for a fraction of the cost (a preview)."""
        z = z.float().permute(0, 3, 1, 2) * self.std + self.mean
        b, c, h, w = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)
        if downscale > 1:
            z = torch.nn.functional.avg_pool2d(z, int(downscale))
        z = z.contiguous(memory_format=self.memory_format)
        n = self._chunk(8 * z.shape[-1])
        if bf16 == "weights" and self.preview is not None and z.is_cuda:
            with torch.autocast(device_type=z.device.type, enabled=False):
                return torch.cat([self.preview(part).float() for part in z.to(torch.bfloat16).split(n)])
        with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16, enabled=bool(bf16) and z.is_cuda):
            return torch.cat([self.vae.decode(part).sample.float() for part in z.split(n)])
