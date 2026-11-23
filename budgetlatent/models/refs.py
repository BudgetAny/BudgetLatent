"""Reference images (editing, multi-reference generation) for both trunk families, kept dense.

    refs = References.from_latents([vae.encode(x1)[0], vae.encode(x2)[0]], family="flux3")    # (h, w, 128) each
    v = model(z, t, ctx, layout, ..., refs=refs)          # the budget covers the generated image only

The references are clean latents that ride along in the joint attention: their tokens are never budgeted, pooled or
decoded (the quadtree, the cascade mixers and refiners touch the generated image's tokens only), and they keep their
full token count in every pass, so a budget speeds an edit up less than text to image (with one 1024 x 1024
reference, 4,096 tokens ride along with the 409 of a 10 % budget).

How each family takes them (the released conventions):
  flux2   FLUX.2 [klein] (diffusers Flux2KleinPipeline): the references' tokens go through the image projection img_in
          and join the image stream after the generated image's tokens, at the image's flow time; reference i has the
          ids (10 + 10 i, h, w, 0) (the time axis tells them apart); the prediction is read on the image's tokens.
  flux3   FLUX 3 Image Klein (BFL's pipeline.py / dit.py): mode "image_cond" -- its own input projection, 5 per-mode
          blocks and modulations, at time 0 (plus the flow map's t_ref embedding); ids (0, h, w, -(1 + i)), h and w
          scaled by image_rope_scale; in the single blocks the sequence is [image, image_cond, text] (BFL's order).

From image files (scripts/inference/sample.py --ref):

    images = prepare_images([Image.open(f) for f in files], size=16 * layout.GRID)
    refs = encode_images(images, vae, family)

The first reference sets the output size, as in the released pipelines; the budget layouts tile a square grid, so it
must be 16 GRID x 16 GRID px (1024 x 1024 at BP_GRID=64).  Each further reference is fitted under 1 MP and cropped to
the 16 px grid, as BFL's prepare_multi_reference and Flux2KleinPipeline do.
"""
import math
from dataclasses import dataclass, field

import torch

FLUX2_TIME_SCALE = 10                   # Flux2KleinPipeline._prepare_image_ids: t = scale + scale * i
MAX_REFERENCES = 10                     # BFL's FLUX 3 Image Klein pipeline (MAX_REFERENCES)
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024    # each reference of a multi-reference edit (BFL's pipeline, Flux2KleinPipeline)
PX = 16                                 # one latent cell


def prepare_images(images, size):
    """PIL images -> RGB references for an output of size x size px.  The first must already be size x size (it sets
    the output size); the others are resized under 1 MP if larger (sides floored) and centre-cropped to multiples of
    16 px.  Raises ValueError for anything else."""
    from PIL import Image, ImageOps
    if not images:
        raise ValueError("no reference images")
    if len(images) > MAX_REFERENCES:
        raise ValueError(f"{len(images)} reference images: at most {MAX_REFERENCES}")
    out = []
    for i, img in enumerate(images):
        img = ImageOps.exif_transpose(img).convert("RGB")
        if i == 0:
            if img.size != (size, size):
                other = "" if size == 1024 else " (or set BP_GRID=64 for 1024 x 1024)"
                raise ValueError(f"the first reference sets the output size and is {img.size[0]} x {img.size[1]} px; the "
                                 f"budget grid is {size} x {size} (BP_GRID={size // PX}): crop or resize it to {size} x {size}"
                                 f"{other}")
            out.append(img)
            continue
        w, h = img.size
        if w * h > MULTI_REFERENCE_MAX_PIXELS:
            s = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (w * h))
            w, h = int(w * s), int(h * s)
            img = img.resize((w, h), Image.LANCZOS)
        w16, h16 = w // PX * PX, h // PX * PX
        if not w16 or not h16:
            raise ValueError(f"reference {i + 1} is {w} x {h} px: smaller than one {PX} px cell")
        left, top = (w - w16) // 2, (h - h16) // 2
        out.append(img.crop((left, top, left + w16, top + h16)))
    return out


@torch.no_grad()
def encode_images(images, vae, family):
    """Prepared PIL images (prepare_images) -> References, through `vae` (encoders/vae.Flux2VAE: the posterior mean,
    normalised, as the released pipelines encode references), on the VAE's device."""
    import numpy as np
    device = vae.mean.device
    lat = []
    for img in images:
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        lat.append(vae.encode(x[None].to(device))[0])
    return References.from_latents(lat, family)


@dataclass
class References:
    tokens: torch.Tensor                # (Nr, 128) normalised latent cells, references in order, each row-major
    positions: torch.Tensor             # (Nr, 4) float64 rotary ids in the family's convention (unscaled h, w)
    sizes: tuple                        # ((h, w), ...) latent cells of each reference
    family: str                         # "flux2" or "flux3" (the arch)
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_latents(cls, latents, family):
        """latents: [(h, w, C) normalised latents] (Flux2VAE.encode(x)[0] of each reference)."""
        arch = "flux3" if str(family).startswith("flux3") else "flux2"
        toks, pos, sizes = [], [], []
        for i, z in enumerate(latents):
            if z.dim() == 4:
                if z.shape[0] != 1:
                    raise ValueError("one reference per latent: (h, w, C) or (1, h, w, C)")
                z = z[0]
            h, w, c = z.shape
            p = torch.zeros(h, w, 4, dtype=torch.float64)
            p[..., 1] = torch.arange(h, dtype=torch.float64)[:, None]
            p[..., 2] = torch.arange(w, dtype=torch.float64)[None, :]
            if arch == "flux2":
                p[..., 0] = FLUX2_TIME_SCALE + FLUX2_TIME_SCALE * i
            else:
                p[..., 3] = -(1 + i)
            toks.append(z.reshape(h * w, c))
            pos.append(p.reshape(h * w, 4))
            sizes.append((h, w))
        if not toks:
            raise ValueError("no references")
        return cls(torch.cat(toks).float(), torch.cat(pos), tuple(sizes), arch)

    @property
    def length(self):
        return self.tokens.shape[0]

    def on(self, device, dtype):
        """The tokens on `device` in `dtype`, (1, Nr, C) (cached)."""
        key = ("tok", str(device), dtype)
        if key not in self._cache:
            self._cache[key] = self.tokens.to(device, dtype)[None]
        return self._cache[key]

    def pe(self, trunk, device):
        """Rotary tables of the references for `trunk` (cos, sin), (1, Nr, Dh / 2) each (cached)."""
        from .flux2_trunk import rope_tables
        key = ("pe", str(device))
        if key not in self._cache:
            pos = self.positions.to(device).clone()
            if self.family == "flux3":
                r = float(trunk.cfg.image_rope_scale)
                pos[:, 1:3] *= r
            self._cache[key] = rope_tables(pos[None], trunk.cfg.axes_dim, trunk.cfg.theta)
        return self._cache[key]
