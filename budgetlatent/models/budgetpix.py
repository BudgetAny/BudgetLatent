"""BudgetPix: a released FLUX trunk run on a quadtree of 16 / 32 / 64 / 128 px tokens.

    model = load_model(device="cuda")                              # FLUX.2 [klein] 4B Base, added modules at zero
    model = load_model(device="cuda", family="flux3")              # FLUX 3 Image Klein Base
    model = load_model(device="cuda", family="flux3_distilled")    # FLUX 3 Image Klein (a flow map: v = model(..., t_ref=))
    v = model(z_t, t, ctx, layout=pack(leaf_masks), txt_len=w)     # velocity dz/dt on the (g, g, 128) latent

(load_model is in weights.py.)  A token of scale s covers s x s latent cells.  Going in, the cells are embedded by
the trunk's own image projection and the cells of a coarse token are merged by the cascade mixers; going out, the
trunk's prediction for a coarse token is opened back into its cells by the cascade refiners, and the high
frequencies the trunk never saw are estimated analytically from the noisy input (the Wiener skip).  Fine tokens
(s = 1) bypass all of it, so with the dense layout the model is the released one.

Two trunk families (families.py), one adaptive-token part:
  flux2   FLUX.2 [klein] (flux2_trunk.Flux2Trunk): double + single stream blocks, a 512-token Qwen3 text window
  flux3   FLUX 3 Image Klein (flux3_trunk.Flux3Trunk): per-mode + single blocks, Qwen3-VL text windows rounded up to
          multiples of 80 tokens; FLUX 3 Image Klein Base, or the step-distilled FLUX 3 Image Klein, a flow map
          (Flux3Config.flow_map: each pass also takes t_ref, the time the sampling step jumps to)
Both trunks expose the same interface (embed_cells, time_vec, project_text, text_pe, image_pe, run_trunk, final), so
BudgetPixMixin below is shared; the added modules take their sizes from the trunk config.

Conventions: latents are (B, g, g, 128), channels last, normalised by the VAE's BatchNorm statistics
(encoders/vae.py).  t in [0, 1] with 0 = noise and 1 = clean, z_t = t x + (1 - t) eps, and the model returns
dz/dt = x - eps (the trunks' own time is sigma = 1 - t and their raw output the velocity eps - x; both are flipped
at the boundary).  txt_len (FLUX 3 only): each row's text window; None = the whole text tensor.  t_ref (FLUX 3
Image Klein, the flow map, only): the time each row's step jumps to; its model then returns the average velocity
over [t, t_ref].
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..tokenization.layout import COARSE, GRID, Layout
from .added_modules import CascadeMixer2D, CascadeRefineStage2D, MipNeRFIPE2DV3, hf_skip_gain
from .convert import is_adapter
from .flux2_trunk import Flux2Trunk, TrunkConfig
from .flux3_trunk import Flux3Config, Flux3Trunk

SCALE_INDEX = {1: 0, 2: 1, 4: 2, 8: 3}  # row of scale_emb
FP32_ONLY = ("refiners.",)              # tiny, and run outside autocast


@dataclass
class BPConfig:                         # `budgetpix_s8` of configs/*.yaml
    grid: int = GRID
    mix_bottleneck: int = 256
    use_ipe_2d: bool = True
    rope_box_damping: bool = True
    refine_width: int = 128
    refine_depth: int = 2
    refine_heads: int = 4
    refine_mlp_hidden: int = 352
    use_hf_skip: bool = True
    hf_rho: float = 0.25

    @classmethod
    def from_config(cls, d):
        d = dict(d or {})
        if "use_ipe_2d_v3" in d:                                     # an older name for the same switch
            d.setdefault("use_ipe_2d", d["use_ipe_2d_v3"])
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def add_modules(mod, cfg, bp: BPConfig):
    """The four added parameter groups, as attributes of `mod` (sizes from the trunk config)."""
    dim = cfg.hidden_size
    mod.scale_emb = nn.Parameter(torch.zeros(4, dim))               # row 0 (s = 1) is never read
    mod.mixers = nn.ModuleDict({str(s): CascadeMixer2D(dim, bp.mix_bottleneck) for s in COARSE})
    mod.ipe_2d = MipNeRFIPE2DV3(dim, grid=bp.grid)
    mod.refiners = nn.ModuleDict({str(s): CascadeRefineStage2D(cfg.in_channels, dim, bp.refine_width, bp.refine_depth,
                                                               bp.refine_heads, bp.refine_mlp_hidden) for s in COARSE})
    return mod


class BudgetPixMixin:
    """The adaptive-token part, on top of a trunk with the interface of Flux2Trunk / Flux3Trunk."""

    supports_prepare = False            # BudgetPixFlux2: prepare(layout) for the steps that share a layout

    def trunk_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if not is_adapter(n)]

    # ---- positions ----------------------------------------------------------------------------------------
    def layout_pe(self, layout: Layout):
        """Rotary tables of the image tokens at the token centres; a coarse token's spatial planes are damped by the
        variance of its box, (s^2 - 1) / 12 cells^2 (the trunk applies its own position scale)."""
        cy, cx = layout.centers()
        var = layout.box_var() if self.bp.rope_box_damping and not layout.is_dense else None
        return self.image_pe(cy, cx, var)

    # ---- tokens in ----------------------------------------------------------------------------------------
    def pool(self, sub, s):
        """(M, s, s, D) cell embeddings of M tokens -> (M, D), one 2x2 merge per stage."""
        m, g, d = sub.shape[0], s, sub.shape[-1]
        for k in COARSE:
            if k > s:
                break
            blocks = sub.reshape(m, g // 2, 2, g // 2, 2, d).permute(0, 1, 3, 2, 4, 5).reshape(-1, 2, 2, d)
            g //= 2
            sub = self.mixers[str(k)](blocks).view(m, g, g, d)
        return sub.view(m, d)

    def embed_tokens(self, z, layout: Layout):
        b, g = z.shape[0], z.shape[1]
        cells = self.embed_cells(z.flatten(1, 2))                    # (B, g * g, D): the trunk's own embedding
        x = cells.new_zeros(b, layout.length, cells.shape[-1])
        grid = cells.view(b, g, g, -1)
        cy, cx = layout.centers()
        for s, (bi, ti, gy, gx) in layout.groups.items():
            if s == 1:
                x[bi, ti] = cells[bi, gy * g + gx]
                continue
            o = torch.arange(s, device=z.device)
            sub = grid[bi[:, None, None], (gy[:, None] + o)[:, :, None], (gx[:, None] + o)[:, None, :]]
            tok = self.pool(sub, s) + self.scale_emb[SCALE_INDEX[s]].to(cells.dtype)
            if self.bp.use_ipe_2d:
                tok = tok + self.ipe_2d(cy[bi, ti], cx[bi, ti], s).to(cells.dtype)
            x[bi, ti] = tok.to(cells.dtype)
        return x

    # ---- tokens out ---------------------------------------------------------------------------------------
    def decode_tokens(self, hidden, base, z, sigma, layout: Layout):
        """hidden (B, N, D) modulated final hidden states, base (B, N, C) the trunk's per-token prediction
        -> the raw prediction eps - x on the cell grid, (B, g, g, C) fp32."""
        b, g, _, c = z.shape
        out = torch.zeros(b, g, g, c, dtype=torch.float32, device=z.device)
        for s, (bi, ti, gy, gx) in layout.groups.items():
            if s == 1:
                out[bi, gy, gx] = base[bi, ti].float()
                continue
            with torch.autocast(device_type=z.device.type, enabled=False):
                lat, cond, feat = base[bi, ti].float()[:, None, None, :], hidden[bi, ti].float(), None
                for k in reversed(COARSE):
                    if k <= s:
                        lat, feat = self.refiners[str(k)](lat, feat, cond)
                o = torch.arange(s, device=z.device)
                yy, xx = (gy[:, None] + o)[:, :, None], (gx[:, None] + o)[:, None, :]
                if self.bp.use_hf_skip:
                    z_sub = z.float()[bi[:, None, None], yy, xx]
                    w = hf_skip_gain(sigma.float(), self.bp.hf_rho)[bi][:, None, None, None]
                    lat = lat + w * (z_sub - z_sub.mean(dim=(1, 2), keepdim=True))
                out[bi[:, None, None], yy, xx] = lat
        return out

    # ---- forward ------------------------------------------------------------------------------------------
    @property
    def flow_map(self) -> bool:
        """Whether the trunk is a flow map (FLUX 3 Image Klein, step-distilled): every pass takes t_ref."""
        return bool(getattr(self.cfg, "flow_map", False))

    def forward(self, z, t, ctx, layout=None, ctx_projected=False, txt_len=None, t_ref=None, txt_cache=None, refs=None):
        """z (B, g, g, C) noisy latent, t (B,), ctx (B, Lt, context dim) text states (or (B, Lt, D) already
        projected), layout from layout.pack, or prepare(layout, Lt) (FLUX.2; None = the released dense model),
        txt_len (B,) text windows (FLUX 3), t_ref (B,) the time a flow map (FLUX 3 Image Klein) jumps to -- required
        by it, refused by the others.  Returns dz/dt, (B, g, g, C) fp32 (a flow map: the average velocity over
        [t, t_ref]).  txt_cache (FLUX 3, caches_text): a dict per sample, see Flux3Trunk.run_trunk.  refs
        (models/refs.References): reference images (editing), dense and never budgeted; None = text to image."""
        if t_ref is not None and not self.flow_map:
            raise ValueError("t_ref is required by, and only by, a flow-map (step-distilled) FLUX 3 model")
        cache = {} if txt_cache is None else {"txt_cache": txt_cache}
        if refs is not None:
            cache["refs"] = refs
        if layout is None:
            if t_ref is None:
                return self.forward_dense(z, t, ctx, ctx_projected, txt_len=txt_len, **cache)
            return self.forward_dense(z, t, ctx, ctx_projected, txt_len=txt_len, t_ref=t_ref, **cache)
        if isinstance(layout, PreparedLayout):
            if refs is not None:
                return self.forward_prepared(z, t, ctx, layout, ctx_projected, txt_len, refs=refs)
            return self.forward_prepared(z, t, ctx, layout, ctx_projected, txt_len)
        sigma = 1.0 - t.float()
        vref = self.ref_vec(t_ref) if self.flow_map else None
        vec = self.time_vec(sigma) if vref is None else self.time_vec(sigma) + vref
        txt = ctx if ctx_projected else self.project_text(ctx)
        kw = dict(cache) if vref is None else {"vref": vref, **cache}
        img = self.run_trunk(self.embed_tokens(z, layout), txt, vec, self.layout_pe(layout),
                             self.text_pe(txt.shape[1], z.device), layout.valid if layout.padded else None, txt_len, **kw)
        hidden, base = self.final(img, vec)
        return -self.decode_tokens(hidden, base, z, sigma, layout)

    def forward_prepared(self, z, t, ctx, layout, ctx_projected=False, txt_len=None, refs=None):
        raise NotImplementedError(f"{type(self).__name__} samples with per-step layouts (no prepare())")

    def forward_cfg(self, z, t, txt_cond, txt_uncond, layout, scale, len_cond=None, len_uncond=None, txt_cache=None,
                    refs=None):
        """One fused [cond; uncond] pass with projected text: v_uncond + scale * (v_cond - v_uncond).  layout: a
        Layout of the B images, or prepare(layout, Lt, repeat=2) (FLUX.2).  Texts of different lengths (FLUX 3: the
        empty prompt's window is 80 tokens) are padded to the longer and masked (len_cond / len_uncond: the rows'
        windows, None = the whole tensor).  A flow map (FLUX 3 Image Klein) has no unconditional branch: refused."""
        if self.flow_map:
            raise ValueError("a flow map (FLUX 3 Image Klein) samples without classifier-free guidance (cfg 1.0, t_ref)")
        b = z.shape[0]
        lc, lu = txt_cond.shape[1], txt_uncond.shape[1]
        txt_len = None
        if lc != lu or len_cond is not None or len_uncond is not None:
            full = lambda n: torch.full((b,), n, dtype=torch.long)
            lens = [full(lc) if len_cond is None else torch.as_tensor(len_cond).cpu().long(),
                    full(lu) if len_uncond is None else torch.as_tensor(len_uncond).cpu().long()]
            length = max(lc, lu)
            pad = lambda x: torch.nn.functional.pad(x, (0, 0, 0, length - x.shape[1])) if x.shape[1] < length else x
            txt_cond, txt_uncond = pad(txt_cond), pad(txt_uncond)
            txt_len = torch.cat(lens)
        if layout is not None and not isinstance(layout, PreparedLayout):
            layout = layout.repeat(2)
        kw = {} if txt_cache is None else {"txt_cache": txt_cache}
        if refs is not None:                                          # both rows see the references (the released pipelines)
            kw["refs"] = refs
        v = self.forward(torch.cat([z, z]), torch.cat([t, t]), torch.cat([txt_cond, txt_uncond]), layout,
                         ctx_projected=True, txt_len=txt_len, **kw)
        v_c, v_u = v.chunk(2)
        return v_u + scale * (v_c - v_u)


class BudgetPixFlux2(BudgetPixMixin, Flux2Trunk):
    """BudgetPix on FLUX.2 [klein] 4B Base, or on FLUX.2 [klein] 4B, its step-distilled release (the same trunk)."""

    supports_prepare = True

    def __init__(self, cfg: TrunkConfig = TrunkConfig(), bp: BPConfig = BPConfig()):
        super().__init__(cfg)
        self.bp = bp
        add_modules(self, cfg, bp)

    # ---- a layout prepared once for the steps that reuse it --------------------------------------------------
    def prepare(self, layout: Layout, text_len: int, repeat: int = 1) -> "PreparedLayout":
        """Everything a forward computes from the layout alone, computed once: the sampler keeps a layout for
        several steps (it is refreshed every 4 by default), and with this the steps in between compute only what
        depends on z and t.  repeat=2 for forward_cfg's fused [cond; uncond] batch.  Same results as passing the
        layout itself (tests/test_units.py checks it); the layout-only embeddings (scale embedding, 2-D IPE) are
        computed by the first forward, in that forward's autocast state, and kept."""
        lay = layout.repeat(repeat)
        b, n, g = lay.batch, lay.length, lay.cell_scale.shape[1]
        device = lay.scale.device
        pe = self.join_pe(self.text_pe(text_len, device), self.layout_pe(lay))
        cy, cx = lay.centers()
        embed, decode = {}, {}
        for s, (bi, ti, gy, gx) in lay.groups.items():
            tok = bi * n + ti                                            # flat (B * N) token slot
            if s == 1:
                cell = bi * (g * g) + gy * g + gx                        # flat (B * g * g) cell
            else:
                o = torch.arange(s, device=device)
                cell = bi[:, None, None] * (g * g) + (gy[:, None] + o)[:, :, None] * g + (gx[:, None] + o)[:, None, :]
            embed[s] = (tok, cell, cy[bi, ti], cx[bi, ti])
            decode[s] = (tok, cell, bi)
        return PreparedLayout(lay, pe, lay.valid if lay.padded else None, embed, decode, text_len, g)

    def embed_prepared(self, z, p: "PreparedLayout"):
        """embed_tokens for a prepared layout."""
        b = z.shape[0]
        cells = self.img_in(z.flatten(1, 2)).flatten(0, 1)             # (B * g * g, D): FLUX.2's own embedding
        x = cells.new_zeros(b * p.layout.length, cells.shape[-1])
        for s, (tok, cell, cy, cx) in p.embed.items():
            if s == 1:
                x[tok] = cells[cell]
                continue
            key = (s, cells.dtype)
            if key not in p.added:                                       # layout-only: computed by the first step
                ipe = self.ipe_2d(cy, cx, s).to(cells.dtype) if self.bp.use_ipe_2d else None
                p.added[key] = (self.scale_emb[SCALE_INDEX[s]].to(cells.dtype), ipe)
            emb, ipe = p.added[key]
            tok_s = self.pool(cells[cell], s) + emb
            if ipe is not None:
                tok_s = tok_s + ipe
            x[tok] = tok_s.to(cells.dtype)
        return x.view(b, p.layout.length, -1)

    def decode_prepared(self, hidden, base, z, sigma, p: "PreparedLayout"):
        """decode_tokens for a prepared layout."""
        b, g, _, c = z.shape
        out = torch.zeros(b * g * g, c, dtype=torch.float32, device=z.device)
        hid, bas = hidden.flatten(0, 1), base.flatten(0, 1)
        zf, w_all = None, None
        for s, (tok, cell, bi) in p.decode.items():
            if s == 1:
                out[cell] = bas[tok].float()
                continue
            with torch.autocast(device_type=z.device.type, enabled=False):
                lat, cond, feat = bas[tok].float()[:, None, None, :], hid[tok].float(), None
                for k in reversed(COARSE):
                    if k <= s:
                        lat, feat = self.refiners[str(k)](lat, feat, cond)
                if self.bp.use_hf_skip:
                    if zf is None:
                        zf = z.float().flatten(0, 2)
                        w_all = hf_skip_gain(sigma.float(), self.bp.hf_rho)
                    z_sub = zf[cell]
                    lat = lat + w_all[bi][:, None, None, None] * (z_sub - z_sub.mean(dim=(1, 2), keepdim=True))
                out[cell] = lat
        return out.view(b, g, g, c)

    def forward_prepared(self, z, t, ctx, layout, ctx_projected=False, txt_len=None, refs=None):
        """forward with prepare(layout, Lt)'s result; refs: reference images after the image tokens (with_refs)."""
        sigma = 1.0 - t.float()
        vec = self.time_vec(sigma)
        txt = ctx if ctx_projected else self.txt_in(ctx)
        if layout.text_len != txt.shape[1] or layout.layout.batch != z.shape[0]:
            raise ValueError(f"layout prepared for {layout.layout.batch} images and {layout.text_len} text tokens, "
                             f"called with {z.shape[0]} and {txt.shape[1]}")
        if txt_len is not None and bool((torch.as_tensor(txt_len) != txt.shape[1]).any()):
            raise ValueError("FLUX.2 attends to its whole 512-token text window; per-row text lengths are FLUX 3's")
        if refs is None:
            img = self.run_blocks(self.embed_prepared(z, layout), txt, vec, layout.pe, layout.mask)
        else:
            if layout.mask is not None:
                raise ValueError("reference images need unpadded image tokens (every row the same token count)")
            n = layout.layout.length
            img, pe_r = self.with_refs(self.embed_prepared(z, layout), (layout.pe[0][:, 0, -n:], layout.pe[1][:, 0, -n:]), refs)
            pe = tuple(torch.cat((p[:, :, :-n], q[:, None].expand(p.shape[0], -1, -1, -1)), dim=2)
                       for p, q in zip(layout.pe, pe_r))
            img = self.run_blocks(img, txt, vec, pe, None)[:, :n]
        hidden, base = self.final_layer(img, vec)
        return -self.decode_prepared(hidden, base, z, sigma, layout)


class BudgetPixFlux3(BudgetPixMixin, Flux3Trunk):
    """BudgetPix on FLUX 3 Image Klein Base, or on FLUX 3 Image Klein, its step-distilled flow map (cfg.flow_map): text
    to image, and editing with reference images (refs=; the trunk built with image_cond).  It samples with per-step
    layouts: no prepare()."""

    def __init__(self, cfg: Flux3Config = Flux3Config(), bp: BPConfig = BPConfig()):
        super().__init__(cfg)
        self.bp = bp
        add_modules(self, cfg, bp)


class PreparedLayout:
    """BudgetPixFlux2.prepare's result: the layout (repeated for a fused batch), the joint [text, image] rotary
    tables, the key mask (None: no padding), flat index tensors of the token encoder and decoder per scale, and
    the layout-only embeddings once the first forward has made them."""
    __slots__ = ("layout", "pe", "mask", "embed", "decode", "text_len", "grid", "added")

    def __init__(self, layout, pe, mask, embed, decode, text_len, grid):
        self.layout, self.pe, self.mask, self.embed, self.decode = layout, pe, mask, embed, decode
        self.text_len, self.grid, self.added = text_len, grid, {}


def count_parameters(model):
    """(released trunk, added modules)."""
    trunk = sum(p.numel() for n, p in model.named_parameters() if not is_adapter(n))
    added = sum(p.numel() for n, p in model.named_parameters() if is_adapter(n))
    return trunk, added
