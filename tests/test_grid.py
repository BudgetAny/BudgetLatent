"""CPU checks of the parts of the code that depend on the grid size, at the grid BP_GRID sets (no weights needed).

    BP_GRID=64 python tests/test_grid.py          # 1024 x 1024: 64 x 64 cells, 64 .. 4096 tokens
    python tests/test_grid.py                     # the default grid (32) passes the same checks

tests/test_units.py and tests/test_units_flux3.py check the code at the 512 x 512 grid; this file checks that nothing
in the layouts, pyramids, sampler noise, either trunk's model (FLUX 3 also as the flow map) or --init still assumes 32
cells per side.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch

from budgetlatent.models import added_modules as M
from budgetlatent.models.budgetpix import BPConfig, BudgetPixFlux2, BudgetPixFlux3
from budgetlatent.tokenization import layout as L
from test_units import OK, TINY, check, rand_pyr, randomize, tiny_inputs


def coverage(active, g):
    b = next(iter(active.values())).shape[0]
    cov = torch.zeros(b, g, g)
    for s, m in active.items():
        cov += m.float().repeat_interleave(s, 1).repeat_interleave(s, 2)
    return cov


def main():
    g = L.GRID
    print(f"[grid {g}: {g * L.CELL_PX} x {g * L.CELL_PX} px, {L.N_MIN} .. {L.N_DENSE} tokens]")
    check("constants", L.N_DENSE == g * g and L.N_MIN == (g // 8) ** 2 and L.MAX_SPLITS == (L.N_DENSE - L.N_MIN) // 3,
          f"N_DENSE {L.N_DENSE}, N_MIN {L.N_MIN}, MAX_SPLITS {L.MAX_SPLITS}")
    vals = [L.snap_tokens(b) for b in (1.0, 0.5, 0.3, 0.1, 1 / 64, 0.001)]
    check("snap_tokens", vals[0] == L.N_DENSE and vals[-2] == vals[-1] == L.N_MIN
          and all((v - L.N_MIN) % 3 == 0 and L.N_MIN <= v <= L.N_DENSE for v in vals), str(vals))
    ts = L.flux2_klein_timesteps(50, L.N_DENSE)
    check("schedule", ts[0] == 0.0 and ts[-1] == 1.0 and all(b > a for a, b in zip(ts, ts[1:])),
          f"mu({L.N_DENSE}, 50) = {L.compute_empirical_mu(L.N_DENSE, 50):.6f}; t[1] = {ts[1]:.6f}, t[49] = {ts[49]:.6f}")
    ts4 = L.schedule(4, "flux2_distilled")
    want = {32: (0.0, 0.041915, 0.116018, 0.282503, 1.0), 64: (0.0, 0.032616, 0.091856, 0.232800, 1.0)}.get(g)
    check("FLUX.2 [klein] 4B schedule", ts4 == L.flux2_klein_timesteps(4, L.N_DENSE)
          and (want is None or max(abs(a - b) for a, b in zip(ts4, want)) < 1e-6),
          f"1 - BFL get_schedule(4, {L.N_DENSE}): t = {[round(v, 4) for v in ts4]}")
    gen = torch.Generator().manual_seed(0)
    t, first = L.sample_times(20000, gen)
    check("training times", abs(float(first.float().mean()) - 0.10) < 0.01 and float(t.max()) < 1.0,
          f"share at t = 0 {float(first.float().mean()):.3f}, median t {float(t[~first].median()):.3f}")

    pyr = rand_pyr(4, gen)
    for b in (1.0, 0.5, 0.3, 0.1, 0.02, 1 / 64):
        act = L.build_exact_budget_layout_s8(pyr, b)
        n = L.count_tokens(act)
        check(f"exact budget {b:g}", bool((n == L.snap_tokens(b)).all()) and bool((coverage(act, g) == 1).all()),
              f"{int(n[0])} tokens on every image, the grid tiled once")
    mix = L.LayoutMix()
    plan = L.draw_plan(256, mix, gen)
    act, mode = L.training_layouts(rand_pyr(256, gen), mix, gen, plan)
    n = L.count_tokens(act)
    check("training mixture", int(n.min()) >= L.N_MIN and int(n.max()) <= L.N_DENSE and bool((coverage(act, g) == 1).all()),
          f"256 layouts, {int(n.min())} .. {int(n.max())} tokens (mean {float(n.float().mean()):.0f}), modes {torch.bincount(mode, minlength=4).tolist()}")

    pix = L.pixel_entropy_pyramid_s8(torch.rand(2, 3, g * L.CELL_PX, g * L.CELL_PX) * 2 - 1)
    lat = L.noise_latent_pyramid_s8(torch.randn(2, g, g, 128))
    check("detail pyramids", all(pix[s].shape == (2, g // s, g // s) == lat[s].shape for s in L.COARSE),
          "pixel entropy and latent proxy on the grid's blocks")
    from budgetlatent.inference.sampling import initial_noise
    check("sampler noise", tuple(initial_noise(2, channels=16).shape) == (2, g, g, 16))

    # the IPE's positions are relative to the grid: the same box (as a fraction of the image) at grid 32 and at grid 64
    # gets the same full-box features; the excess-over-one-cell half differs, since a cell is half as wide at 64
    f32 = M.MipNeRFIPE2DV3(8, grid=32).features(torch.tensor([3.0, 20.0]), torch.tensor([11.0, 0.0]), 2).view(2, 8, 8)
    f64 = M.MipNeRFIPE2DV3(8, grid=64).features(torch.tensor([6.5, 40.5]), torch.tensor([22.5, 0.5]), 4).view(2, 8, 8)
    d_full, d_exc = float((f32[..., :4] - f64[..., :4]).abs().max()), float((f32[..., 4:] - f64[..., 4:]).abs().max())
    check("IPE positions relative to the grid", d_full < 1e-6, f"full-box half max |diff| {d_full:.1e} (excess half {d_exc:.3f}, by design)")

    torch.manual_seed(0)
    model = BudgetPixFlux2(TINY, BPConfig()).eval()
    z, t, ctx = tiny_inputs(b=2, g=g, gen=gen)
    with torch.no_grad():
        d = float((model(z, t, ctx, L.pack(L.dense_layout(2))) - model.forward_dense(z, t, ctx)).abs().max())
    check("dense layout == plain trunk", d == 0.0 and model.ipe_2d.grid == g, f"max |diff| {d:.1e}; IPE grid {model.ipe_2d.grid}")
    randomize(model, gen)
    with torch.no_grad():
        lay = L.pack(L.build_exact_budget_layout_s8(rand_pyr(2, gen), 0.1))
        v = model(z, t, ctx, lay)
        cy, cx = lay.centers()
    check("adaptive forward", v.shape == z.shape and bool(torch.isfinite(v).all()) and float(cy.max()) < g and float(cx.max()) < g,
          f"{lay.length} tokens per image, centres up to {float(cy.max()):.1f}, output {tuple(v.shape)}")

    from budgetlatent.training import trainlib
    with tempfile.TemporaryDirectory() as tmp:
        names = [n for n, _ in model.named_parameters()]
        sd = {n: p.detach().clone() for n, p in model.named_parameters()}
        torch.save({n: sd[n] for n in names[::2]}, os.path.join(tmp, "ema-00.pt"))
        torch.save({n: sd[n] for n in names[1::2]}, os.path.join(tmp, "ema-01.pt"))
        torch.manual_seed(1)
        fresh = BudgetPixFlux2(TINY, BPConfig())
        count = trainlib.load_init(fresh, tmp)
        d = max(float((p - sd[n]).abs().max()) for n, p in fresh.named_parameters())
        check("--init from EMA shards", count == len(names) and d == 0.0, f"{count} tensors from 2 shards, max |diff| {d:.1e}")

    # FLUX 3 at this grid: the same layouts on the FLUX 3 trunk, its resolution-independent schedule
    from test_units_flux3 import TINY3, inputs3
    ts3 = L.schedule(50, "flux3")
    check("FLUX 3 schedule", ts3[0] == 0.0 and ts3[-1] == 1.0 and all(b > a for a, b in zip(ts3, ts3[1:]))
          and abs((1 - ts3[1]) - 6.93 * 0.98 / (1 + 5.93 * 0.98)) < 1e-12, f"constant shift {L.time_shift('flux3')} at every grid")
    torch.manual_seed(0)
    m3 = BudgetPixFlux3(TINY3, BPConfig()).eval()
    z3, t3, c3, l3 = inputs3(b=2, g=g, gen=gen)
    with torch.no_grad():
        d = float((m3(z3, t3, c3, L.pack(L.dense_layout(2)), txt_len=l3) - m3.forward_dense(z3, t3, c3, txt_len=l3)).abs().max())
    check("FLUX 3 dense layout == plain trunk", d == 0.0 and m3.ipe_2d.grid == g, f"max |diff| {d:.1e}")
    randomize(m3, gen)
    with torch.no_grad():
        lay = L.pack(L.build_exact_budget_layout_s8(rand_pyr(2, gen), 0.1))
        v = m3(z3, t3, c3, lay, txt_len=l3)
        pe = m3.layout_pe(lay)
    check("FLUX 3 adaptive forward", v.shape == z3.shape and bool(torch.isfinite(v).all()) and pe[0].shape[1] == lay.length,
          f"{lay.length} tokens per image, output {tuple(v.shape)}")

    # FLUX 3 Image Klein (the flow map) at this grid: its 16-step schedule, t_ref through the dense and adaptive passes
    from budgetlatent.models.flux3_trunk import Flux3Config
    ts16 = L.schedule(16, "flux3_distilled")
    check("FLUX 3 Image Klein schedule", ts16 == L.schedule(16, "flux3") and abs((1 - ts16[1]) - 6.93 * 0.9375 / (1 + 5.93 * 0.9375)) < 1e-12,
          f"16 steps, constant shift {L.time_shift('flux3_distilled')} at every grid")
    torch.manual_seed(0)
    m3d = randomize(BudgetPixFlux3(Flux3Config(**{**TINY3.to_dict(), "flow_map": True}), BPConfig()).eval(), gen)
    t_ref = torch.clamp(t3 + 0.1, max=1.0)
    with torch.no_grad():
        d = float((m3d(z3, t3, c3, L.pack(L.dense_layout(2)), txt_len=l3, t_ref=t_ref)
                   - m3d.forward_dense(z3, t3, c3, txt_len=l3, t_ref=t_ref)).abs().max())
        v = m3d(z3, t3, c3, lay, txt_len=l3, t_ref=t_ref)
    check("FLUX 3 Image Klein dense layout == plain trunk, adaptive forward (t_ref)", d == 0.0 and v.shape == z3.shape
          and bool(torch.isfinite(v).all()), f"max |diff| {d:.1e}")
    print(f"\n{len(OK)} checks passed (grid {g})")


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    main()
