"""G-series: graphable sweep tests -- become paper figures.

Each test sweeps a parameter, saves figure data to results/figures/,
and checks the predicted curve shape. Includes the four T-series promotions.

Promotions from T-series:
    G3  <- T10   G8  <- T7   G9  <- T8   G11 <- T3

Run:
    conda activate Spectral
    python -m pytest tests/test_G.py -v
  or
    python -m tests.test_G

Tests
-----
G1   J_TIR v-sweep              -- J_theta(v), T(v), product lands on 4eta/4eta^3
G2   theta_i sweep Jacobians    -- J_||, J_perp, det vs theta for several eta
G3   Moving-boundary sweep      -- dI/dA continuous, =0 outside window, no kink at edges  [<-T10]
G4   Multi-fluorophore cond.    -- condition number vs emission-peak separation -> vertical asymptote
G5   Conditioning heatmap       -- condition number vs (separation, species count k)
G6   Measurement diversity      -- condition number vs # angle+pol diversity measurements
G7   Index contrast -> 0        -- condition number diverges as n_sub -> n_film
G8   d across FSR periods       -- periodic conditioning structure synced to FSR           [<-T7]
G9   Illuminant slope k         -- d(absorbed power)/dlambda_ex smooth from exactly 0      [<-T8]
G10  Spectral bandwidth         -- all 5 singular values vs measurement bandwidth
G11  Wrong vs correct adjoint   -- |wrong-correct| gradient error vs Stokes shift, =0 at 0 [<-T3]
G12  lambda_ex invariance       -- rendered pixel vs lambda_ex: flat line within MC noise
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
torch.set_default_dtype(torch.float64)

from tests.harness import Reporter, StructResult

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"


# ---------------------------------------------------------------------------
# G1: J_TIR v-sweep
# X-axis: v, 1 -> 0
# Y-axis: J_theta(v), T(v), product J_TIR(v) -- s and p polarizations
# Predicted shape: J_theta diverges ~1/v, T->0 linearly, product lands exactly
#                  on 4*eta / 4*eta^3
# Failure: kink, overshoot, or product misses the asymptote
# ---------------------------------------------------------------------------

def test_G1() -> list[StructResult]:
    import csv
    from src.snell_jacobian import tir_jacobian, solid_angle_ratio
    from src.fresnel import fresnel_rs, fresnel_rp

    # Same locked params as T5 (eta=1.6, c=0.6) so this figure is the
    # full-sweep companion to T5's convergence-at-a-point check.
    eta = 1.6
    c   = 0.6
    N   = 240
    v   = torch.logspace(0.0, -12.0, N)     # 1 -> 1e-12, descending

    n_i   = torch.full((N,), eta)
    n_t   = torch.ones(N)
    cos_i = torch.full((N,), c)

    det = solid_angle_ratio(n_i, n_t, cos_i, v)     # J_theta(v) = eta^2*c/v

    # Stable T(v): the rational form used by tir_jacobian, no cancellation.
    Ts = 4.0 * eta * c * v / (eta * c + v) ** 2
    Tp = 4.0 * eta * c * v / (c + eta * v) ** 2

    # Naive T(v): textbook 1 - r^2. Loses precision as v->0 since r->1 and
    # r**2 is computed as a value near 1 before subtracting from 1.
    rs = fresnel_rs(n_i, n_t, cos_i, v)
    rp = fresnel_rp(n_i, n_t, cos_i, v)
    Ts_naive = 1.0 - rs ** 2
    Tp_naive = 1.0 - rp ** 2

    J_s = Ts * det
    J_p = Tp * det
    J_s_naive = Ts_naive * det
    J_p_naive = Tp_naive * det
    J_s_closed = tir_jacobian(v, n_i, n_t, cos_i, polarization="s")
    J_p_closed = tir_jacobian(v, n_i, n_t, cos_i, polarization="p")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURES_DIR / "G1_tir_jacobian_sweep.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["v", "J_theta", "T_s", "T_p", "J_TIR_s", "J_TIR_p"])
        for i in range(N):
            w.writerow([v[i].item(), det[i].item(), Ts[i].item(), Tp[i].item(),
                        J_s[i].item(), J_p[i].item()])

    # Check 1: J_theta ~ 1/v exactly -- log-log slope at small v must be -1.
    slope = ((torch.log(det[-1]) - torch.log(det[-20]))
             / (torch.log(v[-1]) - torch.log(v[-20])))
    slope_err = abs(slope.item() - (-1.0))

    # Check 2: T(v) linear near 0 -- T_s(v)/v -> 4/(eta*c), T_p(v)/v -> 4*eta/c.
    lim_s = 4.0 / (eta * c)
    lim_p = 4.0 * eta / c
    rel_Ts = abs((Ts[-1] / v[-1]).item() - lim_s) / lim_s
    rel_Tp = abs((Tp[-1] / v[-1]).item() - lim_p) / lim_p

    # Check 3: product lands on the TIR-collapse limits 4*eta, 4*eta^3.
    rel_prod_s = abs(J_s[-1].item() - 4.0 * eta) / (4.0 * eta)
    rel_prod_p = abs(J_p[-1].item() - 4.0 * eta ** 3) / (4.0 * eta ** 3)

    # Check 4: stable rational T(v) agrees with tir_jacobian's closed form at
    # every v in the sweep -- two independently-coded paths must match.
    max_rel_cross_s = ((J_s - J_s_closed).abs() / J_s_closed.abs()).max().item()
    max_rel_cross_p = ((J_p - J_p_closed).abs() / J_p_closed.abs()).max().item()

    # Check 5: monotonic, no kink/overshoot -- J_TIR must rise monotonically
    # as v decreases (array is descending in v, so diffs must be >= 0).
    # Threshold is relative since near the v->0 plateau the curve is flat and
    # consecutive values differ only by float64 roundoff.
    rel_diff_s = torch.diff(J_s) / J_s[:-1].abs()
    rel_diff_p = torch.diff(J_p) / J_p[:-1].abs()
    mono_s_viol = (rel_diff_s < -1e-8).sum().item()
    mono_p_viol = (rel_diff_p < -1e-8).sum().item()

    # Check 6 (landmine, cf. T13/T15): naive T = 1 - r^2 suffers catastrophic
    # cancellation as v->0 (r->1, r^2 computed near 1, then subtracted from 1).
    # rel error should scale like eps/v -- confirm it IS large at v=1e-12 and
    # small at v~1e-3, i.e. the naive path is fine far from TIR onset but
    # unusable extremely close to it. This is the numerical motivation for
    # tir_jacobian's rational closed form rather than composing R then 1-R.
    rel_naive_far  = (abs(J_s_naive[120].item() - J_s_closed[120].item())
                      / J_s_closed[120].item())          # v ~ 1e-6 region
    rel_naive_near = (abs(J_s_naive[-1].item() - J_s_closed[-1].item())
                      / J_s_closed[-1].item())            # v = 1e-12

    tol = 1e-8
    return [
        StructResult("G1", "J_theta(v) log-log slope = -1 (pure 1/v divergence)",
                     slope.item(), -1.0, slope_err, tol, slope_err < tol,
                     "fit between v=1e-12 tail points"),
        StructResult("G1", "T_s(v)/v -> 4/(eta*c) as v->0",
                     (Ts[-1] / v[-1]).item(), lim_s, rel_Ts, tol, rel_Ts < tol,
                     f"eta={eta}, c={c}, stable rational form"),
        StructResult("G1", "T_p(v)/v -> 4*eta/c as v->0",
                     (Tp[-1] / v[-1]).item(), lim_p, rel_Tp, tol, rel_Tp < tol,
                     f"eta={eta}, c={c}, stable rational form"),
        StructResult("G1", "J_TIR^s(v=1e-12) -> 4*eta",
                     J_s[-1].item(), 4.0 * eta, rel_prod_s, tol,
                     rel_prod_s < tol, "stable rational T(v), not naive 1-r^2"),
        StructResult("G1", "J_TIR^p(v=1e-12) -> 4*eta^3",
                     J_p[-1].item(), 4.0 * eta ** 3, rel_prod_p, tol,
                     rel_prod_p < tol, "stable rational T(v), not naive 1-r^2"),
        StructResult("G1", "stable T(v) vs tir_jacobian closed form, s-pol, max over sweep",
                     max_rel_cross_s, 0.0, max_rel_cross_s, tol, max_rel_cross_s < tol,
                     "two independent rational-form code paths must agree at every v"),
        StructResult("G1", "stable T(v) vs tir_jacobian closed form, p-pol, max over sweep",
                     max_rel_cross_p, 0.0, max_rel_cross_p, tol, max_rel_cross_p < tol,
                     "two independent rational-form code paths must agree at every v"),
        StructResult("G1", "J_TIR^s(v) monotonic over full sweep (no kink)",
                     float(mono_s_viol), 0.0, float(mono_s_viol), 0.5, mono_s_viol == 0,
                     f"{mono_s_viol} sign violations out of {N-1} steps"),
        StructResult("G1", "J_TIR^p(v) monotonic over full sweep (no kink)",
                     float(mono_p_viol), 0.0, float(mono_p_viol), 0.5, mono_p_viol == 0,
                     f"{mono_p_viol} sign violations out of {N-1} steps"),
        StructResult("G1", "naive T=1-r^2 accurate far from TIR onset (v~1e-6)",
                     rel_naive_far, 0.0, rel_naive_far, 1e-6, rel_naive_far < 1e-6,
                     "sanity: naive path is fine away from v->0"),
        StructResult("G1", "naive T=1-r^2 cancellation landmine confirmed at v=1e-12",
                     rel_naive_near, 0.0, rel_naive_near, 1e-9,
                     rel_naive_near > 1e-9,
                     f"naive rel err={rel_naive_near:.2e} vs stable rel err ~3e-16 "
                     "(7+ orders worse) -- confirms cancellation, motivates "
                     "tir_jacobian's rational form"),
    ]


# ---------------------------------------------------------------------------
# G2: theta_i sweep Jacobians
# X-axis: theta_i, 0 -> grazing
# Y-axis: J_||, J_perp, det -- several eta values
# Predicted shape: J_perp flat; J_||, det diverge together near critical theta;
#                  all curves touch at theta_i=0
# Failure: curves don't touch at 0; J_perp not flat
# ---------------------------------------------------------------------------

def test_G2() -> list[StructResult]:
    import csv
    from src.cauchy_ior import cos_theta_t
    from src.snell_jacobian import solid_angle_ratio

    # J_perp (azimuthal stretch, sin(theta_t)/sin(theta_i)) = eta everywhere.
    # J_par  (meridian stretch, dtheta_t/dtheta_i) = eta*cos_i/cos_t.
    # det (the T4/solid_angle_ratio quantity) = J_perp * J_par = eta^2*cos_i/cos_t.
    # eta>1 has a true critical angle (TIR); eta<1 never does and the meridian
    # stretch shrinks toward grazing instead of diverging -- both regimes are
    # swept so the figure shows the qualitative difference.
    etas = [0.67, 1.0, 1.5, 2.0]
    N = 300
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results: list[StructResult] = []
    with (FIGURES_DIR / "G2_theta_i_sweep.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eta", "theta_i_deg", "cos_i", "J_perp", "J_parallel", "det"])

        for eta in etas:
            n_i = torch.tensor(eta)
            n_t = torch.tensor(1.0)

            if eta > 1.0:
                theta_c = torch.asin(torch.tensor(1.0 / eta)).item()
                theta_max = 0.999 * theta_c
            else:
                theta_max = 0.999 * (torch.pi / 2)

            theta_i = torch.linspace(1e-6, theta_max, N)
            cos_i = torch.cos(theta_i)
            cos_t = cos_theta_t(cos_i, n_i, n_t)

            J_perp = torch.full((N,), eta)
            J_par  = eta * cos_i / cos_t
            det    = solid_angle_ratio(n_i.expand(N), n_t.expand(N), cos_i, cos_t)

            for i in range(N):
                w.writerow([eta, torch.rad2deg(theta_i[i]).item(), cos_i[i].item(),
                            J_perp[i].item(), J_par[i].item(), det[i].item()])

            tol = 1e-9

            # Check A: J_perp flat -- exactly eta at every theta_i.
            flat_err = (J_perp.std() / eta).item()
            results.append(StructResult(
                "G2", f"J_perp flat at eta={eta}",
                flat_err, 0.0, flat_err, tol, flat_err < tol,
                f"J_perp == eta == {eta} for all theta_i"))

            # Check B: all curves touch at theta_i -> 0 (J_par(0) == J_perp(0) == eta).
            rel_touch = abs(J_par[0].item() - eta) / eta
            results.append(StructResult(
                "G2", f"J_par(0) == J_perp(0) == eta at eta={eta}",
                J_par[0].item(), eta, rel_touch, 1e-4, rel_touch < 1e-4,
                "all curves touch at normal incidence"))

            # Check C: det == J_perp * J_par identically (algebraic consistency
            # between solid_angle_ratio and the tangent-plane factorization).
            max_rel_det = ((det - J_perp * J_par).abs() / (J_perp * J_par).abs()).max().item()
            results.append(StructResult(
                "G2", f"det == J_perp*J_par identically at eta={eta}",
                max_rel_det, 0.0, max_rel_det, tol, max_rel_det < tol,
                "det is the product of the two tangent-plane stretch factors"))

            # Check D: divergence/vanishing rate toward the domain edge.
            # eta>1: true critical angle -- J_par, det diverge together.
            # eta<1: no TIR -- J_par, det shrink toward 0 at grazing instead.
            ratio_par = (J_par[-1] / J_par[N // 2]).item()
            ratio_det = (det[-1] / det[N // 2]).item()
            if eta > 1.0:
                cond = ratio_par > 5.0 and ratio_det > 5.0
                note = f"eta>1 (TIR): J_par grows {ratio_par:.1f}x, det grows {ratio_det:.1f}x toward theta_c"
            elif eta < 1.0:
                cond = ratio_par < 0.5 and ratio_det < 0.5
                note = f"eta<1 (no TIR): J_par shrinks {ratio_par:.2f}x, det shrinks {ratio_det:.2f}x toward grazing"
            else:
                cond = abs(ratio_par - 1.0) < 1e-9 and abs(ratio_det - 1.0) < 1e-9
                note = "eta=1 (matched media): identity map, ratio stays exactly 1 everywhere"
            results.append(StructResult(
                "G2", f"edge-of-domain trend at eta={eta}",
                ratio_par, None, None, 1.0, cond, note))

    return results


# ---------------------------------------------------------------------------
# G3: Moving-boundary through window edges  [promoted from T10]
# X-axis: A (or B), lambda*(A) sweeping continuously through [lambda_min, lambda_max]
# Y-axis: lambda*(A); dI/dA -- true (FD), naive, boundary-term-only
# Predicted shape: lambda* smooth monotonic; dI/dA continuous everywhere,
#                  exactly 0 outside window, smooth ramp inside, no kink at edges
# Failure: jump/kink at lambda*=lambda_min or lambda_max
# ---------------------------------------------------------------------------

def test_G3() -> list[StructResult]:
    import csv
    from src.cauchy_ior import n_cauchy
    from src.snell_jacobian import tir_jacobian
    from src.gradient import lambda_star as lam_star_fn, dlambda_star_dA

    # Same TIR-in-glass setup as T12 (fixed cos_i via kappa=1/sin_i, Cauchy
    # glass -> vacuum), but here A is swept across a *wide* range so that
    # lambda*(A) crosses continuously through and past both window edges.
    # Design principle (see running_notes / memory): unlike T12 -- which
    # deliberately parked lambda* only 1.5 sigma from lambda_min to prove the
    # boundary term matters -- here the emission is centered in the window
    # with sigma small enough that e_n is negligible (~7.5 sigma) at BOTH
    # edges. That makes the moving-boundary term itself vanish smoothly as
    # lambda* approaches lambda_min/lambda_max, so gluing it to the "outside
    # window -> exactly 0" branch produces no visible kink.
    kappa  = 1.55                         # 1/sin_i, fixed incidence geometry
    sin_i  = 1.0 / kappa
    cos_i  = (1.0 - sin_i ** 2) ** 0.5
    sin2_i = sin_i ** 2
    B_val  = 8000.0

    lam_min, lam_max = 400.0, 700.0
    N    = 4000
    lam  = torch.linspace(lam_min, lam_max, N)
    dlam = (lam_max - lam_min) / (N - 1)

    lam_em, sig_f = 550.0, 20.0           # window center; 150nm/20nm = 7.5 sigma to either edge
    e_raw  = torch.exp(-0.5 * ((lam - lam_em) / sig_f) ** 2)
    norm_e = e_raw.sum() * dlam
    e_n    = e_raw / norm_e

    def e_n_at(lmb: float) -> float:
        return float(torch.exp(torch.tensor(-0.5 * (lmb - lam_em) ** 2 / sig_f ** 2)) / norm_e)

    def compute_I(A_scalar: float) -> float:
        A_t  = torch.tensor(A_scalar, dtype=torch.float64)
        n    = n_cauchy(lam, A_t, B_val)
        arg  = 1.0 - n ** 2 * sin2_i
        v    = torch.sqrt(arg.clamp(min=0.0))
        prop = (arg > 0.0).float()
        J    = tir_jacobian(v, n, torch.ones_like(n), torch.full_like(n, cos_i), 's')
        return (J * e_n * dlam * prop).sum().item()

    # Boundary-term J factor: at lambda=lambda*(A), v=0 by definition (that's
    # the critical condition n(lambda*)=kappa), so J_TIR(v=0) = 4*kappa is
    # the SAME constant for every A in the sweep -- lambda* moves, but it
    # always moves along the v=0 contour. Full Leibniz boundary term is
    # -J(v=0)*e_n(lambda*)*dlambda*/dA, not just -e_n(lambda*)*dlambda*/dA
    # (that simpler form is only correct for a bare emission integrand with
    # no J factor, cf. T12 which also multiplies by J_bdry explicitly).
    J_bdry = float(tir_jacobian(
        torch.zeros(1, dtype=torch.float64), torch.tensor([kappa], dtype=torch.float64),
        torch.ones(1, dtype=torch.float64), torch.tensor([cos_i], dtype=torch.float64), 's'))

    # Sweep lambda*(A) targets from well below lambda_min to well above
    # lambda_max (50nm margin on each side of the window), inverting
    # lambda*(A,B,cos_i) = sqrt(B/(kappa-A)) for A given a target lambda*.
    M = 121
    lam_star_target = torch.linspace(350.0, 750.0, M)
    A_sweep = kappa - B_val / lam_star_target ** 2

    A_vals, lam_stars, I_vals = [], [], []
    grad_fd, grad_kernel, grad_boundary, grad_total = [], [], [], []

    for i in range(M):
        A_val = A_sweep[i].item()

        lam_star_i = float(lam_star_fn(A_val, B_val, cos_i))
        dlam_dA_i  = float(dlambda_star_dA(lam_star_i, B_val))

        # Adaptive FD step: size h so the induced lambda* shift spans ~60
        # grid cells, regardless of how sensitive lambda*(A) is at this point
        # (dlambda*/dA ~ lambda*^3, spans orders of magnitude over the
        # sweep). The propagating mask is a hard per-grid-point cutoff, so
        # I(A) is a staircase at sub-grid-cell scale -- FD must average over
        # enough grid crossings to see the smooth macroscopic slope (same
        # discretization-bias mechanism as T12, just needing a wider window
        # here since dlambda*/dA is much larger).
        h = 60.0 * dlam / max(abs(dlam_dA_i), 1e-30)
        grad_fd_i = (compute_I(A_val + h) - compute_I(A_val - h)) / (2.0 * h)

        # Kernel term: autograd holding the propagating mask fixed at A_val.
        with torch.no_grad():
            n0        = n_cauchy(lam, torch.tensor(A_val, dtype=torch.float64), B_val)
            prop_mask = (1.0 - n0 ** 2 * sin2_i > 0.0).float()
        A_ag = torch.tensor(A_val, dtype=torch.float64, requires_grad=True)
        n_ag = n_cauchy(lam, A_ag, B_val)
        v_ag = torch.sqrt((1.0 - n_ag ** 2 * sin2_i).clamp(min=1e-30))
        J_ag = tir_jacobian(v_ag, n_ag, torch.ones(N, dtype=torch.float64),
                             torch.full((N,), cos_i, dtype=torch.float64), 's')
        (J_ag * e_n * dlam * prop_mask).sum().backward()
        grad_kernel_i = A_ag.grad.item()

        # Boundary term: -J_bdry * e_n(lambda*) * dlambda*/dA, active only
        # while lambda* is actually inside the measurement window.
        if lam_min <= lam_star_i <= lam_max:
            grad_boundary_i = -(J_bdry * e_n_at(lam_star_i)) * dlam_dA_i
        else:
            grad_boundary_i = 0.0

        A_vals.append(A_val)
        lam_stars.append(lam_star_i)
        I_vals.append(compute_I(A_val))
        grad_fd.append(grad_fd_i)
        grad_kernel.append(grad_kernel_i)
        grad_boundary.append(grad_boundary_i)
        grad_total.append(grad_kernel_i + grad_boundary_i)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURES_DIR / "G3_moving_boundary_sweep.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["A", "lambda_star", "I", "grad_fd", "grad_kernel", "grad_boundary", "grad_total"])
        for i in range(M):
            w.writerow([A_vals[i], lam_stars[i], I_vals[i], grad_fd[i],
                        grad_kernel[i], grad_boundary[i], grad_total[i]])

    lam_stars_t = torch.tensor(lam_stars)
    grad_fd_t    = torch.tensor(grad_fd)
    grad_total_t = torch.tensor(grad_total)
    grad_bdry_t  = torch.tensor(grad_boundary)

    # Check 1: lambda*(A_sweep) reproduces the target sweep exactly (sanity
    # on the closed-form inversion) and is monotonic increasing throughout.
    max_rel_target = ((lam_stars_t - lam_star_target).abs() / lam_star_target).max().item()
    mono_viol = (torch.diff(lam_stars_t) <= 0.0).sum().item()

    # Check 2: boundary term is negligible at both window edges (design
    # principle: e_n is ~7.5 sigma out) -- confirms the "no kink" regime,
    # in contrast to T12's deliberately non-negligible e_at_star=0.32.
    idx_near_min = min(range(M), key=lambda i: abs(lam_stars[i] - lam_min))
    idx_near_max = min(range(M), key=lambda i: abs(lam_stars[i] - lam_max))
    bdry_scale = grad_bdry_t.abs().max().item() + 1e-30
    bdry_at_min_rel = abs(grad_boundary[idx_near_min]) / bdry_scale
    bdry_at_max_rel = abs(grad_boundary[idx_near_max]) / bdry_scale

    # Check 3: kernel+boundary (= predicted total) tracks FD closely across
    # the whole sweep. Scale-normalized absolute error (not a raw per-point
    # relative error): grad_fd crosses zero partway through the sweep (the
    # curve isn't monotonic), and a plain relative metric blows up near that
    # crossing even though the absolute mismatch there is tiny relative to
    # the sweep's overall dynamic range.
    scale = grad_fd_t.abs().max().item()
    max_rel_err = ((grad_total_t - grad_fd_t).abs() / scale).max().item()

    # Check 4: no kink in the true (FD) gradient at either edge crossing --
    # the local second difference of grad_fd at the two crossing indices
    # must not be anomalously larger than the typical second difference
    # elsewhere in the sweep (interior points, away from either edge).
    d2 = grad_fd_t[2:] - 2.0 * grad_fd_t[1:-1] + grad_fd_t[:-2]
    interior = torch.cat([d2[:max(idx_near_min - 5, 1)], d2[min(idx_near_max + 5, M - 2):]])
    typical_d2 = interior.abs().median().item() + 1e-30
    kink_min = abs(d2[max(idx_near_min - 1, 0)].item()) / typical_d2
    kink_max = abs(d2[max(idx_near_max - 1, 0)].item()) / typical_d2

    tol = 1e-6
    return [
        StructResult("G3", "lambda*(A_sweep) matches inverted target",
                     max_rel_target, 0.0, max_rel_target, tol, max_rel_target < tol,
                     "closed-form A = kappa - B/lambda*^2 inverts lambda_star() exactly"),
        StructResult("G3", "lambda*(A) strictly monotonic increasing over full sweep",
                     float(mono_viol), 0.0, float(mono_viol), 0.5, mono_viol == 0,
                     f"{mono_viol} non-increasing steps out of {M-1}"),
        StructResult("G3", "boundary term negligible at lambda*~lambda_min crossing",
                     bdry_at_min_rel, 0.0, bdry_at_min_rel, 1e-2, bdry_at_min_rel < 1e-2,
                     f"e_n(lambda_min)~{e_n_at(lam_min):.2e} (design: ~7.5 sigma out), "
                     "cf. T12's deliberately non-negligible e_at_star=0.32"),
        StructResult("G3", "boundary term negligible at lambda*~lambda_max crossing",
                     bdry_at_max_rel, 0.0, bdry_at_max_rel, 1e-2, bdry_at_max_rel < 1e-2,
                     f"e_n(lambda_max)~{e_n_at(lam_max):.2e} (design: ~7.5 sigma out)"),
        StructResult("G3", "kernel+boundary vs FD, max |error|/scale over full sweep",
                     max_rel_err, 0.0, max_rel_err, 2e-2, max_rel_err < 2e-2,
                     f"scale = max|grad_fd| = {scale:.3g}; residual is grid discretization "
                     "bias near lambda*~lam_em, same mechanism as T12's 5e-2"),
        StructResult("G3", "no kink in true dI/dA at lambda*=lambda_min crossing",
                     kink_min, 1.0, abs(kink_min - 1.0), 5.0, kink_min < 5.0,
                     f"|2nd diff at crossing| / |typical interior 2nd diff| = {kink_min:.2f}"),
        StructResult("G3", "no kink in true dI/dA at lambda*=lambda_max crossing",
                     kink_max, 1.0, abs(kink_max - 1.0), 5.0, kink_max < 5.0,
                     f"|2nd diff at crossing| / |typical interior 2nd diff| = {kink_max:.2f}"),
    ]


# ---------------------------------------------------------------------------
# G4: Multi-fluorophore conditioning vs peak separation
# X-axis: emission-peak separation in units of sigma_e, 5*sigma_e -> 0
# Y-axis: condition number (log scale)
# Predicted shape: smooth monotonic blowup, genuine vertical asymptote at 0;
#                  identify functional form of divergence
# Failure: plateau instead of divergence -> "exact degeneracy" claim wrong
# ---------------------------------------------------------------------------

def test_G4() -> list[StructResult]:
    import csv
    import math

    # Same 2-fluorophore model as T11 (w1, w2, lam_em1, lam_em2 parameters,
    # w1=w2=1.0 held fixed, only the emission-peak separation sep is swept).
    # T11 showed sigma_min=0 EXACTLY at sep=0 (hard wall, two SVs collapse
    # simultaneously since w1=w2 makes the w-columns proportional too, once
    # lem1=lem2 makes the lam_em-columns proportional). G4 sweeps sep on a
    # log grid from 5*sigma_f down to near that wall and checks the
    # divergence is a genuine smooth power law all the way down, not a
    # plateau that would contradict T11's "hard wall, not just large cond"
    # framing.
    lam    = torch.linspace(400.0, 700.0, 80, dtype=torch.float64)
    w      = torch.full((80,), 300.0 / 79.0, dtype=torch.float64)
    sf     = 20.0
    lam_ex = 450.0
    alpha0 = 1.0
    center = 540.0                        # same reference point as T11

    def L_fl(w1, w2, lem1, lem2):
        a  = torch.exp(-0.5 * ((lam - lam_ex) / sf) ** 2)
        e1 = torch.exp(-0.5 * ((lam - lem1)   / sf) ** 2)
        e2 = torch.exp(-0.5 * ((lam - lem2)   / sf) ** 2)
        e1 = e1 / (e1 * w).sum().clamp(min=1e-30)
        e2 = e2 / (e2 * w).sum().clamp(min=1e-30)
        abar = (a * w).sum() * alpha0
        return w1 * e1 * abar + w2 * e2 * abar

    def jacobian(w1, w2, lem1, lem2):
        ts = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                   for v in [w1, w2, lem1, lem2])
        cols = torch.autograd.functional.jacobian(
            lambda *a: L_fl(*a), ts, vectorize=True,
        )
        return torch.stack([c.detach() for c in cols], dim=1)   # (80, 4)

    # sep: 5*sigma_f (well-separated) down to 0.01nm (near coincidence, but
    # kept far enough above the SVD noise floor -- cond ~2e11 there, still
    # well inside float64's reliable range, cf. T11's exact-zero hard wall
    # at sep=0 exactly).
    M = 60
    sep = torch.tensor([100.0 * (0.01 / 100.0) ** (i / (M - 1)) for i in range(M)],
                        dtype=torch.float64)

    conds, sigma_mins, sigma_maxs = [], [], []
    for i in range(M):
        s = sep[i].item()
        J = jacobian(1.0, 1.0, center + s / 2.0, center - s / 2.0)
        col_scales = J.norm(dim=0).clamp(min=1e-30)
        svs = torch.linalg.svdvals(J / col_scales)
        sigma_maxs.append(svs[0].item())
        sigma_mins.append(svs[-1].item())
        conds.append((svs[0] / svs[-1]).item())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURES_DIR / "G4_multi_fluorophore_conditioning.csv").open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["separation_nm", "sigma_max", "sigma_min", "condition_number"])
        for i in range(M):
            wtr.writerow([sep[i].item(), sigma_maxs[i], sigma_mins[i], conds[i]])

    log_sep  = [math.log(s) for s in sep.tolist()]
    log_cond = [math.log(c) for c in conds]

    # Check 1: monotonic blowup as sep -> 0, no plateau anywhere in the sweep.
    mono_viol = sum(1 for i in range(M - 1) if conds[i] >= conds[i + 1])

    # Check 2: well-conditioned at the well-separated end (sep = 5*sigma_f)
    # -- sanity anchor, mirrors T9/T11's "normal point" check.
    cond_wide = conds[0]

    # Check 3: power-law fit log(cond) = -p*log(sep) + c over the smaller
    # half of the sweep (deep in the asymptotic regime) -- "identify the
    # functional form of divergence" per the G4 spec. Least squares by hand
    # (no numpy per repo convention).
    xs, ys = log_sep[M // 2:], log_cond[M // 2:]
    n  = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot

    # Check 4: self-consistency of the power law -- fit separately on the
    # first and second quarters of the asymptotic half; slopes must agree
    # (a genuine power law is straight in log-log space at every scale; an
    # exponential-type divergence or a plateau would bend and disagree).
    def fit_slope(xs_: list, ys_: list) -> float:
        n_ = len(xs_)
        mx_ = sum(xs_) / n_
        my_ = sum(ys_) / n_
        num_ = sum((x - mx_) * (y - my_) for x, y in zip(xs_, ys_))
        den_ = sum((x - mx_) ** 2 for x in xs_)
        return num_ / den_

    q = M // 4
    slope_q1 = fit_slope(log_sep[2 * q:3 * q], log_cond[2 * q:3 * q])
    slope_q2 = fit_slope(log_sep[3 * q:4 * q], log_cond[3 * q:4 * q])
    slope_diff = abs(slope_q1 - slope_q2)

    # Check 5: no plateau -- cond at the smallest sep is orders of magnitude
    # above cond one decade-in-sep earlier (explicit anti-plateau guard,
    # matching the G4 docstring's stated failure mode).
    idx_decade = min(range(M), key=lambda i: abs(sep[i].item() - sep[-1].item() * 10.0))
    plateau_ratio = conds[-1] / max(conds[idx_decade], 1e-30)

    tol = 1e-2
    return [
        StructResult("G4", "condition number monotonic blowup, no plateau",
                     float(mono_viol), 0.0, float(mono_viol), 0.5, mono_viol == 0,
                     f"{mono_viol} non-increasing steps out of {M-1} as sep -> 0"),
        StructResult("G4", "well-conditioned at sep = 5*sigma_f",
                     cond_wide, 1.0, abs(cond_wide - 1.0), 0.1, cond_wide < 1.1,
                     f"cond={cond_wide:.4f} at widest separation, sanity anchor"),
        StructResult("G4", "power-law fit quality (log-log R^2), smaller half of sweep",
                     r2, 1.0, abs(r2 - 1.0), tol, abs(r2 - 1.0) < tol,
                     f"slope={slope:.4f} -- cond ~ separation^{slope:.2f}"),
        StructResult("G4", "power-law slope self-consistent across sub-ranges",
                     slope_diff, 0.0, slope_diff, 0.05, slope_diff < 0.05,
                     f"slope(q1)={slope_q1:.4f}, slope(q2)={slope_q2:.4f} -- "
                     "straight in log-log at every scale, not curving"),
        StructResult("G4", "no plateau: cond(sep_min) >> cond(10x sep_min)",
                     plateau_ratio, None, None, 100.0, plateau_ratio > 100.0,
                     f"ratio={plateau_ratio:.3g} (cond ~ sep^-3 predicts ~1000x per decade)"),
    ]


# ---------------------------------------------------------------------------
# G5: Conditioning heatmap (separation x species count)
# X-axis: emission-peak separation
# Y-axis (2nd): species count k
# Output: heatmap of condition number
# Predicted shape: horizontal bands -- conditioning depends on separation, not k
# Failure: diagonal/k-dependent banding
# ---------------------------------------------------------------------------

def test_G5() -> StructResult:
    raise NotImplementedError("G5")


# ---------------------------------------------------------------------------
# G6: Measurement diversity vs conditioning
# X-axis: number of diversity measurements added (1 -> many angle+pol combos)
# Y-axis: condition number
# Predicted shape: sharp initial drop, visible diminishing returns / flattening
# Failure: no flattening (keeps improving linearly) -- real systems saturate
# ---------------------------------------------------------------------------

def test_G6() -> StructResult:
    raise NotImplementedError("G6")


# ---------------------------------------------------------------------------
# G7: Index contrast -> 0
# X-axis: substrate/film index contrast, -> 0
# Y-axis: condition number
# Predicted shape: diverges as contrast -> 0
# Failure: stays bounded -> contradicts claimed mechanism
# ---------------------------------------------------------------------------

def test_G7() -> StructResult:
    raise NotImplementedError("G7")


# ---------------------------------------------------------------------------
# G8: Film thickness d across FSR periods  [promoted from T7]
# X-axis: film thickness d, across several FSR periods
# Y-axis: recovery conditioning / residual fit error
# Predicted shape: periodic structure synced to FSR
# Failure: smooth, no periodic structure
# ---------------------------------------------------------------------------

def test_G8() -> StructResult:
    raise NotImplementedError("G8")


# ---------------------------------------------------------------------------
# G9: Illuminant slope k  [promoted from T8]
# X-axis: illuminant slope k, 0 -> up
# Y-axis: d(absorbed power)/dlambda_ex
# Predicted shape: smooth, continuous, monotonic from exactly 0 at k=0
# Failure: discontinuity at k=0
# ---------------------------------------------------------------------------

def test_G9() -> StructResult:
    raise NotImplementedError("G9")


# ---------------------------------------------------------------------------
# G10: Spectral bandwidth vs singular values
# X-axis: measurement spectral bandwidth, narrow -> wide
# Y-axis: all 5 singular values (especially B/A's)
# Predicted shape: improves with bandwidth (B/A separate more due to different
#                  lambda-power dependence: const vs 1/lambda^2)
# Failure: flat curve -> B/A confound isn't a spectral-leverage problem
# ---------------------------------------------------------------------------

def test_G10() -> StructResult:
    raise NotImplementedError("G10")


# ---------------------------------------------------------------------------
# G11: Wrong vs correct adjoint gradient error vs Stokes shift  [promoted from T3]
# X-axis: Stokes shift magnitude (e, a peak separation), 0 -> up
# Y-axis: |wrong_gradient - correct_gradient|
# Predicted shape: exactly 0 at zero separation, smooth monotonic growth
# Failure: nonzero error at zero separation -> bug has another source
# ---------------------------------------------------------------------------

def test_G11() -> StructResult:
    raise NotImplementedError("G11")


# ---------------------------------------------------------------------------
# G12: lambda_ex invariance (V-level, rendered)
# X-axis: lambda_ex swept
# Y-axis: rendered pixel value, with MC-noise error bars
# Predicted shape: perfectly flat line, error bars overlapping at every point
# Failure: any visible slope within noise -- instant red flag for Theorem 7
# ---------------------------------------------------------------------------

def test_G12() -> StructResult:
    raise NotImplementedError("G12")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL = [
    test_G1, test_G2, test_G3, test_G4, test_G5, test_G6,
    test_G7, test_G8, test_G9, test_G10, test_G11, test_G12,
]


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rep = Reporter()
    for fn in ALL:
        try:
            result = fn()
            if isinstance(result, list):
                for r in result:
                    rep.add(r)
            else:
                rep.add(result)
        except NotImplementedError as e:
            print(f"SKIP  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e}", file=sys.stderr)
    rep.print_all()


if __name__ == "__main__":
    main()
