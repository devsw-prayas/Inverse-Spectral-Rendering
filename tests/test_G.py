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

def test_G5() -> list[StructResult]:
    import csv
    import math

    # Two experiments, per explicit user choice after G4 revealed the naive
    # "horizontal bands" hypothesis is design-dependent:
    #
    #   Panel A "isolated pair"  -- ONE tight pair at separation `sep`, plus
    #     k-2 extra background species parked 35+ sigma_f away (negligible
    #     overlap with the pair or each other). Tests whether UNRELATED
    #     species hurt conditioning of a tight pair. This is what the G5
    #     docstring's "horizontal bands" prediction actually describes.
    #
    #   Panel B "equally-spaced chain" -- k species all spaced `sep` apart
    #     in a single chain (same model as the first G5 attempt). Confirmed
    #     to have STRONG k-dependence (chain multicollinearity compounds
    #     even at fixed nearest-neighbor spacing) -- this contradicts the
    #     naive horizontal-bands hypothesis and is kept as a documented
    #     contrast finding, same spirit as G1's naive-vs-stable comparison.
    sf     = 20.0
    lam_ex = 450.0
    alpha0 = 1.0

    def make_model(lam: torch.Tensor, dlam_w: torch.Tensor, center: float):
        def L_fl_k(k: int, *args):
            ws, lems = args[:k], args[k:]
            a    = torch.exp(-0.5 * ((lam - lam_ex) / sf) ** 2)
            abar = (a * dlam_w).sum() * alpha0
            out  = torch.zeros_like(lam)
            for wi, lemi in zip(ws, lems):
                ei = torch.exp(-0.5 * ((lam - lemi) / sf) ** 2)
                ei = ei / (ei * dlam_w).sum().clamp(min=1e-30)
                out = out + wi * ei * abar
            return out

        def jacobian_k(k: int, lems: list) -> torch.Tensor:
            ws = [1.0] * k
            ts = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                       for v in ws + lems)
            cols = torch.autograd.functional.jacobian(
                lambda *a: L_fl_k(k, *a), ts, vectorize=True,
            )
            return torch.stack([c.detach() for c in cols], dim=1)

        def cond_of(k: int, lems: list) -> float:
            J = jacobian_k(k, lems)
            col_scales = J.norm(dim=0).clamp(min=1e-30)
            svs = torch.linalg.svdvals(J / col_scales)
            return (svs[0] / svs[-1]).item()

        return cond_of

    k_vals   = [2, 3, 4, 5, 6]
    sep_vals = [100.0, 40.0, 20.0, 10.0, 5.0]     # 5*sigma_f down to 0.25*sigma_f

    # Panel A: isolated pair + far background species. Wide synthetic domain
    # so background offsets (700-1300nm, i.e. 35-65 sigma_f) sit nowhere
    # near the pair or the window edges.
    lam_A = torch.linspace(0.0, 3000.0, 300, dtype=torch.float64)
    w_A   = torch.full((300,), 3000.0 / 299.0, dtype=torch.float64)
    center_A = 1500.0
    cond_A_fn = make_model(lam_A, w_A, center_A)

    def bg_positions(n: int) -> list:
        offsets = [700.0, -700.0, 1000.0, -1000.0, 1300.0, -1300.0]
        return [center_A + o for o in offsets[:n]]

    # Panel B: equally-spaced chain, same window/grid as G4/T11.
    lam_B = torch.linspace(400.0, 700.0, 80, dtype=torch.float64)
    w_B   = torch.full((80,), 300.0 / 79.0, dtype=torch.float64)
    center_B = 540.0
    cond_B_fn = make_model(lam_B, w_B, center_B)

    heat_A: dict = {}
    heat_B: dict = {}
    for sep in sep_vals:
        for k in k_vals:
            pair_lems_A = [center_A + sep / 2.0, center_A - sep / 2.0]
            heat_A[(k, sep)] = cond_A_fn(k, pair_lems_A + bg_positions(k - 2))

            chain_lems_B = [center_B + sep * (i - (k - 1) / 2.0) for i in range(k)]
            heat_B[(k, sep)] = cond_B_fn(k, chain_lems_B)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURES_DIR / "G5_conditioning_heatmap.csv").open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["model", "k", "separation_nm", "condition_number"])
        for sep in sep_vals:
            for k in k_vals:
                wtr.writerow(["isolated_pair", k, sep, heat_A[(k, sep)]])
        for sep in sep_vals:
            for k in k_vals:
                wtr.writerow(["chain", k, sep, heat_B[(k, sep)]])

    # Check 1: Panel A is horizontal bands -- cond number at fixed sep varies
    # negligibly across k (background species are numerically orthogonal).
    max_k_var_A = max(
        (max(heat_A[(k, sep)] for k in k_vals) - min(heat_A[(k, sep)] for k in k_vals))
        / min(heat_A[(k, sep)] for k in k_vals)
        for sep in sep_vals
    )

    # Check 2: Panel B is NOT horizontal -- cond strictly increases with k at
    # every fixed sep (documented contrast finding, not a bug).
    chain_k_mono = all(
        heat_B[(k, sep)] < heat_B[(k + 1, sep)]
        for sep in sep_vals for k in k_vals[:-1]
    )

    # Check 3: quantify how strong Panel B's k-dependence is, at the
    # tightest separation -- this is the number that would falsify a naive
    # "conditioning depends on separation only" reading of Table 1.
    chain_ratio_k6_k2 = heat_B[(6, 5.0)] / heat_B[(2, 5.0)]

    # Check 4: cross-validation -- Panel A and Panel B are literally the
    # same 2-species model at k=2 (isolated pair IS the chain at k=2, no
    # background/extra chain links yet), computed on different grids/
    # domains via independently-built jacobian_k closures. They must agree.
    cross_rel_errs = [
        abs(heat_A[(2, sep)] - heat_B[(2, sep)]) / heat_B[(2, sep)] for sep in sep_vals
    ]
    max_cross_rel_err = max(cross_rel_errs)

    tol = 1e-6
    return [
        StructResult("G5", "Panel A (isolated pair + background): horizontal bands, "
                            "max rel var across k at fixed sep",
                     max_k_var_A, 0.0, max_k_var_A, 1e-8, max_k_var_A < 1e-8,
                     "background species 35+ sigma_f away are numerically orthogonal -- "
                     "confirms the docstring's original horizontal-bands hypothesis"),
        StructResult("G5", "Panel B (equally-spaced chain): cond strictly increases with k "
                            "at every fixed sep",
                     float(chain_k_mono), 1.0, float(not chain_k_mono), 0.5, chain_k_mono,
                     "contradicts naive horizontal-bands hypothesis -- chain multicollinearity "
                     "compounds even at fixed nearest-neighbor spacing (real finding, not a bug)"),
        StructResult("G5", "Panel B k-dependence magnitude: cond(k=6)/cond(k=2) at sep=5nm",
                     chain_ratio_k6_k2, None, None, 100.0, chain_ratio_k6_k2 > 100.0,
                     f"ratio={chain_ratio_k6_k2:.3g} -- species count matters as much as "
                     "separation for a densely-packed multi-fluorophore mix"),
        StructResult("G5", "Panel A vs Panel B agree at k=2 (same model, independent code paths, "
                            "different grids/domains)",
                     max_cross_rel_err, 0.0, max_cross_rel_err, tol, max_cross_rel_err < tol,
                     "isolated-pair and chain models are identical at k=2 -- cross-validates "
                     "both jacobian_k closures"),
    ]


# ---------------------------------------------------------------------------
# G6: Measurement diversity vs conditioning
# X-axis: number of diversity measurements added (1 -> many angle+pol combos)
# Y-axis: condition number
# Predicted shape: sharp initial drop, visible diminishing returns / flattening
# Failure: no flattening (keeps improving linearly) -- real systems saturate
# ---------------------------------------------------------------------------

def test_G6() -> list[StructResult]:
    import csv
    from src.kernels import fabry_airy_R

    # Thin-film (d, A, B) recovery from unpolarized R(lambda) measurements at
    # progressively more incidence angles. A sparse 6-channel wavelength grid
    # (not the usual 80-point fine grid) so a single measurement alone is
    # genuinely under-determined and angle diversity has real work to do --
    # with 80 fine spectral samples one measurement already pins (d,A,B)
    # fairly well (checked: cond~7-8 even at n=1), which would hide the
    # "sharp drop then flatten" shape this test is meant to demonstrate.
    lam    = torch.linspace(450.0, 650.0, 6, dtype=torch.float64)
    d0, A0, B0 = 80.0, 1.5, 5000.0

    # Angles ordered from normal incidence (least diverse) to grazing (most
    # diverse) -- unpolarized only. (s vs p diversity was also explored and
    # does help further, but interacts non-monotonically with angle order
    # since s=p exactly at normal incidence and their relative informativeness
    # varies with angle -- unpolarized angle-only sweep gives the cleanest,
    # strictly monotonic signal for this test's shape claim.)
    angles = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60,
              0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
    M = len(angles)

    def build_jacobian(n_meas: int) -> torch.Tensor:
        ts = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                   for v in [d0, A0, B0])

        def fn(d, A, B):
            outs = [fabry_airy_R(lam, ci, d, A, B, "unpolarized")
                    for ci in angles[:n_meas]]
            return torch.cat(outs)

        cols = torch.autograd.functional.jacobian(fn, ts, vectorize=True)
        return torch.stack([c.detach() for c in cols], dim=1)

    conds = []
    for n in range(1, M + 1):
        J = build_jacobian(n)
        col_scales = J.norm(dim=0).clamp(min=1e-30)
        svs = torch.linalg.svdvals(J / col_scales)
        conds.append((svs[0] / svs[-1]).item())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURES_DIR / "G6_measurement_diversity.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_measurements", "cos_i_added", "condition_number"])
        for i in range(M):
            w.writerow([i + 1, angles[i], conds[i]])

    rel_drops = [(conds[i] - conds[i + 1]) / conds[i] for i in range(M - 1)]

    # Check 1: strictly monotonic -- more diversity never hurts conditioning.
    mono_viol = sum(1 for d in rel_drops if d < 0.0)

    # Check 2: sharp initial drop -- the very first added measurement alone
    # buys a large fraction improvement.
    first_drop = rel_drops[0]

    # Check 3: diminishing returns -- average |relative drop| over the first
    # 4 additions is much larger than over the last 4 (visible flattening).
    first4_avg = sum(rel_drops[:4]) / 4.0
    last4_avg  = sum(rel_drops[-4:]) / 4.0
    dr_ratio   = first4_avg / max(last4_avg, 1e-30)

    # Check 4: substantial net improvement over the whole sweep.
    net_ratio = conds[0] / conds[-1]

    # Check 5: near-flat tail -- adding the last few (most redundant, most
    # grazing) angles changes conditioning by only a small fraction, unlike
    # the first addition (Check 2). This is the literal "flattening" the G6
    # docstring predicts, and its failure mode ("no flattening, keeps
    # improving linearly") would show up here as a large value.
    tail_change = abs(conds[-1] - conds[-4]) / conds[-4]

    tol = 1e-9
    return [
        StructResult("G6", "condition number strictly monotonic non-increasing with n_measurements",
                     float(mono_viol), 0.0, float(mono_viol), 0.5, mono_viol == 0,
                     f"{mono_viol} increasing steps out of {M-1} as measurements are added"),
        StructResult("G6", "sharp initial drop: relative improvement from 1st added measurement",
                     first_drop, None, None, 0.1, first_drop > 0.1,
                     f"cond drops {100*first_drop:.1f}% after just the 2nd angle is added"),
        StructResult("G6", "diminishing returns: avg |rel drop| first-4 / last-4 additions",
                     dr_ratio, None, None, 2.0, dr_ratio > 2.0,
                     f"first4_avg={first4_avg:.4f}, last4_avg={last4_avg:.4f}"),
        StructResult("G6", "substantial net conditioning improvement, cond(1)/cond(M)",
                     net_ratio, None, None, 4.0, net_ratio > 4.0,
                     f"cond(1)={conds[0]:.3g} -> cond({M})={conds[-1]:.3g}"),
        StructResult("G6", "near-flat tail: |cond(M)-cond(M-3)|/cond(M-3), most-grazing angles",
                     tail_change, 0.0, tail_change, 0.15, tail_change < 0.15,
                     "visible flattening near the end of the sweep, not linear improvement"),
    ]


# ---------------------------------------------------------------------------
# G7: Index contrast -> 0
# X-axis: substrate/film index contrast, -> 0
# Y-axis: condition number
# Predicted shape: diverges as contrast -> 0
# Failure: stays bounded -> contradicts claimed mechanism
# ---------------------------------------------------------------------------

def test_G7() -> list[StructResult]:
    import csv
    import math
    from src.cauchy_ior import n_cauchy, cos_theta_t
    from src.fresnel import fresnel_rs

    # Extends T6's 3-layer air/film/substrate stack (reused R3_s) with a full
    # (d,A,B) Jacobian, not just the single ||dR/dd|| column T6 checked.
    #
    # Real finding (flagged to and confirmed by user before writing this
    # test): the G7 docstring's naive "condition number diverges as contrast
    # -> 0" does NOT hold. ||dR/dd|| (T6's raw sensitivity) does shrink
    # smoothly and linearly with contrast -- but after column-normalization,
    # the condition number of the (d,A,B) Jacobian PLATEAUS at a finite value
    # for every nonzero contrast (even down to 1e-12) and only becomes
    # singular in the literal contrast=0 limit (hard wall, exact sigma_min=0
    # -- r23 identically 0 for ALL d, an exact structural degeneracy). This
    # is the same landmine family as T11/T13/T15: a discontinuous jump at a
    # single point, not a smooth blowup. Both quantities are reported so the
    # distinction between "raw sensitivity" and "relative conditioning" is
    # visible in the same figure.
    lam   = torch.linspace(400.0, 700.0, 60, dtype=torch.float64)
    cos_i = torch.ones(60, dtype=torch.float64)

    A_film, B_film, d_film = 1.5, 5000.0, 120.0
    D_sub = 5000.0   # == B_film -- contrast=0 means substrate == film exactly

    def R3_s(d, A, B, C_sub):
        n_f   = n_cauchy(lam, A, B)
        n_s   = n_cauchy(lam, C_sub, D_sub)
        cos_f = cos_theta_t(cos_i, 1.0, n_f)
        cos_s = cos_theta_t(cos_f, n_f, n_s)
        r12   = fresnel_rs(1.0, n_f, cos_i, cos_f)
        r23   = fresnel_rs(n_f, n_s, cos_f, cos_s)
        phi   = 4.0 * torch.pi * n_f * cos_f * d / lam
        eiphi = torch.polar(torch.ones_like(phi), phi)
        return (r12 + r23 * eiphi).abs() ** 2 / (1.0 + r12 * r23 * eiphi).abs() ** 2

    def build_jacobian(C_sub: float) -> torch.Tensor:
        ts = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                   for v in [d_film, A_film, B_film])
        cols = torch.autograd.functional.jacobian(
            lambda d, A, B: R3_s(d, A, B, C_sub), ts, vectorize=True,
        )
        return torch.stack([c.detach() for c in cols], dim=1)   # (60, 3)

    # Log-spaced contrast sweep, 0.3 down to 1e-10 -- deep enough to confirm
    # the plateau holds many orders of magnitude below where a real power-law
    # divergence would already be enormous (cf. G4's cond ~ sep^-3).
    M = 20
    contrasts = [0.3 * (1e-10 / 0.3) ** (i / (M - 1)) for i in range(M)]

    d_norms, conds = [], []
    for c in contrasts:
        J = build_jacobian(A_film - c)
        d_norms.append(J[:, 0].norm().item())
        col_scales = J.norm(dim=0).clamp(min=1e-30)
        svs = torch.linalg.svdvals(J / col_scales)
        conds.append((svs[0] / svs[-1]).item())

    # Exact contrast=0: literal hard wall.
    J0 = build_jacobian(A_film)
    d_norm_0 = J0[:, 0].norm().item()
    svs0 = torch.linalg.svdvals(J0 / J0.norm(dim=0).clamp(min=1e-30))
    sigma_min_0 = svs0[-1].item()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURES_DIR / "G7_index_contrast.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "dR_dd_norm", "condition_number"])
        for i in range(M):
            w.writerow([contrasts[i], d_norms[i], conds[i]])
        w.writerow([0.0, d_norm_0, "inf"])

    # Check 1: ||dR/dd|| shrinks smoothly, ~linearly with contrast (T6's
    # quantity, extended over many more decades) -- power-law fit slope ~1.
    log_c = [math.log(c) for c in contrasts]
    log_d = [math.log(n) for n in d_norms]
    n_    = len(log_c)
    mx, my = sum(log_c) / n_, sum(log_d) / n_
    num = sum((x - mx) * (y - my) for x, y in zip(log_c, log_d))
    den = sum((x - mx) ** 2 for x in log_c)
    slope_d = num / den
    ss_res = sum((y - (slope_d * x + my - slope_d * mx)) ** 2 for x, y in zip(log_c, log_d))
    ss_tot = sum((y - my) ** 2 for y in log_d)
    r2_d = 1.0 - ss_res / ss_tot

    # Check 2: condition number PLATEAUS deep in the small-contrast regime --
    # NOT across the whole sweep, which does show a real transition between
    # "moderate contrast" and "deep asymptotic" behavior (cond rises from
    # 10.7 at contrast=0.3 to ~31 by contrast~1e-3, matching T6's direction).
    # The claim under test is specifically what happens once you're already
    # deep in the small-contrast regime: does it keep climbing (divergence)
    # or flatten out (plateau)? Use the smaller half of the log-spaced sweep
    # (contrast <~ 5.5e-4) for this check.
    tail = conds[M // 2:]
    cond_min, cond_max = min(tail), max(tail)
    cond_plateau_rel_var = (cond_max - cond_min) / cond_min

    # Check 3: exact hard wall at contrast=0 -- sigma_min = 0 exactly (r23
    # identically 0 for all d -- structural rank deficiency, not asymptotic).
    # Meanwhile ||dR/dd|| at exact contrast=0 is also exactly 0 (consistent
    # with the smooth trend in Check 1 extrapolated to its endpoint).
    tol_wall = 1e-12

    # Check 4: quantitative separation between the two notions -- ||dR/dd||
    # changes by many orders of magnitude across the sweep while cond changes
    # by only a small fraction, over the SAME contrast range.
    d_norm_ratio = d_norms[0] / d_norms[-1]

    tol = 1e-2
    return [
        StructResult("G7", "||dR/dd|| ~ contrast^1 (smooth, log-log R^2)",
                     r2_d, 1.0, abs(r2_d - 1.0), tol, abs(r2_d - 1.0) < tol,
                     f"slope={slope_d:.4f} -- raw sensitivity shrinks linearly with contrast, T6 extended"),
        StructResult("G7", "condition number PLATEAUS (does not diverge) deep in small-contrast regime",
                     cond_plateau_rel_var, 0.0, cond_plateau_rel_var, 0.05,
                     cond_plateau_rel_var < 0.05,
                     f"cond range [{cond_min:.3f}, {cond_max:.3f}] over contrast in "
                     f"[{contrasts[-1]:.1e}, {contrasts[M // 2]:.1e}] -- contradicts naive "
                     "divergence hypothesis, confirmed with user before asserting this"),
        StructResult("G7", "exact hard wall: sigma_min = 0 at contrast = 0 exactly",
                     sigma_min_0, 0.0, sigma_min_0, tol_wall, sigma_min_0 < tol_wall,
                     "r23 identically 0 for ALL d at exact contrast=0 -- structural, not asymptotic"),
        StructResult("G7", "raw sensitivity spans many decades while conditioning barely moves",
                     d_norm_ratio, None, None, 1e6, d_norm_ratio > 1e6,
                     f"||dR/dd|| ratio={d_norm_ratio:.3g} across the sweep vs cond varying "
                     f"only {100*cond_plateau_rel_var:.2f}% -- 'unobservable in absolute "
                     "sensitivity' != 'ill-conditioned in relative terms'"),
    ]


# ---------------------------------------------------------------------------
# G8: Film thickness d across FSR periods  [promoted from T7]
# X-axis: film thickness d, across several FSR periods
# Y-axis: recovery conditioning / residual fit error
# Predicted shape: periodic structure synced to FSR
# Failure: smooth, no periodic structure
# ---------------------------------------------------------------------------

def test_G8() -> list[StructResult]:
    import csv
    import math
    from src.kernels import fabry_airy_R

    # T7/G8's own doc entry explicitly anticipates BOTH outcomes as valid
    # ("Smooth, no periodic structure -> near-rational-FSR risk was never
    # real (useful negative result either way)") -- unlike G5/G7, no need to
    # check in with the user before writing assertions either way; just
    # measure it honestly.
    #
    # Free-standing thin film, (d,A,B) recovery Jacobian, deliberately sparse
    # 6-channel wavelength grid (same aliasing-prone setup as G6) so any
    # FSR/sample-spacing resonance has the best chance of showing up.
    # d is swept over several FSR periods (defined at the window's center
    # wavelength) and the resulting condition-number curve is tested for a
    # dominant single-frequency component locked to the FSR, via a
    # least-squares single-sinusoid fit at exactly f=1/d_fringe (more
    # robust than an FFT peak search, which suffers window-leakage smearing
    # for a signal that isn't perfectly periodic across the sweep).
    lam    = torch.linspace(450.0, 650.0, 6, dtype=torch.float64)
    cos_i  = torch.tensor(1.0, dtype=torch.float64)
    A0, B0 = 1.5, 5000.0
    lam_c  = 550.0
    n_ref  = A0 + B0 / lam_c ** 2
    d_fringe = lam_c / (2.0 * n_ref)      # d-spacing for one full 2*pi phase cycle

    def build_jacobian(d: float) -> torch.Tensor:
        ts = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                   for v in [d, A0, B0])
        cols = torch.autograd.functional.jacobian(
            lambda dd, A, B: fabry_airy_R(lam, cos_i, dd, A, B, "unpolarized"),
            ts, vectorize=True,
        )
        return torch.stack([c.detach() for c in cols], dim=1)   # (6, 3)

    N = 2048
    n_periods = 16
    span = n_periods * d_fringe
    d0 = 300.0
    d_sweep = torch.linspace(d0, d0 + span, N, dtype=torch.float64)

    conds = torch.zeros(N, dtype=torch.float64)
    for i in range(N):
        J = build_jacobian(d_sweep[i].item())
        col_scales = J.norm(dim=0).clamp(min=1e-30)
        svs = torch.linalg.svdvals(J / col_scales)
        conds[i] = svs[0] / svs[-1]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURES_DIR / "G8_fsr_periodicity.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d_nm", "condition_number"])
        for i in range(N):
            w.writerow([d_sweep[i].item(), conds[i].item()])

    # Linear detrend -- separates a slow drift/envelope (which showed up as
    # the spuriously "dominant" FFT bin in exploration, at the scale of the
    # whole sweep span) from any genuine fringe-period oscillation.
    x = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
    X = torch.stack([torch.ones(N, dtype=torch.float64), x], dim=1)
    beta  = torch.linalg.lstsq(X, conds.unsqueeze(1)).solution.squeeze()
    detr  = conds - X @ beta
    total_var = (detr ** 2).sum().item()

    def r2_at_period(period: float) -> float:
        theta = 2.0 * math.pi * d_sweep / period
        Xs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        b  = torch.linalg.lstsq(Xs, detr.unsqueeze(1)).solution.squeeze()
        ss_res = ((detr - Xs @ b) ** 2).sum().item()
        return 1.0 - ss_res / total_var

    r2_fundamental = r2_at_period(d_fringe)
    r2_2nd_harm    = r2_at_period(2.0 * d_fringe)
    r2_half        = r2_at_period(0.5 * d_fringe)

    # Sanity: the conditioning genuinely varies a lot over the sweep (rules
    # out a degenerate "nothing happens at all" reading of the negative
    # result below).
    cv = (conds.std() / conds.mean()).item()

    tol_periodic = 0.2   # below this, "dominant single-tone periodicity" is not supported
    return [
        StructResult("G8", "coefficient of variation of cond(d) over the sweep (sanity: real variation exists)",
                     cv, None, None, 0.2, cv > 0.2,
                     f"mean={conds.mean().item():.3g}, std={conds.std().item():.3g} -- "
                     "conditioning does fluctuate substantially across the sweep"),
        StructResult("G8", "R^2 of single sinusoid at FSR fundamental (d_fringe) vs detrended cond(d)",
                     r2_fundamental, 0.0, r2_fundamental, tol_periodic,
                     r2_fundamental < tol_periodic,
                     f"d_fringe={d_fringe:.2f}nm explains only {100*r2_fundamental:.1f}% of "
                     "detrended variance -- NOT a dominant FSR-locked periodicity"),
        StructResult("G8", "R^2 of single sinusoid at 2nd harmonic (2*d_fringe) vs detrended cond(d)",
                     r2_2nd_harm, 0.0, r2_2nd_harm, tol_periodic, r2_2nd_harm < tol_periodic,
                     f"{100*r2_2nd_harm:.1f}% of detrended variance"),
        StructResult("G8", "R^2 of single sinusoid at half-fringe (0.5*d_fringe) vs detrended cond(d)",
                     r2_half, 0.0, r2_half, tol_periodic, r2_half < tol_periodic,
                     f"{100*r2_half:.1f}% of detrended variance -- rules out a x2-aliased lock too"),
        StructResult("G8", "conclusion: smooth/broadband, no periodic structure synced to FSR",
                     max(r2_fundamental, r2_2nd_harm, r2_half), 0.0,
                     max(r2_fundamental, r2_2nd_harm, r2_half), tol_periodic,
                     max(r2_fundamental, r2_2nd_harm, r2_half) < tol_periodic,
                     "matches the doc's own anticipated negative-result branch: "
                     "near-rational-FSR aliasing risk was never a real Jacobian-conditioning "
                     "concern at this (Python-oracle) level -- doesn't settle V8's separate, "
                     "render-level sampling-aliasing question"),
    ]


# ---------------------------------------------------------------------------
# G9: Illuminant slope k  [promoted from T8]
# X-axis: illuminant slope k, 0 -> up
# Y-axis: d(absorbed power)/dlambda_ex
# Predicted shape: smooth, continuous, monotonic from exactly 0 at k=0
# Failure: discontinuity at k=0
# ---------------------------------------------------------------------------

def test_G9() -> list[StructResult]:
    import csv
    import math

    # sec9's Lemma B contrast case (already verified analytically by sympy,
    # "verified by contrast"): under a linear-ramp illuminant L_i(lam')=L0+k*lam',
    # absorbed power = sqrt(2*pi)*alpha0*sigma_f*(L0 + k*lam_ex), so
    # d(absorbed power)/d(lam_ex) = sqrt(2*pi)*alpha0*sigma_f*k exactly -- a
    # perfectly straight line through the origin in k. G9 promotes this from
    # a single point (T8) to a continuous sweep through k=0 (including
    # negative k, not just "0->up") specifically to test for the docstring's
    # named failure mode: a discontinuity/kink right at k=0 that would mean
    # Lemma B's flat-illuminant boundary is a measure-zero technicality
    # rather than a real regime change.
    #
    # lam_ex is centered in-window, sigma_f small enough that truncation is
    # negligible (T14 lesson: this integral is only exact over ALL lambda';
    # the renderer integrates over [lam_min,lam_max] only -- centering
    # lam_ex with many sigma of margin avoids that separate, already-
    # documented truncation effect from contaminating this test).
    lam_min, lam_max = 400.0, 700.0
    N = 300
    lam  = torch.linspace(lam_min, lam_max, N, dtype=torch.float64)
    dlam = (lam_max - lam_min) / (N - 1)
    w    = torch.full((N,), dlam, dtype=torch.float64)

    alpha0, sigma_f, L0 = 1.0, 20.0, 1.0
    lam_ex_val = 550.0   # window center, 7.5 sigma_f from either edge

    def absorbed_power(lam_ex_t: torch.Tensor, k: float) -> torch.Tensor:
        L_i = L0 + k * lam
        a   = alpha0 * torch.exp(-0.5 * ((lam - lam_ex_t) / sigma_f) ** 2)
        return (L_i * a * w).sum()

    M = 21
    k_vals = [-0.02 + 0.07 * i / (M - 1) for i in range(M)]   # -0.02 .. 0.05, includes 0 exactly
    k_vals[(M - 1) // 2] = 0.0

    autograds, analytics, fds = [], [], []
    for k in k_vals:
        lam_ex_t = torch.tensor(lam_ex_val, dtype=torch.float64, requires_grad=True)
        absorbed_power(lam_ex_t, k).backward()
        autograds.append(lam_ex_t.grad.item())

        analytics.append(alpha0 * math.sqrt(2.0 * math.pi) * sigma_f * k)

        h = 1e-3
        with torch.no_grad():
            fp = absorbed_power(torch.tensor(lam_ex_val + h, dtype=torch.float64), k)
            fm = absorbed_power(torch.tensor(lam_ex_val - h, dtype=torch.float64), k)
        fds.append(((fp - fm) / (2.0 * h)).item())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURES_DIR / "G9_illuminant_slope.csv").open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["k", "autograd", "analytic", "fd"])
        for i in range(M):
            wtr.writerow([k_vals[i], autograds[i], analytics[i], fds[i]])

    zero_idx = (M - 1) // 2
    scale = max(abs(a) for a in analytics)

    # Check 1: exactly 0 at k=0, all three ways.
    ag0, an0, fd0 = autograds[zero_idx], analytics[zero_idx], fds[zero_idx]

    # Check 2: autograd matches the closed-form analytic slope across the
    # whole sweep (scale-normalized, since values pass through exactly 0).
    max_ag_an_err = max(abs(a - b) for a, b in zip(autograds, analytics)) / scale

    # Check 3: three-way oracle -- autograd vs central FD (per CLAUDE.md
    # convention), same scale normalization.
    max_ag_fd_err = max(abs(a - b) for a, b in zip(autograds, fds)) / scale

    # Check 4: no kink at k=0 -- independently fit a line through the
    # negative-k half and the positive-k half (excluding k=0 itself) and
    # confirm the slopes agree (single straight line, not two different
    # ones meeting at a corner).
    def fit_slope(ks: list, ys: list) -> float:
        n = len(ks)
        mx, my = sum(ks) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(ks, ys))
        den = sum((x - mx) ** 2 for x in ks)
        return num / den

    neg_k = k_vals[:zero_idx]
    neg_y = autograds[:zero_idx]
    pos_k = k_vals[zero_idx + 1:]
    pos_y = autograds[zero_idx + 1:]
    slope_neg = fit_slope(neg_k, neg_y)
    slope_pos = fit_slope(pos_k, pos_y)
    slope_expected = alpha0 * math.sqrt(2.0 * math.pi) * sigma_f
    kink = abs(slope_neg - slope_pos) / slope_expected

    tol = 1e-6
    return [
        StructResult("G9", "d(absorbed power)/d(lam_ex) = 0 exactly at k=0 -- autograd",
                     ag0, 0.0, abs(ag0), 1e-9, abs(ag0) < 1e-9, "flat illuminant, Lemma B"),
        StructResult("G9", "d(absorbed power)/d(lam_ex) = 0 exactly at k=0 -- FD",
                     fd0, 0.0, abs(fd0), 1e-6, abs(fd0) < 1e-6, "flat illuminant, Lemma B"),
        StructResult("G9", "autograd matches closed-form analytic slope across full k sweep",
                     max_ag_an_err, 0.0, max_ag_an_err, tol, max_ag_an_err < tol,
                     f"scale-normalized max|autograd-analytic|, slope={slope_expected:.4f} per unit k"),
        StructResult("G9", "autograd matches central FD across full k sweep (3-way oracle)",
                     max_ag_fd_err, 0.0, max_ag_fd_err, 1e-4, max_ag_fd_err < 1e-4,
                     "scale-normalized max|autograd-FD|"),
        StructResult("G9", "no kink at k=0: slope(k<0) matches slope(k>0)",
                     kink, 0.0, kink, tol, kink < tol,
                     f"slope_neg={slope_neg:.4f}, slope_pos={slope_pos:.4f}, "
                     "expected both = sqrt(2*pi)*alpha0*sigma_f -- single straight line through k=0"),
    ]


# ---------------------------------------------------------------------------
# G10: Spectral bandwidth vs singular values
# X-axis: measurement spectral bandwidth, narrow -> wide
# Y-axis: all 5 singular values (especially B/A's)
# Predicted shape: improves with bandwidth (B/A separate more due to different
#                  lambda-power dependence: const vs 1/lambda^2)
# Failure: flat curve -> B/A confound isn't a spectral-leverage problem
# ---------------------------------------------------------------------------

def test_G10() -> list[StructResult]:
    import csv

    # Reuses T9's exact 5-param model (fabry_airy_R + normalized fluorescence
    # term, lam_ex held fixed since Sec9 already removes it from the
    # parameter set): L_out(d,A,B,sf,lem) = R(lam;d,A,B) + e(lam;lem,sf)*abar.
    # Sec10/Remark4's finding (independently reproduced by T9): the worst-
    # conditioned direction is dominated by B (-0.78) paired with A (+0.58),
    # because n(lam)=A+B/lam^2 -- dphi/dA ~ 1/lam, dphi/dB ~ 1/lam^3 -- and
    # over a narrow band, 1/lam and 1/lam^3 both look locally linear in lam
    # (Taylor expansion), making the A and B columns nearly collinear. G10
    # tests the predicted fix: widening the observed band should expose the
    # curvature difference between the two power laws and decorrelate them.
    #
    # lam_ex=520, lem=580 (Stokes shift 60nm = 6*sigma_f) are centered around
    # band-center=550 with enough margin (sf=10nm) that even the narrowest
    # bandwidth tested leaves both peaks resolved with >=3 sigma of margin to
    # the nearest edge (T14 lesson: truncation of the fluorescence Gaussian
    # against the window edge is a separate, already-documented effect that
    # would contaminate the sf/lem columns if not kept small here).
    from src.kernels import fabry_airy_R

    center = 550.0
    lam_ex = 520.0
    d0, A0, B0, sf0, lem0 = 120.0, 1.5, 5000.0, 10.0, 580.0
    alpha0 = 1.0
    cos_i = torch.tensor(1.0, dtype=torch.float64)

    def L_out_fn(lam, w, d, A, B, sf, lem):
        R = fabry_airy_R(lam, cos_i, d, A, B)
        a = torch.exp(-0.5 * ((lam - lam_ex) / sf) ** 2)
        e = torch.exp(-0.5 * ((lam - lem) / sf) ** 2)
        e = e / (e * w).sum().clamp(min=1e-30)
        return R + e * (a * w).sum() * alpha0

    def jacobian(lam, w):
        ts = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                   for v in [d0, A0, B0, sf0, lem0])
        cols = torch.autograd.functional.jacobian(
            lambda *args: L_out_fn(lam, w, *args), ts, vectorize=True,
        )
        return torch.stack([c.detach() for c in cols], dim=1)   # (N, 5)

    dlam = 1.0   # fixed sampling density -- N grows with bandwidth, not resolution
    M = 14
    half_widths = [60.0 * (300.0 / 60.0) ** (i / (M - 1)) for i in range(M)]

    sv_all, conds, cos_ab = [], [], []
    Jn_narrowest = None
    for hw in half_widths:
        lam_min, lam_max = center - hw, center + hw
        N = int(round(2.0 * hw / dlam)) + 1
        lam = torch.linspace(lam_min, lam_max, N, dtype=torch.float64)
        w = torch.full((N,), (lam_max - lam_min) / (N - 1), dtype=torch.float64)

        J = jacobian(lam, w)
        col_scales = J.norm(dim=0).clamp(min=1e-30)
        Jn = J / col_scales
        svs = torch.linalg.svdvals(Jn)
        sv_all.append(svs.tolist())
        conds.append((svs[0] / svs[-1]).item())
        if Jn_narrowest is None:
            Jn_narrowest = Jn

        colA, colB = J[:, 1], J[:, 2]
        cos_ab.append((colA @ colB / (colA.norm() * colB.norm())).item())

    sigma_min = [svs[-1] for svs in sv_all]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURES_DIR / "G10_spectral_bandwidth.csv").open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["half_width", "sv1", "sv2", "sv3", "sv4", "sv5",
                      "condition_number", "cos_angle_A_B"])
        for i in range(M):
            wtr.writerow([half_widths[i], *sv_all[i], conds[i], cos_ab[i]])

    # Check 1: sigma_min improves (non-decreasing) as bandwidth widens --
    # allow a small fraction of steps to violate strict monotonicity from
    # numerical noise, same style as G6's mono_viol check.
    mono_viol_min = sum(1 for i in range(M - 1) if sigma_min[i + 1] < sigma_min[i] - 1e-9)

    # Check 2: |cos(A,B)| decreases (decorrelates) as bandwidth widens.
    abs_cos = [abs(c) for c in cos_ab]
    mono_viol_cos = sum(1 for i in range(M - 1) if abs_cos[i + 1] > abs_cos[i] + 1e-9)

    # Check 3: substantial net improvement, not just noise-level drift.
    sigma_min_ratio = sigma_min[-1] / max(sigma_min[0], 1e-30)

    # Check 4: substantial net decorrelation of the A,B columns.
    cos_drop = abs_cos[0] - abs_cos[-1]

    # Check 5: sanity check against Sec10/Remark4/T9's own finding -- at the
    # narrowest bandwidth tested, the worst singular vector's two largest-
    # magnitude components must be the A,B columns (indices 1,2), confirming
    # this model reproduces the known bottleneck before testing how bandwidth
    # affects it.
    _, _, Vh = torch.linalg.svd(Jn_narrowest)
    worst_vec = Vh[-1].abs()
    top2 = torch.topk(worst_vec, 2).indices.tolist()
    bottleneck_ok = set(top2) == {1, 2}

    tol = 0.5
    return [
        StructResult("G10", "sigma_min monotonic non-decreasing with bandwidth",
                     float(mono_viol_min), 0.0, float(mono_viol_min), tol,
                     mono_viol_min == 0,
                     f"{mono_viol_min} regressions out of {M - 1} steps as bandwidth widens"),
        StructResult("G10", "|cos(colA,colB)| monotonic non-increasing with bandwidth",
                     float(mono_viol_cos), 0.0, float(mono_viol_cos), tol,
                     mono_viol_cos == 0,
                     f"{mono_viol_cos} regressions out of {M - 1} steps; "
                     f"cos: {abs_cos[0]:.4f} -> {abs_cos[-1]:.4f}"),
        StructResult("G10", "substantial sigma_min improvement, narrow->wide",
                     sigma_min_ratio, None, None, 2.0, sigma_min_ratio > 2.0,
                     f"sigma_min({half_widths[0]:.0f}nm)={sigma_min[0]:.4g} -> "
                     f"sigma_min({half_widths[-1]:.0f}nm)={sigma_min[-1]:.4g}"),
        StructResult("G10", "substantial A,B column decorrelation, narrow->wide",
                     cos_drop, None, None, 0.05, cos_drop > 0.05,
                     f"|cos(A,B)| drops by {cos_drop:.4f} over the sweep"),
        StructResult("G10", "worst singular vector dominated by (A,B) at narrowest bandwidth",
                     float(bottleneck_ok), 1.0, float(not bottleneck_ok), 0.5, bottleneck_ok,
                     f"top-2 |components| indices={top2} (0=d,1=A,2=B,3=sf,4=lem) -- "
                     "matches Sec10/Remark4/T9's B-paired-with-A finding"),
    ]


# ---------------------------------------------------------------------------
# G11: Wrong vs correct adjoint gradient error vs Stokes shift  [promoted from T3]
# X-axis: Stokes shift magnitude (e, a peak separation), 0 -> up
#
# Primary assertion:   ||K_x - K_x^T||_2  (== ||T - T^T||_2, M_R symmetric drops
#                       out) -- exactly 0 at shift=0, monotonically increasing
#                       thereafter. This is the operator-level asymmetry the
#                       non-self-adjointness argument actually guarantees.
# Secondary assertion: |wrong_gradient - correct_gradient| for one FIXED
#                       (L_e, S) pair -- exactly 0 at shift=0, nonzero for
#                       every shift>0, but NO shape claim beyond that. This is
#                       a signed bilinear projection of the operator asymmetry
#                       onto fixed illumination/sensor directions and is not
#                       required to track the operator norm monotonically
#                       (verified: it dips, crosses zero, then grows -- a real
#                       finding, not a bug, kept here only as a weaker
#                       existence check).
# Failure: nonzero primary norm at shift=0, or the primary norm failing to
#          grow monotonically -- either would mean the self-adjointness
#          argument itself is wrong, not just this test's original overreach.
# ---------------------------------------------------------------------------

def test_G11() -> list[StructResult]:
    import csv
    from src.kernels import kernel_thinfilm, kernel_fluorescence, fabry_airy_dR_dd
    from src.gradient import kernel_gradient, kernel_gradient_wrong_adjoint, neumann_forward

    # Sec4's non-self-adjointness bug: the correct adjoint solves (I-T*)G=S
    # (S = dloss/dL, theta-independent); the wrong form sources G by
    # (dT/dtheta)L instead. These coincide only when T*=T, which for the
    # fluorescence rank-1 kernel K_fl(lam,lam')=e(lam)a(lam') requires a=e
    # (zero Stokes shift).
    #
    # First draft of this test asserted a single monotonic |wrong-correct|
    # curve (running_notes' T3 point-check, -69.5542 vs -36.7595, promoted to
    # a sweep). That curve turned out non-monotonic: verified (with random
    # L_e/g and several quantum_yield values, not just this test's specific
    # scene) that the SIGNED error dips negative, crosses back through zero
    # around shift~100nm, then grows -- a real effect, not noise, because
    # |wrong-correct| for one fixed (L_e,S) pair is a bilinear projection of
    # the growing operator asymmetry onto fixed directions, and a projection
    # of a monotonically growing quantity need not itself be monotonic.
    #
    # Split into two assertions instead (flagged and confirmed with the user
    # before finalizing, same pattern as G5/G7):
    #   Primary:   ||K_x - K_x^T||_2 == ||T - T^T||_2 (M_R symmetric, drops
    #              out) -- the actual operator-level asymmetry non-self-
    #              adjointness guarantees. Exactly 0 at shift=0, monotonic
    #              thereafter.
    #   Secondary: the original |wrong-correct| metric, kept as a weaker
    #              existence check only -- exactly 0 at shift=0 (proves the
    #              bug's necessary condition), nonzero for every shift>0
    #              (proves it's real for a concrete scene), no shape claim.
    #
    # Differentiates w.r.t. d -- a thin-film (elastic-type) parameter that
    # doesn't even appear in K_fl -- deliberately the running_notes' sharpest
    # demonstration: the wrong adjoint is contaminated by non-self-
    # adjointness of the FLUORESCENT channel even when the parameter being
    # differentiated lives entirely in the (self-adjoint) thin-film channel.
    #
    # Wide band (300-800nm, beyond the usual 400-700 visible convention) and
    # lam_ex fixed far from both edges (150nm = 7.5 sigma_f minimum, growing
    # as the sweep proceeds) so fluorescence-Gaussian truncation (T14 lesson)
    # never contaminates this test -- the effect under test is purely the
    # adjoint bug, not a truncation artifact.
    lam_min, lam_max = 300.0, 800.0
    N = 501
    lam = torch.linspace(lam_min, lam_max, N, dtype=torch.float64)
    w   = torch.full((N,), (lam_max - lam_min) / (N - 1), dtype=torch.float64)

    d0, A0, B0 = 120.0, 1.5, 5000.0
    cos_i      = torch.tensor(1.0, dtype=torch.float64)
    lam_ex     = 450.0
    sigma_f    = 20.0
    quantum_yield = 0.5   # keeps rho(T) comfortably < 1 alongside K_tf's R

    L_e = torch.ones(N, dtype=torch.float64)   # flat source -- not the object under test
    g   = torch.ones(N, dtype=torch.float64)   # loss = L.sum(), dloss/dL = 1
    max_depth = 32

    def build_K_fl(lam_em: float) -> torch.Tensor:
        return kernel_fluorescence(lam, lam_ex, lam_em, sigma_f, w, quantum_yield)

    def build_T(lam_em: float) -> torch.Tensor:
        K_tf = kernel_thinfilm(lam, cos_i, d0, A0, B0)
        return K_tf + build_K_fl(lam_em)

    def loss_fn(lam_em: float, d_val: float) -> torch.Tensor:
        K_tf = kernel_thinfilm(lam, cos_i, d_val, A0, B0)
        T    = K_tf + build_K_fl(lam_em)
        return (g * neumann_forward(T, L_e, max_depth)).sum()

    M = 15
    shifts = [250.0 * i / (M - 1) for i in range(M)]   # 0 .. 250nm, includes 0 exactly

    asym_norms, correct, wrong, fds = [], [], [], []
    for shift in shifts:
        lam_em = lam_ex + shift
        K_fl   = build_K_fl(lam_em)
        asym_norms.append(torch.linalg.matrix_norm(K_fl - K_fl.T, ord=2).item())

        T     = build_T(lam_em)
        dR_dd = fabry_airy_dR_dd(lam, cos_i, d0, A0, B0)   # (N,) -- K_fl doesn't depend on d

        correct.append(kernel_gradient(T, dR_dd, L_e, g, max_depth).item())
        wrong.append(kernel_gradient_wrong_adjoint(T, dR_dd, L_e, g, max_depth).item())

        h = 1e-4
        with torch.no_grad():
            fp = loss_fn(lam_em, d0 + h)
            fm = loss_fn(lam_em, d0 - h)
        fds.append(((fp - fm) / (2.0 * h)).item())

    errs = [abs(w_ - c) for w_, c in zip(wrong, correct)]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURES_DIR / "G11_stokes_shift_adjoint_error.csv").open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["stokes_shift", "asym_norm2", "correct_grad", "wrong_grad",
                      "fd_grad", "abs_error"])
        for i in range(M):
            wtr.writerow([shifts[i], asym_norms[i], correct[i], wrong[i], fds[i], errs[i]])

    scale = max(abs(c) for c in correct)
    asym_scale = max(asym_norms)

    # --- Primary: operator-level asymmetry ---------------------------------

    # P1: ||K_x - K_x^T||_2 == 0 exactly at shift=0 (a=e, T*=T).
    asym0 = asym_norms[0]

    # P2: monotonic non-decreasing across the sweep -- allow the tiny
    # numerical wobble the saturated tail actually shows (~1e-7 absolute,
    # confirmed by direct computation), same style as other mono_viol checks.
    asym_mono_viol = sum(
        1 for i in range(M - 1) if asym_norms[i + 1] < asym_norms[i] - 1e-6 * asym_scale
    )

    # P3: substantial growth -- confirms this is a real effect, not noise
    # sitting near the P2 tolerance.
    asym_growth = asym_norms[-1] / max(asym_scale, 1e-30)

    # --- Secondary: fixed-(L_e,g) bilinear projection, weakened -------------

    # S1: |wrong-correct| == 0 exactly at shift=0 -- necessary condition.
    err0 = errs[0]

    # S2: nonzero for every shift>0 -- the bug is real for this concrete
    # scene at every tested separation, even though its magnitude wobbles
    # (no shape claim beyond "nonzero").
    min_nonzero_err = min(errs[1:]) / scale

    # S3: correct adjoint matches FD across the whole sweep (three-way oracle
    # per CLAUDE.md) -- confirms the analytic "correct" form is actually
    # right, independent of the wrong-form comparison or its non-monotonicity.
    max_correct_fd_err = max(abs(c - fd) for c, fd in zip(correct, fds)) / scale

    tol = 1e-6
    return [
        StructResult("G11", "[primary] ||K_x - K_x^T||_2 = 0 exactly at zero Stokes shift",
                     asym0, 0.0, asym0 / max(asym_scale, 1e-30), tol,
                     asym0 / max(asym_scale, 1e-30) < tol,
                     "a=e exactly at shift=0 -- T*=T, operator is exactly self-adjoint"),
        StructResult("G11", "[primary] ||K_x - K_x^T||_2 monotonic non-decreasing beyond shift=0",
                     float(asym_mono_viol), 0.0, float(asym_mono_viol), 0.5, asym_mono_viol == 0,
                     f"{asym_mono_viol} regressions out of {M - 1} steps -- operator asymmetry "
                     "grows monotonically as Stokes shift grows"),
        StructResult("G11", "[primary] substantial operator-asymmetry growth",
                     asym_growth, 1.0, abs(asym_growth - 1.0), 0.5, asym_growth > 0.5,
                     f"asym_norm: {asym_norms[0]:.4f} -> {asym_norms[-1]:.4f} "
                     "(saturates once e,a stop overlapping -- real growth, not noise)"),
        StructResult("G11", "[secondary] |wrong-correct| = 0 exactly at zero Stokes shift",
                     err0, 0.0, err0 / scale, tol, err0 / scale < tol,
                     "necessary condition for the bug, fixed (L_e,S) projection"),
        StructResult("G11", "[secondary] |wrong-correct| nonzero for every shift>0 (no shape claim)",
                     min_nonzero_err, None, None, tol, min_nonzero_err > tol,
                     f"min over shift>0 of |wrong-correct|/scale = {min_nonzero_err:.2e} -- "
                     "deliberately NOT asserting monotonicity here (verified non-monotonic: "
                     "signed error dips, crosses zero near shift~100nm, then grows -- a real "
                     "bilinear-projection effect, confirmed generic across other L_e/g choices)"),
        StructResult("G11", "[secondary] correct adjoint matches FD across full shift sweep (3-way oracle)",
                     max_correct_fd_err, 0.0, max_correct_fd_err, 1e-4, max_correct_fd_err < 1e-4,
                     "scale-normalized max|correct-FD| -- validates the analytic form itself"),
    ]


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
