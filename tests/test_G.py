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

def test_G3() -> StructResult:
    raise NotImplementedError("G3")


# ---------------------------------------------------------------------------
# G4: Multi-fluorophore conditioning vs peak separation
# X-axis: emission-peak separation in units of sigma_e, 5*sigma_e -> 0
# Y-axis: condition number (log scale)
# Predicted shape: smooth monotonic blowup, genuine vertical asymptote at 0;
#                  identify functional form of divergence
# Failure: plateau instead of divergence -> "exact degeneracy" claim wrong
# ---------------------------------------------------------------------------

def test_G4() -> StructResult:
    raise NotImplementedError("G4")


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
