"""T-series: toy numerical tests (small scenes / discretized models).

Each test pushes a specific limit or edge case numerically without a full
path tracer. T0 is the build gate -- run it first every build.

Promotions (point -> sweep, now G-series):
    T3  -> G11   T7  -> G8   T8  -> G9   T10 -> G3

Run:
    conda activate Spectral
    python -m pytest tests/test_T.py -v
  or
    python -m tests.test_T

Tests
-----
T0   Energy conservation gate    -- T_s+R_s=1, T_p+R_p=1 at lossless dielectric
T1   M_R non-compact             -- near-boundary lambda: ||M_R e_n|| -> R(lambda0)
T2   Column-sum bound failure    -- L2 norm blowup at column sums exactly 1
T4   det = eta^2 * c/v           -- both eta<1 and eta>1
T5   J_TIR finite at v=0         -- propagating-side limit + domain guard on evanescent side
T6   Substrate confound rank     -- n_substrate -> n_film: conditioning degrades to unobservable
T9   Rank < 5 in joint limit     -- sigma_f->0 and d->0 simultaneously
T11  Exact rank deficiency       -- coincident emission peaks: hard rank deficiency
T12  TIR + moving boundary combo -- both correction terms compose additively
T13  Substrate TIR clip bug      -- clip(cos,0,None) hides real non-smoothness in R(lambda)
T14  Emission truncation         -- Theorem 7 invariance degrades as lambda_em -> lambda_max
T15  kappa->inf at normal inc.   -- moving-boundary term -> 0 gracefully, no 0*inf
"""
from __future__ import annotations

import sys
import torch
torch.set_default_dtype(torch.float64)

from tests.harness import Reporter, StructResult, GradResult


# ---------------------------------------------------------------------------
# T0: Energy conservation gate  [BUILD GATE -- run every build]
# Claim: T_s + R_s = 1 and T_p + R_p = 1 exactly at a lossless dielectric
#        interface, for arbitrary angle and IOR.
# Pass: exact to float64 machine precision across a sweep of (theta_i, eta).
# ---------------------------------------------------------------------------

def test_T0() -> list[StructResult]:
    # Independent check: compute T from amplitude transmission coeff, not 1-R.
    # n_i=1.0, n_t in [1.1, 2.5] -> no TIR at any angle in the sweep.
    n_i = 1.0
    n_t_vals = torch.linspace(1.1, 2.5, 20)          # (20,)
    theta_deg = torch.linspace(5.0, 85.0, 50)         # (50,)
    cos_i = torch.cos(theta_deg * (torch.pi / 180.0)) # (50,)

    # broadcast: (50, 20)
    cos_i  = cos_i.unsqueeze(1)
    n_t    = n_t_vals.unsqueeze(0)

    sin2_i = 1.0 - cos_i ** 2
    cos_t  = torch.sqrt(1.0 - (n_i / n_t) ** 2 * sin2_i)

    # amplitude coefficients
    rs = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t)
    rp = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t)
    ts = (2.0 * n_i * cos_i) / (n_i * cos_i + n_t * cos_t)
    tp = (2.0 * n_i * cos_i) / (n_t * cos_i + n_i * cos_t)

    # power (independent computation -- not 1-R)
    ratio = n_t * cos_t / (n_i * cos_i)
    Rs = rs ** 2
    Rp = rp ** 2
    Ts = ratio * ts ** 2
    Tp = ratio * tp ** 2

    err_s = (Rs + Ts - 1.0).abs().max().item()
    err_p = (Rp + Tp - 1.0).abs().max().item()
    tol   = 1e-13

    return [
        StructResult("T0", "max |R_s + T_s - 1|", err_s, 0.0, err_s, tol,
                     err_s < tol, "50 angles x 20 eta, s-pol"),
        StructResult("T0", "max |R_p + T_p - 1|", err_p, 0.0, err_p, tol,
                     err_p < tol, "50 angles x 20 eta, p-pol"),
    ]


# ---------------------------------------------------------------------------
# T1: M_R non-compact
# Claim (ss2): ||M_R e_n|| -> R(lambda0) must hold even with lambda0 near the
#              domain edge (lambda_min / lambda_max), not just mid-spectrum.
# Setup: disjoint-bump construction near lambda_min/lambda_max boundary.
# ---------------------------------------------------------------------------

def test_T1() -> list[StructResult]:
    from src.spectral_grid import make_grid
    from src.cauchy_ior import n_cauchy
    from src.fresnel import fresnel_R

    grid = make_grid()
    lam  = grid.lam
    w    = grid.weights
    N    = grid.N

    # Dispersive Cauchy glass s-pol at 45 deg -- gives a non-trivial R(lambda)
    n_t   = n_cauchy(lam, 1.5, 5000.0)
    cos_i = torch.cos(torch.tensor(45.0) * (torch.pi / 180.0)).expand(N)
    R     = fresnel_R(1.0, n_t, cos_i, polarization="s")

    def norm_MR(sl):
        """||M_R e|| for the L2-normalized indicator on slice sl."""
        w_sl   = w[sl]
        norm_sq = w_sl.sum()
        return ((R[sl] ** 2 * w_sl).sum() / norm_sq).sqrt().item()

    def inner_prod(sl_a, sl_b):
        """Weighted L2 inner product of two normalized indicators."""
        ea = torch.zeros(N, dtype=torch.float64)
        eb = torch.zeros(N, dtype=torch.float64)
        ea[sl_a] = 1.0 / w[sl_a].sum().sqrt()
        eb[sl_b] = 1.0 / w[sl_b].sum().sqrt()
        return (ea * eb * w).sum().item()

    results = []
    # blocks shrink as they approach the edge: widths 16,8,4,2,1 grid pts
    # near lambda_min (index 0): blocks lie to the right, converging left
    slices_lo = [slice(16,32), slice(8,16), slice(4,8), slice(2,4), slice(1,2)]
    R0_lo     = R[0].item()
    norms_lo  = [norm_MR(s) for s in slices_lo]
    err_lo    = abs(norms_lo[-1] - R0_lo)
    results.append(StructResult(
        "T1", "||M_R e_n|| -> R(lam_min)",
        norms_lo[-1], R0_lo,
        err_lo / max(abs(R0_lo), 1e-10),
        1e-3, err_lo < 1e-3,
        "seq: " + " ".join(f"{x:.5f}" for x in norms_lo),
    ))

    # near lambda_max (index N-1): blocks converge right
    slices_hi = [slice(N-32,N-16), slice(N-16,N-8), slice(N-8,N-4),
                 slice(N-4,N-2),   slice(N-2,N-1)]
    R0_hi    = R[-1].item()
    norms_hi = [norm_MR(s) for s in slices_hi]
    err_hi   = abs(norms_hi[-1] - R0_hi)
    results.append(StructResult(
        "T1", "||M_R e_n|| -> R(lam_max)",
        norms_hi[-1], R0_hi,
        err_hi / max(abs(R0_hi), 1e-10),
        1e-3, err_hi < 1e-3,
        "seq: " + " ".join(f"{x:.5f}" for x in norms_hi),
    ))

    # disjoint blocks: inner product must be exactly 0
    mid     = N // 2
    ip_dis  = inner_prod(slice(mid-20, mid), slice(mid, mid+20))
    results.append(StructResult(
        "T1", "disjoint <e1,e2> = 0",
        abs(ip_dis), 0.0, abs(ip_dis), 1e-14,
        abs(ip_dis) < 1e-14,
        "disjoint support -> exact orthogonality",
    ))

    # nested blocks: inner product must NOT be 0 (audit catch from ss2)
    # outer=[mid-20:mid+20], inner=[mid-10:mid+10] -- e2 support is inside e1
    ip_nest = inner_prod(slice(mid-20, mid+20), slice(mid-10, mid+10))
    results.append(StructResult(
        "T1", "nested <e1,e2> != 0 (audit catch)",
        abs(ip_nest), None, None, 0.1,
        abs(ip_nest) > 0.1,
        f"nested |<e1,e2>| = {abs(ip_nest):.4f} >> 0 (wrong construction)",
    ))

    return results


# ---------------------------------------------------------------------------
# T2: Column-sum-only bound fails to control L2 norm
# Claim (ss3): toy matrix with column sums exactly 1 (not 0.9) has operator
#              2-norm that blows up arbitrarily as off-diagonal concentration
#              increases -- confirms failure isn't an artifact of the 0.9 margin.
# ---------------------------------------------------------------------------

def test_T2() -> list[StructResult]:
    # Construction: A(c)[i,j] = (1-c)*delta[i,j] + c*delta[i,0]
    #   -- each column has mass c on row-0 and (1-c) on the diagonal.
    #   Column sum = c + (1-c) = 1 exactly for every c and every j.
    #   As c->1 the matrix approaches e_0 * 1^T, whose 2-norm = sqrt(N).
    N  = 50
    cs = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99])

    def make_A(c):
        A = (1.0 - c) * torch.eye(N, dtype=torch.float64)
        A[0, :] += c                  # add c to every entry in row 0
        return A

    col_sum_errs = []
    norms        = []
    for c in cs:
        A = make_A(c.item())
        col_sum_errs.append((A.sum(dim=0) - 1.0).abs().max().item())
        norms.append(torch.linalg.matrix_norm(A, ord=2).item())

    max_col_err = max(col_sum_errs)
    norm_at_high = norms[-1]   # c = 0.99

    return [
        StructResult(
            "T2", "max |col_sum - 1| over all c",
            max_col_err, 0.0, max_col_err, 1e-13,
            max_col_err < 1e-13,
            "column sums exactly 1 by construction",
        ),
        StructResult(
            "T2", "||A||_2 at c=0.99, N=50",
            norm_at_high, None, None, 5.0,
            norm_at_high > 5.0,
            "norms: " + " ".join(f"{x:.3f}" for x in norms),
        ),
    ]


# ---------------------------------------------------------------------------
# T4: det = eta^2 * c/v
# Claim (ss6): Snell Jacobian determinant formula holds for both eta<1 and eta>1.
#              Guards against hidden eta>1-only assumption.
# ---------------------------------------------------------------------------

def test_T4() -> list[StructResult]:
    from src.snell_jacobian import solid_angle_ratio

    theta_deg = torch.linspace(5.0, 70.0, 14)   # (T,)

    def max_rel_err(eta_vals: torch.Tensor) -> float:
        E, T   = len(eta_vals), len(theta_deg)
        eta_g  = eta_vals.view(E, 1).expand(E, T)
        thr    = theta_deg.view(1, T).expand(E, T) * (torch.pi / 180.0)
        c_g    = torch.cos(thr)
        sin_i  = torch.sin(thr)
        sin_t  = eta_g * sin_i

        # exclude TIR and near-TIR (FD accuracy degrades as v->0)
        valid  = sin_t.abs() < 0.95
        v_g    = torch.where(valid,
                             torch.sqrt((1.0 - sin_t**2).clamp(min=1e-12)),
                             torch.ones_like(sin_t))

        # analytic: solid_angle_ratio on the valid subset (flattened)
        m = valid.reshape(-1)
        det_an = solid_angle_ratio(
            eta_g.reshape(-1)[m], torch.ones(m.sum(), dtype=torch.float64),
            c_g.reshape(-1)[m],   v_g.reshape(-1)[m],
        )

        # FD of solid-angle Jacobian: dOmega_t/dOmega_i = (sin_t/sin_i)|d theta_t/d theta_i|
        eps      = 1e-5
        sin_t_p  = eta_g * torch.sin(thr + eps)
        sin_t_m  = eta_g * torch.sin(thr - eps)
        valid_fd = valid & (sin_t_p.abs() < 0.95) & (sin_t_m.abs() < 0.95)
        mf       = valid_fd.reshape(-1)

        theta_t_p = torch.asin(sin_t_p.clamp(-0.9999, 0.9999))
        theta_t_m = torch.asin(sin_t_m.clamp(-0.9999, 0.9999))
        dtheta_t  = (theta_t_p - theta_t_m) / (2.0 * eps)

        det_fd = ((sin_t / sin_i).abs() * dtheta_t.abs()).reshape(-1)[mf]
        det_an_fd = det_an[mf[m]]   # subset of analytic that also passes valid_fd

        return ((det_an_fd - det_fd) / det_an_fd).abs().max().item()

    err_lt1 = max_rel_err(torch.tensor([0.55, 0.65, 0.75, 0.85, 0.95]))
    err_gt1 = max_rel_err(torch.tensor([1.10, 1.25, 1.40, 1.55, 1.70]))
    tol = 1e-7

    return [
        StructResult("T4", "max rel err det=eta^2*c/v, eta<1",
                     err_lt1, 0.0, err_lt1, tol, err_lt1 < tol,
                     "solid_angle_ratio vs FD, 5 eta x 14 theta"),
        StructResult("T4", "max rel err det=eta^2*c/v, eta>1",
                     err_gt1, 0.0, err_gt1, tol, err_gt1 < tol,
                     "solid_angle_ratio vs FD, 5 eta x 14 theta"),
    ]


# ---------------------------------------------------------------------------
# T5: J_TIR finite at v=0
# Claim (ss7): J_TIR^s,p(v) converges to 4*eta / 4*eta^3 on the propagating
#              side. Formula must NOT be evaluated past v=0 (evanescent side)
#              without an explicit domain guard.
# ---------------------------------------------------------------------------

def test_T5() -> list[StructResult]:
    from src.snell_jacobian import tir_jacobian

    # Params from ss7: eta=1.6, c=0.6
    eta = torch.tensor(1.6)
    c   = torch.tensor(0.6)
    n_i = eta.unsqueeze(0)          # (1,)
    n_t = torch.ones(1)
    cos_i = c.unsqueeze(0)

    limit_s = 4.0 * eta             # J_TIR^s(0) = 4*eta
    limit_p = 4.0 * eta ** 3        # J_TIR^p(0) = 4*eta^3

    # -- convergence sweep: v from 0.1 down to 1e-8 (propagating side, v > 0) --
    # convergence is first-order in v: rel_err ~ 2v/(eta*c)
    # at v=1e-10 this gives ~2e-10 for s-pol and ~5e-10 for p-pol
    vs = torch.logspace(-1, -10, 10)   # (10,) spanning six orders of magnitude
    errs_s, errs_p = [], []
    for v_val in vs:
        v = v_val.view(1)
        js = tir_jacobian(v, n_i, n_t, cos_i, polarization="s").item()
        jp = tir_jacobian(v, n_i, n_t, cos_i, polarization="p").item()
        errs_s.append(abs(js - limit_s.item()) / limit_s.item())
        errs_p.append(abs(jp - limit_p.item()) / limit_p.item())

    tol_conv = 1e-9
    results = [
        StructResult(
            "T5", "J_TIR^s(v=1e-8) rel err vs 4*eta",
            errs_s[-1], 0.0, errs_s[-1], tol_conv, errs_s[-1] < tol_conv,
            "converges: " + " ".join(f"{e:.1e}" for e in errs_s),
        ),
        StructResult(
            "T5", "J_TIR^p(v=1e-8) rel err vs 4*eta^3",
            errs_p[-1], 0.0, errs_p[-1], tol_conv, errs_p[-1] < tol_conv,
            "converges: " + " ".join(f"{e:.1e}" for e in errs_p),
        ),
    ]

    # -- domain guard: at v = -0.1 (evanescent), J_TIR^p diverges toward its
    #    pole at v = -c/eta = -0.375.  Relative error must be large (>> 0). --
    v_evan = torch.tensor([-0.1])
    jp_evan = tir_jacobian(v_evan, n_i, n_t, cos_i, polarization="p").item()
    rel_guard = abs(jp_evan - limit_p.item()) / limit_p.item()
    results.append(StructResult(
        "T5", "domain guard: |J_p(v=-0.1) - 4*eta^3|/4*eta^3 >> 0",
        rel_guard, None, None, 0.5,
        rel_guard > 0.5,
        f"J_p(v=-0.1) = {jp_evan:.4f}, limit = {limit_p:.4f} -- guard necessary",
    ))

    return results


# ---------------------------------------------------------------------------
# T6: Substrate confound rank / conditioning
# Claim (ss8): As n_substrate -> n_film (film becomes invisible), conditioning
#              degrades further -- film thickness becomes unobservable with
#              zero index contrast.
# ---------------------------------------------------------------------------

def test_T6() -> list[StructResult]:
    from src.cauchy_ior import n_cauchy, cos_theta_t
    from src.fresnel import fresnel_rs

    lam   = torch.linspace(400.0, 700.0, 60, dtype=torch.float64)
    cos_i = torch.ones(60, dtype=torch.float64)   # normal incidence

    # film params (fixed)
    A_film = 1.5
    B_film = 5000.0
    d_film = 120.0

    def R3_s(d, A, B, C_sub, D_sub=5000.0):
        """3-layer s-pol Fabry-Airy: air / Cauchy(A,B,d) / Cauchy(C_sub,D_sub)."""
        n_f   = n_cauchy(lam, A, B)
        n_s   = n_cauchy(lam, C_sub, D_sub)
        cos_f = cos_theta_t(cos_i, 1.0, n_f)
        cos_s = cos_theta_t(cos_f, n_f, n_s)
        r12   = fresnel_rs(1.0,  n_f, cos_i, cos_f)
        r23   = fresnel_rs(n_f,  n_s, cos_f, cos_s)
        phi   = 4.0 * torch.pi * n_f * cos_f * d / lam
        eiphi = torch.polar(torch.ones_like(phi), phi)
        return (r12 + r23 * eiphi).abs() ** 2 / (1.0 + r12 * r23 * eiphi).abs() ** 2

    def dR_dd_norm(C_sub):
        """||dR/dd||_2: d-column norm of Jacobian at given substrate C.

        Goes to zero as C_sub -> A_film (r23 -> 0, d becomes unobservable).
        """
        d_t = torch.tensor(d_film, dtype=torch.float64, requires_grad=True)
        A_t = torch.tensor(float(A_film), dtype=torch.float64)
        B_t = torch.tensor(float(B_film), dtype=torch.float64)
        col_d = torch.autograd.functional.jacobian(
            lambda d: R3_s(d, A_t, B_t, C_sub), d_t, vectorize=True,
        )
        return col_d.detach().norm().item()

    # sweep C_sub from well-separated (1.8) toward film value (A_film=1.5)
    C_vals = [1.80, 1.70, 1.60, 1.55, 1.52, 1.505]
    norms  = [dR_dd_norm(c) for c in C_vals]

    # claim 1: ||dR/dd|| decreases monotonically as C_sub -> A_film
    monotone = all(norms[i] > norms[i+1] for i in range(len(norms)-1))

    # claim 2: ratio far/close >> 1 (d becomes invisible at zero contrast)
    ratio = norms[0] / max(norms[-1], 1e-30)

    return [
        StructResult(
            "T6", "||dR/dd|| decreases as C_sub -> A_film",
            float(monotone), 1.0, float(not monotone), 0.5,
            monotone,
            "norms: " + " ".join(f"{n:.4f}" for n in norms),
        ),
        StructResult(
            "T6", "||dR/dd||(C=1.80) / ||dR/dd||(C=1.505) >> 1",
            ratio, None, None, 10.0,
            ratio > 10.0,
            f"ratio = {ratio:.1f}x -- d unobservable at zero contrast",
        ),
    ]


# ---------------------------------------------------------------------------
# T9: Rank drops below 5 in joint degenerate limit
# Claim (ss10): As sigma_f->0 AND d->0 simultaneously, rank must actually drop
#               *below 5*, not just worsen -- else the "B is bottleneck among 5"
#               framing hides a deeper exact degeneracy.
# ---------------------------------------------------------------------------

def test_T9() -> list[StructResult]:
    from src.kernels import fabry_airy_R

    lam    = torch.linspace(400.0, 700.0, 80, dtype=torch.float64)
    w      = torch.full((80,), 300.0 / 79.0, dtype=torch.float64)
    cos_i  = torch.tensor(1.0, dtype=torch.float64)
    lam_ex = 450.0
    alpha0 = 1.0

    def L_out_fn(d, A, B, sf, lem):
        R  = fabry_airy_R(lam, cos_i, d, A, B)
        a  = torch.exp(-0.5 * ((lam - lam_ex) / sf) ** 2)
        e  = torch.exp(-0.5 * ((lam - lem)    / sf) ** 2)
        e  = e / (e * w).sum().clamp(min=1e-30)
        return R + e * (a * w).sum() * alpha0

    def jacobian(d, A, B, sf, lem):
        ts = tuple(
            torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for v in [d, A, B, sf, lem]
        )
        cols = torch.autograd.functional.jacobian(
            lambda *args: L_out_fn(*args), ts, vectorize=True,
        )
        return torch.stack([c.detach() for c in cols], dim=1)  # (80, 5)

    # normal operating point — rank should be 5
    J_norm      = jacobian(120.0, 1.5, 5000.0, 30.0, 550.0)
    col_scales  = J_norm.norm(dim=0).clamp(min=1e-30)
    J_norm_n    = J_norm / col_scales
    svs_n       = torch.linalg.svdvals(J_norm_n)
    thr         = 0.01 * svs_n[0].item()   # 1% of largest SV
    rank_normal = int((svs_n > thr).sum().item())

    # joint degenerate limit: sf->0 AND d->0
    J_degen  = jacobian(1e-3, 1.5, 5000.0, 0.1, 550.0)
    J_degen_s = J_degen / col_scales        # same normalization as normal point
    svs_d    = torch.linalg.svdvals(J_degen_s)
    rank_degen = int((svs_d > thr).sum().item())

    return [
        StructResult(
            "T9", "rank at normal point = 5",
            rank_normal, 5.0, abs(rank_normal - 5) / 5.0, 0.01,
            rank_normal == 5,
            "svs: " + " ".join(f"{s:.4f}" for s in svs_n.tolist()),
        ),
        StructResult(
            "T9", "rank drops below 5 in joint degen limit",
            rank_degen, None, None, 4.5,
            rank_degen < 5,
            "svs: " + " ".join(f"{s:.4f}" for s in svs_d.tolist()),
        ),
    ]


# ---------------------------------------------------------------------------
# T11: Exact rank deficiency at coincident emission peaks
# Claim (ss14): When lambda_em,1 = lambda_em,2 (exact coincidence of two
#               fluorophore peaks) the Jacobian becomes exactly rank-deficient
#               (hard wall), not just very large condition number.
# ---------------------------------------------------------------------------

def test_T11() -> StructResult:
    raise NotImplementedError("T11")


# ---------------------------------------------------------------------------
# T12: TIR-adjacent + moving lambda* near boundary, combined
# Claim (ss5/13): In a single scene where v~0 (TIR-adjacent) AND lambda* is
#                 simultaneously near lambda_min/lambda_max, both correction
#                 terms compose additively and correctly.
# ---------------------------------------------------------------------------

def test_T12() -> StructResult:
    raise NotImplementedError("T12")


# ---------------------------------------------------------------------------
# T13: Substrate-side TIR clipping (concrete latent bug)
# Claim (ss8): The clip(cos,0,None) in the substrate-side Fresnel computation
#              smooths a genuine non-smoothness in R(lambda) into a plausible-
#              looking wrong answer. Test that the gradient estimator handles
#              the real discontinuity instead.
# ---------------------------------------------------------------------------

def test_T13() -> StructResult:
    raise NotImplementedError("T13")


# ---------------------------------------------------------------------------
# T14: Emission-line truncation degrades Theorem 7 invariance
# Claim (ss9/14): The exact-invariance proof assumes integration over all lambda'.
#                 The renderer integrates over [lambda_min, lambda_max] only.
#                 As lambda_em approaches lambda_max (or lambda_em + k*sigma_f
#                 exits the band), invariance should measurably degrade.
#                 Must be documented as a real boundary of the theorem.
# ---------------------------------------------------------------------------

def test_T14() -> StructResult:
    raise NotImplementedError("T14")


# ---------------------------------------------------------------------------
# T15: kappa -> inf at normal incidence
# Claim (ss13): As theta_i -> 0, kappa = 1/sin(theta_i) -> inf and lambda*
#               leaves the band. Moving-boundary term must go to exactly 0,
#               not produce a 0*inf in the dlambda*/dtheta formulas.
# ---------------------------------------------------------------------------

def test_T15() -> StructResult:
    raise NotImplementedError("T15")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL = [
    test_T0, test_T1, test_T2, test_T4, test_T5, test_T6,
    test_T9, test_T11, test_T12, test_T13, test_T14, test_T15,
]


def main() -> None:
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
