"""A-series: analytic / sympy proofs.

Each test verifies a closed-form identity or symbolic limit from the theory.
No numerical rendering; most use sympy or pure torch algebra.

Run:
    conda activate Spectral
    python -m pytest tests/test_A.py -v
  or
    python -m tests.test_A

Tests
-----
A1   Airy R in [0,1]              -- symbolic denom-numer=(1-r1^2)(1-r2^2), delta-independent
A2   dR/dd at d->0                -- symbolic limit must vanish or match bare-Fresnel
A3   ||K_x|| equality case        -- Cauchy-Schwarz equality at f = a/||a||
A4   K_x* adjoint degenerate      -- T*=T exactly when e=a; bug iff e != a
A5   v ~ sqrt(2u) branch order    -- Puiseux series, leading u^(1/2) with coeff sqrt(2)
A6   J_||, J_perp, det limits     -- theta_i -> 0: J_||=J_perp=eta; theta_i -> 90: det rate
A7   J_TIR collapse at eta=1      -- J_TIR^s(0)=J_TIR^p(0)=4 when eta=1
A8   Brewster vs v=0              -- T_p(theta_Brewster)=1 is distinct from v=0 cancellation
A9   a_bar -> 0 as sigma_f -> 0   -- no spurious finite floor in the absorption integral
A10  lambda*(A,B) derivatives     -- kappa-A -> 0: dlambda*/dA and dlambda*/dB diverge at same rate
A11  lambda_ex,j invariance       -- symbolic k species: d/dlambda_ex,j of sum is zero
A12  H_wp necessary?              -- open: construct/rule out counterexample where H_wp fails but rho<1
A13  Discrete event score fn      -- open: does event-type selection probability depend on theta?
"""
from __future__ import annotations

import sys
import torch
torch.set_default_dtype(torch.float64)

from tests.harness import Reporter, StructResult, OpenTheoreticalQuestion


# ---------------------------------------------------------------------------
# A1: Airy R in [0,1]
# Claim (ss1): R(lambda;d,A,B) in [0,1] for all phase delta, |r1|,|r2| < 1.
# Symbolic: (1+r1^2*r2^2+2*r1*r2*cos_d) - (r1^2+r2^2+2*r1*r2*cos_d)
#           = (1-r1^2)(1-r2^2), independent of delta.
#
# Note on running_notes' table entry: it writes the residual as
# "(1-r1^2)(1-r2^2)(1-cos(delta))", but sympy confirms the cos(delta) terms
# cancel EXACTLY between numerator and denominator -- the true residual has
# zero delta-dependence (d/d(delta) of the residual is symbolically 0). The
# "(1-cos delta)" factor in the notes appears to be a transcription slip;
# flagging here rather than silently coding around it, same discipline as
# G-series's docstring-vs-reality checks.
# ---------------------------------------------------------------------------

def test_A1() -> list[StructResult]:
    import sympy as sp
    from src.kernels import _fabry_airy_pol

    r1, r2, delta = sp.symbols("r1 r2 delta", real=True)
    numer = r1**2 + r2**2 + 2*r1*r2*sp.cos(delta)
    denom = 1 + r1**2*r2**2 + 2*r1*r2*sp.cos(delta)
    residual = sp.expand(denom - numer)
    target   = sp.expand((1 - r1**2) * (1 - r2**2))

    # Check 1: symbolic identity, exact (no tolerance -- either sympy
    # simplifies the difference to the zero polynomial or it doesn't).
    identity_diff = sp.expand(residual - target)

    # Check 2: residual has zero delta-dependence (contradicts the notes'
    # literal "(1-cos delta)" factor -- confirms it's a transcription slip).
    ddelta = sp.diff(residual, delta)

    # Check 3: free-standing (air-film-air) corollary, r2 = -r1 -- this is
    # the specific case src/kernels.py's _fabry_airy_pol actually implements.
    # Confirms the general two-interface algebra above specializes exactly
    # to the production formula: numer -> 2*r1^2*(1-cos(delta)),
    # denom -> 1 - 2*r1^2*cos(delta) + r1^4.
    numer_ff = sp.expand(numer.subs(r2, -r1))
    denom_ff = sp.expand(denom.subs(r2, -r1))
    numer_ff_target = sp.expand(2*r1**2*(1 - sp.cos(delta)))
    denom_ff_target = sp.expand(1 - 2*r1**2*sp.cos(delta) + r1**4)
    ff_diff = sp.expand((numer_ff - numer_ff_target) + (denom_ff - denom_ff_target))

    # Check 4: numeric cross-check of the free-standing corollary against
    # the ACTUAL runtime code (_fabry_airy_pol), not just its symbolic
    # target -- random r12 in (-1,1), random phi -- confirms sympy's
    # specialization matches what the codebase actually computes.
    torch.manual_seed(0)
    n_samples = 2000
    r12_t = (torch.rand(n_samples, dtype=torch.float64) * 2 - 1) * 0.999
    phi_t = torch.rand(n_samples, dtype=torch.float64) * 2 * torch.pi
    R_code = _fabry_airy_pol(r12_t, phi_t)
    R_formula = (2 * r12_t**2 * (1 - torch.cos(phi_t))) / (1 - 2*r12_t**2*torch.cos(phi_t) + r12_t**4)
    max_code_formula_err = (R_code - R_formula).abs().max().item()

    # Check 5: numeric bound sweep over the GENERAL two-interface formula
    # (independent r1,r2, not just the free-standing r2=-r1 case) -- confirm
    # R stays in [0,1] and the analytic margin (1-r1^2)(1-r2^2) matches the
    # directly-computed denom-numer to machine precision, for many random
    # (r1,r2,delta) triples with |r1|,|r2|<1.
    r1_t = (torch.rand(n_samples, dtype=torch.float64) * 2 - 1) * 0.999
    r2_t = (torch.rand(n_samples, dtype=torch.float64) * 2 - 1) * 0.999
    d_t  = torch.rand(n_samples, dtype=torch.float64) * 2 * torch.pi
    numer_n = r1_t**2 + r2_t**2 + 2*r1_t*r2_t*torch.cos(d_t)
    denom_n = 1 + r1_t**2*r2_t**2 + 2*r1_t*r2_t*torch.cos(d_t)
    R_n = numer_n / denom_n
    margin_analytic = (1 - r1_t**2) * (1 - r2_t**2)
    margin_direct   = denom_n - numer_n
    max_margin_err  = (margin_analytic - margin_direct).abs().max().item()
    R_min, R_max = R_n.min().item(), R_n.max().item()

    return [
        StructResult("A1", "symbolic identity: denom-numer == (1-r1^2)(1-r2^2) exactly",
                     0.0 if identity_diff == 0 else 1.0, 0.0,
                     0.0 if identity_diff == 0 else 1.0, 0.0, identity_diff == 0,
                     f"sympy.expand(residual - target) = {identity_diff}"),
        StructResult("A1", "residual has zero delta-dependence (notes' '(1-cos delta)' factor doesn't survive)",
                     0.0 if ddelta == 0 else 1.0, 0.0,
                     0.0 if ddelta == 0 else 1.0, 0.0, ddelta == 0,
                     f"d(residual)/d(delta) = {ddelta} -- flags a transcription slip in running_notes' A1 row"),
        StructResult("A1", "free-standing (r2=-r1) corollary matches code's target formula symbolically",
                     0.0 if ff_diff == 0 else 1.0, 0.0,
                     0.0 if ff_diff == 0 else 1.0, 0.0, ff_diff == 0,
                     f"sympy.expand(...) residual = {ff_diff} -- general algebra specializes exactly "
                     "to _fabry_airy_pol's air-film-air convention"),
        StructResult("A1", "free-standing corollary matches ACTUAL _fabry_airy_pol runtime, numerically",
                     max_code_formula_err, 0.0, max_code_formula_err, 1e-13, max_code_formula_err < 1e-13,
                     f"max|R_code - R_formula| over {n_samples} random (r12,phi) samples"),
        StructResult("A1", "numeric bound sweep: R in [0,1] and analytic margin matches denom-numer directly",
                     max(max_margin_err, max(0.0, -R_min), max(0.0, R_max - 1.0)), 0.0,
                     max(max_margin_err, max(0.0, -R_min), max(0.0, R_max - 1.0)), 1e-12,
                     max_margin_err < 1e-12 and R_min >= -1e-12 and R_max <= 1.0 + 1e-12,
                     f"R range [{R_min:.6f}, {R_max:.6f}] over {n_samples} random (r1,r2,delta) "
                     f"triples, |r1|,|r2|<1; max|analytic margin - direct margin| = {max_margin_err:.2e}"),
    ]


# ---------------------------------------------------------------------------
# A2: dR/dd at d->0
# Claim (ss1): symbolic limit of dR/dd as d->0 must vanish or match
#              bare-Fresnel derivative -- zero-thickness film has no interference.
#
# Resolved as "vanishes", by a cleaner argument than direct differentiation:
# phi(d) = k*d is ODD in d (k = 4*pi*n*cos_t/lam, constant w.r.t. d), and
# R depends on d only through cos(phi(d)) -- cos is even in its argument, so
# R(-d) = R(d) IDENTICALLY (free-standing air-film-air case, r2=-r1). An even
# function's derivative at 0 is always exactly 0, independent of the precise
# rational-function form -- a more general proof than computing dR/dd
# explicitly and taking the limit (done too, as a second confirmation).
# Physically: shrinking a free-standing film to zero thickness makes it
# disappear into the surrounding medium (air-air, no interface at all), not
# "the bare single-surface Fresnel reflectance" -- that alternative only
# applies to a film on a DIFFERENT substrate, which fabry_airy_R doesn't
# model (kernel_thinfilm is air-film-air only; see G7's separate 3-layer
# R3_s for the substrate case).
# ---------------------------------------------------------------------------

def test_A2() -> list[StructResult]:
    import sympy as sp
    from src.kernels import fabry_airy_dR_dd

    r1, k, d = sp.symbols("r1 k d", real=True)
    phi   = k * d
    numer = 2 * r1**2 * (1 - sp.cos(phi))
    denom = 1 - 2 * r1**2 * sp.cos(phi) + r1**4
    R     = numer / denom

    # Check 1: R(d) is even in d -- symbolic, general r1,k.
    even_diff = sp.simplify(R - R.subs(d, -d))

    # Check 2: therefore dR/dd at d=0 is exactly 0 -- direct symbolic
    # confirmation via differentiation + limit, general r1,k.
    dRdd    = sp.diff(R, d)
    dRdd_at0 = sp.limit(dRdd, d, 0)

    # Check 3: the ACTUAL code's analytic fabry_airy_dR_dd gives exactly 0 at
    # d=0, for many random physical (A,B,cos_i) configurations -- not just
    # the abstract r1,k symbols above.
    torch.manual_seed(2)
    n_trials = 50
    N = 20
    lam = torch.linspace(400.0, 700.0, N, dtype=torch.float64)
    max_d0_val = 0.0
    for _ in range(n_trials):
        A_r     = (torch.rand(()) * 1.0 + 1.0).item()          # A in [1,2)
        B_r     = (torch.rand(()) * 8000.0 + 1000.0).item()    # B in [1000,9000)
        cos_i_r = (torch.rand(()) * 0.9 + 0.1).item()          # cos_i in [0.1,1.0)
        dR = fabry_airy_dR_dd(lam, torch.tensor(cos_i_r), 0.0, A_r, B_r)
        max_d0_val = max(max_d0_val, dR.abs().max().item())

    # Check 4: sanity -- away from d=0, dR/dd is generically nonzero. This
    # isn't a formula that's identically 0 everywhere; the vanishing is
    # specific to the degenerate d=0 point.
    dR_away = fabry_airy_dR_dd(lam, torch.tensor(0.8), 120.0, 1.5, 5000.0)
    away_max = dR_away.abs().max().item()

    return [
        StructResult("A2", "symbolic: R(d) is even in d (free-standing film)",
                     0.0 if even_diff == 0 else 1.0, 0.0,
                     0.0 if even_diff == 0 else 1.0, 0.0, even_diff == 0,
                     f"sympy.simplify(R(d)-R(-d)) = {even_diff}"),
        StructResult("A2", "symbolic: dR/dd -> 0 exactly at d=0 (even function's derivative at 0)",
                     0.0 if dRdd_at0 == 0 else 1.0, 0.0,
                     0.0 if dRdd_at0 == 0 else 1.0, 0.0, dRdd_at0 == 0,
                     f"sympy.limit(dR/dd, d, 0) = {dRdd_at0}"),
        StructResult("A2", "actual fabry_airy_dR_dd = 0 exactly at d=0, over random (A,B,cos_i)",
                     max_d0_val, 0.0, max_d0_val, 1e-13, max_d0_val < 1e-13,
                     f"max|dR/dd| at d=0 over {n_trials} random physical configs"),
        StructResult("A2", "sanity: dR/dd is generically NONZERO away from d=0",
                     away_max, None, None, 1e-6, away_max > 1e-6,
                     f"max|dR/dd| at d=120nm = {away_max:.4f} -- vanishing is specific to d=0, "
                     "not a trivially-zero formula"),
    ]


# ---------------------------------------------------------------------------
# A3: ||K_x|| = ||e||_2 * ||a||_2, equality case
# Claim (ss2): Cauchy-Schwarz equality is achieved at f = a/||a||_2;
#              closed-form identity, not just an inequality.
#
# (K_x f)(lam) = e(lam) * <a,f> -- a rank-1 operator, so ||K_x f||_w =
# ||e||_w * |<a,f>_w| exactly, and Cauchy-Schwarz gives |<a,f>_w| <=
# ||a||_w ||f||_w with equality iff f is proportional to a. Symbolic proof
# via Lagrange's identity (general N, shown here for N=3): the classic
# "sum of squares" form makes both the inequality AND its equality condition
# transparent in one identity, not just an inequality bound.
# ---------------------------------------------------------------------------

def test_A3() -> list[StructResult]:
    import sympy as sp
    from src.kernels import kernel_fluorescence

    # --- Symbolic: Lagrange's identity, N=3 ---------------------------------
    a1, a2, a3, f1, f2, f3, w1, w2, w3 = sp.symbols(
        "a1 a2 a3 f1 f2 f3 w1 w2 w3", real=True)
    a_s, f_s, w_s = [a1, a2, a3], [f1, f2, f3], [w1, w2, w3]

    lhs = (sum(a_s[i]**2 * w_s[i] for i in range(3))
           * sum(f_s[i]**2 * w_s[i] for i in range(3))
           - sum(a_s[i] * f_s[i] * w_s[i] for i in range(3))**2)
    rhs = sum(w_s[i] * w_s[j] * (a_s[i]*f_s[j] - a_s[j]*f_s[i])**2
              for i in range(3) for j in range(i + 1, 3))

    # Check 1: (||a||_w^2)(||f||_w^2) - <a,f>_w^2 == sum-of-squares form
    # exactly -- makes ||a||_w||f||_w >= |<a,f>_w| manifest (RHS >= 0).
    lagrange_diff = sp.expand(lhs - rhs)

    # Check 2: substituting f=c*a into the sum-of-squares form gives
    # identically 0 -- confirms equality holds exactly (not just in the
    # limit) precisely when f is proportional to a.
    c = sp.symbols("c", real=True)
    rhs_at_prop = sp.expand(rhs.subs({f1: c*a1, f2: c*a2, f3: c*a3}))

    # --- Numeric: actual kernel_fluorescence, weighted L2 norms -------------
    torch.manual_seed(3)
    N = 80
    lam = torch.linspace(400.0, 700.0, N, dtype=torch.float64)
    w   = torch.full((N,), 300.0 / (N - 1), dtype=torch.float64)
    lam_ex, lam_em, sigma_f = 450.0, 550.0, 20.0

    K = kernel_fluorescence(lam, lam_ex, lam_em, sigma_f, w, quantum_yield=1.0)
    a = torch.exp(-0.5 * ((lam - lam_ex) / sigma_f) ** 2)
    e = torch.exp(-0.5 * ((lam - lam_em) / sigma_f) ** 2)
    e = e / (e * w).sum()

    def norm_w(g: torch.Tensor) -> torch.Tensor:
        return ((g ** 2 * w).sum()) ** 0.5

    norm_e, norm_a = norm_w(e), norm_w(a)
    bound = (norm_e * norm_a).item()

    # Check 3: Cauchy-Schwarz bound never exceeded, many random unit-w-norm f.
    max_ratio = 0.0
    for _ in range(500):
        f = torch.randn(N, dtype=torch.float64)
        f = f / norm_w(f)
        max_ratio = max(max_ratio, (norm_w(K @ f) / bound).item())

    # Check 4: exact equality at f* = a/||a||_w -- the closed-form case,
    # not just an inequality.
    f_star = a / norm_a
    equality_val = norm_w(K @ f_star).item()

    # Check 5: the supremum is achieved SPECIFICALLY at f proportional to a
    # -- construct the w-whitened operator M = diag(sqrt(w)) K diag(1/sqrt(w))
    # (converts sup_{||f||_w=1}||Kf||_w into an ordinary Euclidean operator
    # norm problem), confirm its top singular value matches the bound AND
    # its top right singular vector, mapped back to f-space, is parallel to a.
    sqrt_w = torch.sqrt(w)
    M = torch.diag(sqrt_w) @ K @ torch.diag(1.0 / sqrt_w)
    svals = torch.linalg.svdvals(M)
    top_sv = svals[0].item()

    _, _, Vh = torch.linalg.svd(M)
    f_top = Vh[0] / sqrt_w
    f_top = f_top / norm_w(f_top)
    cos_align = (f_top * a / norm_a * w).sum().item()   # <f_top, a/||a||>_w

    tol = 1e-10
    return [
        StructResult("A3", "symbolic: Lagrange's identity (N=3) exact",
                     0.0 if lagrange_diff == 0 else 1.0, 0.0,
                     0.0 if lagrange_diff == 0 else 1.0, 0.0, lagrange_diff == 0,
                     f"sympy.expand(lhs-rhs) = {lagrange_diff}"),
        StructResult("A3", "symbolic: equality-condition term vanishes exactly at f=c*a",
                     0.0 if rhs_at_prop == 0 else 1.0, 0.0,
                     0.0 if rhs_at_prop == 0 else 1.0, 0.0, rhs_at_prop == 0,
                     f"sum-of-squares form at f=c*a: {rhs_at_prop}"),
        StructResult("A3", "Cauchy-Schwarz bound never exceeded (500 random unit-w-norm f)",
                     max_ratio, None, None, 1.0 + tol, max_ratio <= 1.0 + tol,
                     f"max ||Kf||_w / (||e||_w ||a||_w) = {max_ratio:.6f} (<=1 required)"),
        StructResult("A3", "exact equality at f* = a/||a||_w (closed-form case)",
                     equality_val, bound, abs(equality_val - bound) / bound, tol,
                     abs(equality_val - bound) / bound < tol,
                     f"||K f*||_w = {equality_val:.10f} vs ||e||_w*||a||_w = {bound:.10f}"),
        StructResult("A3", "sup is achieved specifically at f proportional to a (top singular vector)",
                     abs(cos_align), 1.0, abs(abs(cos_align) - 1.0), tol, abs(cos_align) > 1.0 - tol,
                     f"top singular value = {top_sv:.10f} (matches bound); "
                     f"|<f_top, a/||a||>_w| = {abs(cos_align):.10f} -- confirms which f achieves the sup"),
    ]


# ---------------------------------------------------------------------------
# A4: K_x* adjoint degenerate
# Claim (ss4): T*=T exactly when e=a (zero Stokes shift).
#              Wrong adjoint must become numerically correct *only* in this limit.
#
# Nails down the "iff" symbolically/exactly, complementing G11's numeric
# sweep (which observed the same fact empirically as a byproduct of a
# different test). K_x(lam,lam')=e(lam)a(lam') is symmetric (K_x=K_x^T,
# i.e. e_i a_j = e_j a_i for all i,j) iff e and a are proportional as
# vectors -- proved both for general vectors (small symbolic N) and for the
# specific Gaussian a,e used everywhere in this codebase (proportional iff
# lam_ex=lam_em exactly, via a log-linear argument independent of sigma_f).
# ---------------------------------------------------------------------------

def test_A4() -> list[StructResult]:
    import sympy as sp
    from src.kernels import kernel_fluorescence

    # --- Symbolic: general-vector symmetry <=> proportionality -------------
    e1, e2, e3, a1, a2, a3, k = sp.symbols("e1 e2 e3 a1 a2 a3 k", real=True)
    e_s, a_s = [e1, e2, e3], [a1, a2, a3]

    def asym(i: int, j: int):
        return e_s[i] * a_s[j] - e_s[j] * a_s[i]

    # Check 1: sufficient direction -- e=k*a (proportional) makes every
    # antisymmetric entry vanish identically, for symbolic k,a1,a2,a3.
    prop_subs = {e1: k*a1, e2: k*a2, e3: k*a3}
    prop_vals_sym = [sp.simplify(asym(i, j).subs(prop_subs))
                      for i in range(3) for j in range(i + 1, 3)]
    prop_vals_zero = [v == 0 for v in prop_vals_sym]   # symbolic (contain free k,a1,a2,a3) -- compare structurally

    # Check 2: NOT vacuous -- a generic non-proportional numeric substitution
    # gives nonzero antisymmetric entries (confirms Check 1 isn't trivially
    # true for every e,a; symmetry is a genuine constraint).
    generic_vals = [float(asym(i, j).subs({e1: 1, e2: 2, e3: 3, a1: 1, a2: 1, a3: 1}))
                     for i in range(3) for j in range(i + 1, 3)]

    # Check 3: specific Gaussian case -- e_raw(lam)/a_raw(lam) has a
    # log-derivative w.r.t. lam that is the CONSTANT (lam_em-lam_ex)/sigma^2,
    # independent of lam and of sigma_f's value -- so the ratio itself is
    # constant across lam (i.e. e_raw proportional to a_raw) iff that
    # constant is 0, i.e. iff lam_ex = lam_em exactly. Any nonzero
    # separation gives a genuinely lam-dependent (non-constant) ratio.
    lam, lam_ex, lam_em, sigma = sp.symbols("lam lam_ex lam_em sigma", positive=True, real=True)
    log_ratio  = -((lam - lam_em)**2 - (lam - lam_ex)**2) / (2*sigma**2)
    dlog_dlam  = sp.simplify(sp.diff(log_ratio, lam))
    dlog_target = (lam_em - lam_ex) / sigma**2
    dlog_diff  = sp.simplify(dlog_dlam - dlog_target)

    # --- Numeric: actual kernel_fluorescence, symmetry sweep ----------------
    N = 80
    lam_t = torch.linspace(400.0, 700.0, N, dtype=torch.float64)
    w     = torch.full((N,), 300.0 / (N - 1), dtype=torch.float64)
    sigma_f = 20.0

    def asym_norm(lex: float, lem: float) -> float:
        K = kernel_fluorescence(lam_t, lex, lem, sigma_f, w)
        return (K - K.T).abs().max().item()

    # Check 4: exactly symmetric (0 to machine precision) at lam_ex=lam_em.
    asym_at_zero = asym_norm(500.0, 500.0)

    # Check 5: strictly asymmetric (nonzero) for EVERY nonzero separation
    # tested, down to 1e-6nm -- confirms the "iff" holds pointwise, not just
    # "small separation gives small asymmetry" (which alone wouldn't rule
    # out an accidental exact-zero at some small nonzero separation).
    seps = [1e-6, 1e-3, 0.1, 1.0, 10.0, 50.0]
    asym_vals = [asym_norm(500.0, 500.0 + s) for s in seps]
    min_nonzero_asym = min(asym_vals)

    all_prop_zero = all(prop_vals_zero)
    tol = 1e-12
    return [
        StructResult("A4", "symbolic: symmetric identically when e=k*a (sufficient direction)",
                     0.0 if all_prop_zero else 1.0, 0.0,
                     0.0 if all_prop_zero else 1.0, 0.0, all_prop_zero,
                     f"asymmetric entries at e=k*a: {prop_vals_sym}"),
        StructResult("A4", "symbolic: NOT vacuous -- generic e,a give nonzero asymmetry",
                     float(min(abs(v) for v in generic_vals) > 0), 1.0,
                     0.0, 0.0, all(v != 0 for v in generic_vals),
                     f"asymmetric entries at generic e,a: {generic_vals}"),
        StructResult("A4", "symbolic: d/dlam log(e_raw/a_raw) = (lam_em-lam_ex)/sigma^2 exactly",
                     0.0 if dlog_diff == 0 else 1.0, 0.0,
                     0.0 if dlog_diff == 0 else 1.0, 0.0, dlog_diff == 0,
                     f"sympy.simplify(dlog_dlam - target) = {dlog_diff} -- ratio is lam-independent "
                     "(proportional) iff lam_ex=lam_em exactly, for ANY sigma_f"),
        StructResult("A4", "actual kernel_fluorescence: exactly symmetric at lam_ex=lam_em",
                     asym_at_zero, 0.0, asym_at_zero, tol, asym_at_zero < tol,
                     "||K-K^T||_max at lam_ex=lam_em=500 -- machine precision"),
        StructResult("A4", "actual kernel_fluorescence: strictly asymmetric for every nonzero separation",
                     min_nonzero_asym, None, None, tol, min_nonzero_asym > tol,
                     f"min ||K-K^T||_max over separations {seps} = {min_nonzero_asym:.3e} -- "
                     "no accidental exact-symmetry point at any nonzero separation tested"),
    ]


# ---------------------------------------------------------------------------
# A5: v ~ sqrt(2u) branch order
# Claim (ss5): Puiseux series of v(u) about u=0 has leading order u^(1/2)
#              with coefficient sqrt(2); no u^1 or u^0 term at leading order.
#
# Section 5's setup: u := 1-s, s = (n_i/n_t)*sin(theta_i) (distance from
# criticality in the natural sampling variable), v = cos(theta_t) =
# sqrt(1-s^2) = sqrt((1-s)(1+s)) = sqrt(u*(2-u)) since 1+s = 2-u. Section 5
# asserts v ~ sqrt(2u) near u=0 and (separately, from prior-session scratch
# work) that every subsequent term in the series sits at a half-integer
# power (3/2, 5/2, ...) -- no integer-power (u^0, u^1) leakage that would
# make v locally look linear or offset near the branch point.
# ---------------------------------------------------------------------------

def test_A5() -> list[StructResult]:
    import sympy as sp
    from src.cauchy_ior import cos_theta_t

    u = sp.symbols("u", positive=True)
    v = sp.sqrt(u * (2 - u))

    # Check 1: symbolic Puiseux series about u=0, leading term coefficient.
    n_terms = 4
    ser = sp.series(v, u, 0, n_terms).removeO()
    terms = sp.Add.make_args(sp.expand(ser))
    coeff_exp = sorted((t.as_coeff_exponent(u) for t in terms), key=lambda ce: ce[1])
    leading_coeff, leading_exp = coeff_exp[0]
    leading_diff = sp.simplify(leading_coeff - sp.sqrt(2))

    # Check 2: leading exponent is exactly 1/2 (the branch-point order).
    exp_diff = sp.simplify(leading_exp - sp.Rational(1, 2))

    # Check 3: no u^0 or u^1 (integer-power) term survives anywhere in the
    # expansion -- reject if either appears among the recovered exponents.
    exponents = [ce[1] for ce in coeff_exp]
    has_integer_leak = any(e == 0 or e == 1 for e in exponents)

    # Check 4: every recovered exponent is a half-integer (odd multiple of
    # 1/2) -- the "half-integer powers only" claim from prior scratch work,
    # not just "the leading term happens to be u^(1/2)".
    all_half_integer = all(sp.simplify(2 * e - sp.floor(2 * e)) == 0 and
                            sp.simplify(e - sp.floor(e)) != 0 for e in exponents)

    # Check 5: numeric, using the abstract sqrt(u*(2-u)) formula directly --
    # v/sqrt(2u) -> 1 as u -> 0, swept across many orders of magnitude. The
    # asymptotic is a NEAR-u=0 statement, not a global identity (at u=0.5 the
    # O(u) correction is a genuine 13% effect) -- so the check is convergence
    # (deviation shrinks monotonically, final point near-exact), not a flat
    # bound over the whole sweep.
    u_vals = torch.tensor([0.5 * 10 ** (-k) for k in range(8)], dtype=torch.float64)
    v_vals = torch.sqrt(u_vals * (2 - u_vals))
    dev_abstract = (v_vals / torch.sqrt(2 * u_vals) - 1.0).abs()
    monotone_abstract = bool((dev_abstract[1:] <= dev_abstract[:-1] + 1e-15).all())
    final_dev_abstract = dev_abstract[-1].item()

    # Check 6: cross-check against the ACTUAL cos_theta_t() code (not just
    # the abstract u,v symbols) -- scalar Snell setup matching sec5's
    # verification note (n_i=1.5, n_t=A=1.0), sweeping incidence angle to
    # approach the critical angle from u=0.5 down to u=5e-9. cos_theta_t()
    # must match the sqrt(u(2-u)) formula EXACTLY at every u (both compute
    # the same sqrt(1-s^2), just via different intermediate variables) --
    # that's a flat check; only the v/sqrt(2u)->1 trend is asymptotic.
    n_i, n_t = 1.5, 1.0
    eta = n_t / n_i
    s_vals = 1.0 - u_vals
    sin_i_vals = s_vals * eta
    cos_i_vals = torch.sqrt(1.0 - sin_i_vals ** 2)
    v_code = cos_theta_t(cos_i_vals, n_i, n_t)
    max_code_formula_err = (v_code - v_vals).abs().max().item()
    final_dev_code = (v_code[-1] / torch.sqrt(2 * u_vals[-1]) - 1.0).abs().item()

    tol = 1e-9
    conv_tol = 2e-8   # final point is u=5e-8; deviation ~ u/4, not machine-eps
    return [
        StructResult("A5", "symbolic: Puiseux leading coefficient == sqrt(2) exactly",
                     0.0 if leading_diff == 0 else 1.0, 0.0,
                     0.0 if leading_diff == 0 else 1.0, 0.0, leading_diff == 0,
                     f"leading term = {leading_coeff}*u^{leading_exp}; coeff - sqrt(2) = {leading_diff}"),
        StructResult("A5", "symbolic: leading exponent == 1/2 exactly",
                     0.0 if exp_diff == 0 else 1.0, 0.0,
                     0.0 if exp_diff == 0 else 1.0, 0.0, exp_diff == 0,
                     f"leading exponent = {leading_exp}"),
        StructResult("A5", "symbolic: no u^0 or u^1 term survives in the expansion",
                     1.0 if has_integer_leak else 0.0, 0.0,
                     1.0 if has_integer_leak else 0.0, 0.0, not has_integer_leak,
                     f"recovered exponents = {exponents}"),
        StructResult("A5", "symbolic: every surviving term sits at a half-integer power",
                     0.0 if all_half_integer else 1.0, 0.0,
                     0.0 if all_half_integer else 1.0, 0.0, all_half_integer,
                     f"exponents {exponents} -- all odd multiples of 1/2, no even (integer) powers"),
        StructResult("A5", "numeric (abstract sqrt(u(2-u))): v/sqrt(2u) -> 1 as u -> 0, monotonically",
                     final_dev_abstract, 0.0, final_dev_abstract, conv_tol,
                     monotone_abstract and final_dev_abstract < conv_tol,
                     f"deviation |v/sqrt(2u)-1| shrinks monotonically over u in {u_vals.tolist()}; "
                     f"final (u={u_vals[-1].item():.0e}) = {final_dev_abstract:.2e}"),
        StructResult("A5", "actual cos_theta_t() code matches sqrt(u(2-u)) formula exactly, and same v/sqrt(2u)->1 limit",
                     max(max_code_formula_err, final_dev_code), 0.0,
                     max(max_code_formula_err, final_dev_code),
                     max(tol, conv_tol), max_code_formula_err < tol and final_dev_code < conv_tol,
                     f"n_i={n_i}, n_t={n_t}: max|v_code - sqrt(u(2-u))| = {max_code_formula_err:.2e} (flat, all u), "
                     f"final |v_code/sqrt(2u) - 1| = {final_dev_code:.2e} (asymptotic, u={u_vals[-1].item():.0e})"),
    ]


# ---------------------------------------------------------------------------
# A6: J_||, J_perp, det limits
# Claim (ss6): At theta_i=0: J_||=J_perp=eta exactly (formulas collapse).
#              At theta_i->90 deg: confirm divergence *rate* of det.
#
# J_perp=eta (constant), J_par=eta*cos_i/cos_t, det=J_perp*J_par=eta^2*cos_i/cos_t
# (G2's tangent-plane factorization; eta=n_i/n_t).
#
# Note on the table row's literal "theta_i->90 deg" phrasing: the valid
# incidence-angle domain is theta_i in [0,90) only for eta<=1 (n_i<=n_t, no
# TIR possible) -- for eta>1 the domain is truncated at the critical angle
# theta_c=asin(1/eta)<90 deg, beyond which cos_t is not real at all. So a
# literal theta_i->90 limit and a "det diverges" claim can't both hold for
# the SAME eta: at eta<=1, theta_i really does reach 90 deg, but det->0 (or
# ->1 exactly, eta=1) there, not infinity; the genuine divergence only
# happens at eta>1, approaching theta_c (the actual domain edge, which the
# table's phrasing loosely calls "90 deg"). Same flag-and-proceed discipline
# as A1's "(1-cos delta)" transcription slip -- both regimes are implemented
# below so the table row is fully covered rather than silently picking one.
# The eta>1 divergence rate connects directly to A5: v~sqrt(2u) near the
# critical angle implies det=eta^2*c/v ~ (eta^2*c_c)/sqrt(2u), a u^(-1/2)
# pole, not the naive "diverges like 1/u" a bare 1/v pole might suggest.
# ---------------------------------------------------------------------------

def test_A6() -> list[StructResult]:
    import sympy as sp
    from src.cauchy_ior import cos_theta_t
    from src.snell_jacobian import solid_angle_ratio

    # --- Check 1: symbolic collapse at theta_i=0, general eta --------------
    eta_s = sp.symbols("eta", positive=True)
    # theta_i=0 -> sin2_i=0 -> sin2_t=eta^2*0=0 -> cos_t=1 exactly (Snell).
    cos_i_0, cos_t_0 = sp.Integer(1), sp.Integer(1)
    J_par_0  = eta_s * cos_i_0 / cos_t_0
    J_perp_0 = eta_s
    collapse_diff = sp.simplify(J_par_0 - J_perp_0)

    # --- Check 2: numeric cross-check against the ACTUAL code at theta_i=0,
    # several eta values (both <1 and >1).
    etas = [0.5, 0.67, 1.0, 1.5, 2.0]
    n_t = 1.0
    max_collapse_err = 0.0
    for eta in etas:
        n_i = eta * n_t
        cos_i = torch.tensor(1.0)
        cos_t = cos_theta_t(cos_i, n_i, n_t)
        J_perp = torch.tensor(eta)
        J_par = eta * cos_i / cos_t
        max_collapse_err = max(max_collapse_err, abs(J_par.item() - J_perp.item()) / eta)

    # --- Check 3: eta=1 (matched media) -- det is IDENTICALLY 1 across the
    # WHOLE domain including theta_i->90, not just at 0 -- no divergence at
    # all when there's no index contrast, the cleanest possible sanity edge.
    # theta_i=89.9999 deg is deliberately excluded: cos_theta_t's internal
    # under=1-sin2_t=1-(1-cos_i^2) is a genuine catastrophic-cancellation
    # landmine there (cos_i~1.7e-6, so 1-(1-cos_i^2) loses ~4 digits) -- same
    # numerical family as T13/T15/G1, not a bug, just outside this check's
    # useful range; 89.999 deg already confirms the identity to 3.7e-8.
    theta_i = torch.tensor([1.0, 30.0, 60.0, 89.0, 89.9, 89.99, 89.999]) * torch.pi / 180
    cos_i = torch.cos(theta_i)
    cos_t_matched = cos_theta_t(cos_i, 1.0, 1.0)
    det_matched = solid_angle_ratio(torch.tensor(1.0), torch.tensor(1.0), cos_i, cos_t_matched)
    max_matched_dev = (det_matched - 1.0).abs().max().item()

    # --- Check 4: eta<1 (no TIR possible, domain genuinely reaches 90 deg)
    # -- det -> 0 as theta_i -> 90, at an exact LINEAR rate in cos_i:
    # det/cos_i -> eta^2/sqrt(1-eta^2), NOT a divergence. This is the case
    # the table's literal "theta_i->90" phrasing actually describes without
    # contradiction (domain reaches 90 for real), but the limit is finite.
    # The rate is asymptotic (near theta_i=90 only, same caveat as A5's
    # v/sqrt(2u)) so the check is convergence -- deviation shrinks
    # monotonically and the final point is near-exact -- not a flat bound
    # over a range that also includes far-from-grazing angles.
    eta_sub = 0.625
    n_i_sub = eta_sub * n_t
    theta_i_grazing = torch.tensor([80.0, 85.0, 88.0, 89.0, 89.9, 89.99, 89.999, 89.9999]) * torch.pi / 180
    cos_i_grazing = torch.cos(theta_i_grazing)
    cos_t_sub = cos_theta_t(cos_i_grazing, n_i_sub, n_t)
    det_sub = solid_angle_ratio(torch.tensor(n_i_sub), torch.tensor(n_t), cos_i_grazing, cos_t_sub)
    predicted_rate = eta_sub ** 2 / (1.0 - eta_sub ** 2) ** 0.5
    rate_dev = (det_sub / cos_i_grazing - predicted_rate).abs()
    rate_monotone = bool((rate_dev[1:] <= rate_dev[:-1] + 1e-15).all())
    final_rate_dev = (rate_dev[-1] / predicted_rate).item()

    # --- Check 5: eta>1 (true critical angle theta_c < 90 deg) -- det
    # DOES diverge, but at theta_i->theta_c (the actual domain edge), not
    # literal 90 deg, and at rate u^(-1/2) per A5's v~sqrt(2u) result, not a
    # bare 1/u pole. u := 1 - eta*sin(theta_i) (distance from criticality).
    eta_sup = 1.6
    n_i_sup = eta_sup * n_t
    u_vals = torch.tensor([10.0 ** (-k) for k in range(2, 9)], dtype=torch.float64)
    sin_i_sup = (1.0 - u_vals) / eta_sup
    cos_i_sup = torch.sqrt(1.0 - sin_i_sup ** 2)
    cos_t_sup = cos_theta_t(cos_i_sup, n_i_sup, n_t)
    det_sup = solid_angle_ratio(torch.tensor(n_i_sup), torch.tensor(n_t), cos_i_sup, cos_t_sup)
    log_u, log_det = torch.log(u_vals), torch.log(det_sup)
    slopes = (log_det[1:] - log_det[:-1]) / (log_u[1:] - log_u[:-1])
    final_slope = slopes[-1].item()
    slope_converging = bool((slopes[1:] - (-0.5)).abs().max() <= (slopes[:-1] - (-0.5)).abs().max())

    tol = 1e-9
    return [
        StructResult("A6", "symbolic: J_par(0) == J_perp(0) == eta exactly, general eta",
                     0.0 if collapse_diff == 0 else 1.0, 0.0,
                     0.0 if collapse_diff == 0 else 1.0, 0.0, collapse_diff == 0,
                     f"sympy.simplify(J_par(0) - J_perp(0)) = {collapse_diff}"),
        StructResult("A6", "actual code: J_par(0) == J_perp(0) == eta, across eta<1 and eta>1",
                     max_collapse_err, 0.0, max_collapse_err, tol, max_collapse_err < tol,
                     f"max relative |J_par(0)-J_perp(0)|/eta over eta in {etas}"),
        StructResult("A6", "eta=1: det identically 1 across full domain incl. theta_i->90 (no divergence)",
                     max_matched_dev, 0.0, max_matched_dev, 1e-6, max_matched_dev < 1e-6,
                     f"max|det-1| over theta_i up to 89.999 deg at matched media"),
        StructResult("A6", "eta<1: det->0 linearly as theta_i->90 (finite rate, not a divergence)",
                     final_rate_dev, 0.0, final_rate_dev, 1e-6, rate_monotone and final_rate_dev < 1e-6,
                     f"eta={eta_sub}: det/cos_i -> {predicted_rate:.6f} = eta^2/sqrt(1-eta^2), "
                     f"deviation shrinks monotonically, final (theta_i=89.9999 deg) rel dev {final_rate_dev:.2e}"),
        StructResult("A6", "eta>1: det diverges at theta_i->theta_c like u^(-1/2), matching A5's v~sqrt(2u)",
                     abs(final_slope - (-0.5)), -0.5, abs(final_slope - (-0.5)), 1e-3,
                     abs(final_slope - (-0.5)) < 1e-3 and slope_converging,
                     f"eta={eta_sup}: log-log slope of det vs u -> {final_slope:.6f} "
                     "(target -0.5, a sqrt pole, not a bare 1/u pole)"),
    ]


# ---------------------------------------------------------------------------
# A7: J_TIR collapse at eta=1
# Claim (ss7): J_TIR^s(v=0) = 4*eta and J_TIR^p(v=0) = 4*eta^3 both equal 4
#              when eta=1 (no interface, no TIR, polarizations must agree).
#
# Theorem 3 (Sec7): J_TIR^s(v)=4eta^3c^2/(eta*c+v)^2, J_TIR^p(v)=4eta^3c^2/(c+eta*v)^2.
# At v=0: J_TIR^s(0)=4eta^3c^2/(eta*c)^2=4eta, J_TIR^p(0)=4eta^3c^2/c^2=4eta^3 --
# both independent of c (matches Sec7's own "nonzero denominator at v=0" note).
# At eta=1 (matched media, s and p Fresnel coefficients are identical, so
# there is exactly one physical answer, not two): both collapse to 4.
# ---------------------------------------------------------------------------

def test_A7() -> list[StructResult]:
    import sympy as sp
    from src.snell_jacobian import tir_jacobian

    eta_s, c_s, v_s = sp.symbols("eta c v", positive=True)
    J_s = 4 * eta_s ** 3 * c_s ** 2 / (eta_s * c_s + v_s) ** 2
    J_p = 4 * eta_s ** 3 * c_s ** 2 / (c_s + eta_s * v_s) ** 2

    # Check 1: symbolic limit v->0, general eta,c -- must be 4*eta and 4*eta^3
    # respectively, with the c-dependence cancelling out completely.
    Js0 = sp.simplify(sp.limit(J_s, v_s, 0))
    Jp0 = sp.simplify(sp.limit(J_p, v_s, 0))
    s_diff = sp.simplify(Js0 - 4 * eta_s)
    p_diff = sp.simplify(Jp0 - 4 * eta_s ** 3)

    # Check 2: at eta=1, both v=0 limits collapse to exactly 4, general c.
    Js0_eta1 = sp.simplify(Js0.subs(eta_s, 1))
    Jp0_eta1 = sp.simplify(Jp0.subs(eta_s, 1))
    collapse_diff = sp.simplify(Js0_eta1 - Jp0_eta1)
    collapse_val_diff = sp.simplify(Js0_eta1 - 4)

    # Check 3: NOT vacuous -- away from eta=1, s and p limits genuinely
    # differ (4*eta != 4*eta^3 unless eta in {0,1,-1}).
    eta_away = sp.Rational(3, 2)
    away_diff = float(sp.simplify((4 * eta_away) - (4 * eta_away ** 3)))

    # Check 4: actual tir_jacobian() code, v=0, sweep of eta including 1.0,
    # confirms J_s(0)=4*eta, J_p(0)=4*eta^3 numerically, and that they agree
    # only at eta=1 among the tested values.
    etas = torch.tensor([0.5, 0.8, 1.0, 1.3, 2.0], dtype=torch.float64)
    cos_i = torch.full_like(etas, 0.6)
    v0 = torch.zeros_like(etas)
    J_s_code = tir_jacobian(v0, etas, torch.ones_like(etas), cos_i, polarization="s")
    J_p_code = tir_jacobian(v0, etas, torch.ones_like(etas), cos_i, polarization="p")
    max_s_err = (J_s_code - 4 * etas).abs().max().item()
    max_p_err = (J_p_code - 4 * etas ** 3).abs().max().item()
    eta1_idx = 2
    collapse_err_code = abs(J_s_code[eta1_idx].item() - J_p_code[eta1_idx].item())
    other_idx_diffs = [(J_s_code[i] - J_p_code[i]).abs().item() for i in range(len(etas)) if i != eta1_idx]

    tol = 1e-12
    return [
        StructResult("A7", "symbolic: J_TIR^s(0)=4*eta and J_TIR^p(0)=4*eta^3 exactly, general eta,c",
                     0.0 if (s_diff == 0 and p_diff == 0) else 1.0, 0.0,
                     0.0 if (s_diff == 0 and p_diff == 0) else 1.0, 0.0,
                     s_diff == 0 and p_diff == 0,
                     f"sympy.limit(J_s,v,0)-4eta = {s_diff}, sympy.limit(J_p,v,0)-4eta^3 = {p_diff} "
                     "(both independent of c, matching Sec7's nonzero-denominator note)"),
        StructResult("A7", "symbolic: at eta=1, J_TIR^s(0)=J_TIR^p(0)=4 exactly",
                     0.0 if (collapse_diff == 0 and collapse_val_diff == 0) else 1.0, 0.0,
                     0.0 if (collapse_diff == 0 and collapse_val_diff == 0) else 1.0, 0.0,
                     collapse_diff == 0 and collapse_val_diff == 0,
                     f"J_s(0)-J_p(0) at eta=1: {collapse_diff}; J_s(0)-4 at eta=1: {collapse_val_diff}"),
        StructResult("A7", "symbolic: NOT vacuous -- s,p limits genuinely differ away from eta=1",
                     abs(away_diff), None, None, 1e-9, abs(away_diff) > 1e-9,
                     f"4*eta - 4*eta^3 at eta=3/2: {away_diff:.4f} (nonzero -- collapse is specific to eta=1)"),
        StructResult("A7", "actual tir_jacobian() code: J_s(0)=4*eta, J_p(0)=4*eta^3 for all tested eta",
                     max(max_s_err, max_p_err), 0.0, max(max_s_err, max_p_err), tol,
                     max_s_err < tol and max_p_err < tol,
                     f"max|J_s(0)-4eta| = {max_s_err:.2e}, max|J_p(0)-4eta^3| = {max_p_err:.2e} over eta in {etas.tolist()}"),
        StructResult("A7", "actual tir_jacobian() code: s,p agree ONLY at eta=1 among tested values",
                     collapse_err_code, 0.0, collapse_err_code, tol,
                     collapse_err_code < tol and all(d > 1e-3 for d in other_idx_diffs),
                     f"|J_s(0)-J_p(0)| at eta=1: {collapse_err_code:.2e}; at other etas: {other_idx_diffs}"),
    ]


# ---------------------------------------------------------------------------
# A8: Brewster angle vs v=0 cancellation
# Claim (ss7): T_p(theta_Brewster)=1 (r_p=0) is a different special angle
#              from v=0; symbolic confirmation they are distinct.
# ---------------------------------------------------------------------------

def test_A8() -> StructResult:
    raise NotImplementedError("A8")


# ---------------------------------------------------------------------------
# A9: a_bar -> 0 as sigma_f -> 0
# Claim (ss9): a_bar = sqrt(2*pi)*alpha0*sigma_f; limit as sigma_f->0 must
#              vanish at fixed alpha0 -- no spurious finite floor.
# ---------------------------------------------------------------------------

def test_A9() -> StructResult:
    raise NotImplementedError("A9")


# ---------------------------------------------------------------------------
# A10: lambda*(A,B) derivatives as kappa-A -> 0
# Claim (ss13): lambda* -> inf, dlambda*/dA and dlambda*/dB diverge at the
#               *same rate* relative to lambda*: dlambda*/dA * B / lambda*^3 -> 1/2.
# ---------------------------------------------------------------------------

def test_A10() -> StructResult:
    raise NotImplementedError("A10")


# ---------------------------------------------------------------------------
# A11: lambda_ex,j invariance for k species (symbolic)
# Claim (ss14): d/dlambda_ex,j of sum_j term_j = 0 independent of k,
#               provable as one-line linearity for symbolic k.
# ---------------------------------------------------------------------------

def test_A11() -> StructResult:
    raise NotImplementedError("A11")


# ---------------------------------------------------------------------------
# A12: H_wp necessary? (OPEN THEORY QUESTION)
# Task: construct or rule out a counterexample where H_wp fails but rho(T)<1.
#       Would mean H_wp is overly conservative.
# ---------------------------------------------------------------------------

def test_A12() -> StructResult:
    raise NotImplementedError("A12 -- open theory question, resolve analytically first")


# ---------------------------------------------------------------------------
# A13: Discrete event-type score function (OPEN THEORY QUESTION)
# Task: determine analytically whether the event-type selection probability
#       (elastic / fluorescent / Raman) depends on theta, and if so whether
#       a missing REINFORCE-style term exists in the gradient estimator.
# ---------------------------------------------------------------------------

def test_A13() -> StructResult:
    raise NotImplementedError("A13 -- open theory question, resolve analytically first")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL = [
    test_A1, test_A2, test_A3, test_A4, test_A5, test_A6, test_A7,
    test_A8, test_A9, test_A10, test_A11, test_A12, test_A13,
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
        except OpenTheoreticalQuestion as e:
            print(f"OPEN  {fn.__name__}: {e}")
        except NotImplementedError as e:
            print(f"SKIP  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e}", file=sys.stderr)
    rep.print_all()


if __name__ == "__main__":
    main()
