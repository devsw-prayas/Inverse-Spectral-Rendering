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
# ---------------------------------------------------------------------------

def test_A2() -> StructResult:
    raise NotImplementedError("A2")


# ---------------------------------------------------------------------------
# A3: ||K_x|| = ||e||_2 * ||a||_2, equality case
# Claim (ss2): Cauchy-Schwarz equality is achieved at f = a/||a||_2;
#              closed-form identity, not just an inequality.
# ---------------------------------------------------------------------------

def test_A3() -> StructResult:
    raise NotImplementedError("A3")


# ---------------------------------------------------------------------------
# A4: K_x* adjoint degenerate
# Claim (ss4): T*=T exactly when e=a (zero Stokes shift).
#              Wrong adjoint must become numerically correct *only* in this limit.
# ---------------------------------------------------------------------------

def test_A4() -> StructResult:
    raise NotImplementedError("A4")


# ---------------------------------------------------------------------------
# A5: v ~ sqrt(2u) branch order
# Claim (ss5): Puiseux series of v(u) about u=0 has leading order u^(1/2)
#              with coefficient sqrt(2); no u^1 or u^0 term at leading order.
# ---------------------------------------------------------------------------

def test_A5() -> StructResult:
    raise NotImplementedError("A5")


# ---------------------------------------------------------------------------
# A6: J_||, J_perp, det limits
# Claim (ss6): At theta_i=0: J_||=J_perp=eta exactly (formulas collapse).
#              At theta_i->90 deg: confirm divergence *rate* of det.
# ---------------------------------------------------------------------------

def test_A6() -> StructResult:
    raise NotImplementedError("A6")


# ---------------------------------------------------------------------------
# A7: J_TIR collapse at eta=1
# Claim (ss7): J_TIR^s(v=0) = 4*eta and J_TIR^p(v=0) = 4*eta^3 both equal 4
#              when eta=1 (no interface, no TIR, polarizations must agree).
# ---------------------------------------------------------------------------

def test_A7() -> StructResult:
    raise NotImplementedError("A7")


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
