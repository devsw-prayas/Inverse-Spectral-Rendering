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
    raise NotImplementedError("T0")


# ---------------------------------------------------------------------------
# T1: M_R non-compact
# Claim (ss2): ||M_R e_n|| -> R(lambda0) must hold even with lambda0 near the
#              domain edge (lambda_min / lambda_max), not just mid-spectrum.
# Setup: disjoint-bump construction near lambda_min/lambda_max boundary.
# ---------------------------------------------------------------------------

def test_T1() -> StructResult:
    raise NotImplementedError("T1")


# ---------------------------------------------------------------------------
# T2: Column-sum-only bound fails to control L2 norm
# Claim (ss3): toy matrix with column sums exactly 1 (not 0.9) has operator
#              2-norm that blows up arbitrarily as off-diagonal concentration
#              increases -- confirms failure isn't an artifact of the 0.9 margin.
# ---------------------------------------------------------------------------

def test_T2() -> StructResult:
    raise NotImplementedError("T2")


# ---------------------------------------------------------------------------
# T4: det = eta^2 * c/v
# Claim (ss6): Snell Jacobian determinant formula holds for both eta<1 and eta>1.
#              Guards against hidden eta>1-only assumption.
# ---------------------------------------------------------------------------

def test_T4() -> StructResult:
    raise NotImplementedError("T4")


# ---------------------------------------------------------------------------
# T5: J_TIR finite at v=0
# Claim (ss7): J_TIR^s,p(v) converges to 4*eta / 4*eta^3 on the propagating
#              side. Formula must NOT be evaluated past v=0 (evanescent side)
#              without an explicit domain guard.
# ---------------------------------------------------------------------------

def test_T5() -> StructResult:
    raise NotImplementedError("T5")


# ---------------------------------------------------------------------------
# T6: Substrate confound rank / conditioning
# Claim (ss8): As n_substrate -> n_film (film becomes invisible), conditioning
#              degrades further -- film thickness becomes unobservable with
#              zero index contrast.
# ---------------------------------------------------------------------------

def test_T6() -> StructResult:
    raise NotImplementedError("T6")


# ---------------------------------------------------------------------------
# T9: Rank drops below 5 in joint degenerate limit
# Claim (ss10): As sigma_f->0 AND d->0 simultaneously, rank must actually drop
#               *below 5*, not just worsen -- else the "B is bottleneck among 5"
#               framing hides a deeper exact degeneracy.
# ---------------------------------------------------------------------------

def test_T9() -> StructResult:
    raise NotImplementedError("T9")


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
