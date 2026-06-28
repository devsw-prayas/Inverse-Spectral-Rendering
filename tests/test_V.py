"""V-series: verification against the Python forward oracle.

V1-V9 run on the Python oracle (Phase 1).
V10-V12 require the C++ path tracer (Phase 2) -- stubbed here for completeness.

Run:
    conda activate Spectral
    python -m pytest tests/test_V.py -v
  or
    python -m tests.test_V

Tests
-----
V1   TIR energy conservation     -- furnace test: closed cavity including TIR interface
V2   lambda_ex invariance        -- flat-illuminant fluorescent scene, two lambda_ex, same seed
V3   Correct vs wrong adjoint    -- single-bounce scene, FD vs both adjoint forms, all 5 params
V4   TIR + moving boundary full  -- two-bounce scene combining TIR-adjacent and lambda* crossing
V5   Substrate confound, grad.   -- gradient-descent recovery variance matches SVD prediction
V6   Inverse-crime check         -- ground truth from deliberately mismatched forward model
V7   Three-estimator bias test   -- Zeltner taxonomy, scene with K_x and TIR-adjacent sampling
V8   Near-rational FSR aliasing  -- d swept so FSR/delta_lambda crosses near-integer ratios
V9   C10 at full scene           -- two-bounce fluor-behind-glass with lambda* in emission band

--- Phase 2 (require C++ path tracer) ---
V10  C++ vs Python oracle        -- identical scene: C++ must match oracle to FD precision
V11  MIS balance correctness     -- multi-strategy scene: weights sum correctly, estimator unbiased
V12  Suite sensitivity check     -- deliberately revert to wrong form, suite must flag it
"""
from __future__ import annotations

import sys
import torch
torch.set_default_dtype(torch.float64)

from tests.harness import Reporter, StructResult, GradResult


# ---------------------------------------------------------------------------
# V1: TIR energy conservation (furnace test)
# Setup: closed uniform-temperature cavity with a TIR-adjacent interface.
# Claim: L_out = L_in everywhere at equilibrium.
#        Leak/pooling *near* the critical angle isolates a J_TIR-specific bug.
# ---------------------------------------------------------------------------

def test_V1() -> StructResult:
    raise NotImplementedError("V1")


# ---------------------------------------------------------------------------
# V2: lambda_ex exact invariance (cheapest, strongest falsifier -- run first)
# Setup: flat-illuminant fluorescent scene; render at two different lambda_ex
#        (or vectors), same RNG seed.
# Claim: two renders agree within pure MC noise, zero systematic residual.
# ---------------------------------------------------------------------------

def test_V2() -> StructResult:
    raise NotImplementedError("V2")


# ---------------------------------------------------------------------------
# V3: Correct vs wrong adjoint gradient (single-bounce)
# Setup: single-bounce fluorescent scene; gradient w.r.t. each of the 5
#        non-degenerate parameters.
# Claim: FD on re-rendered image agrees with correct adjoint; wrong adjoint
#        must fail on *every* parameter, including elastic ones.
# ---------------------------------------------------------------------------

def test_V3() -> list[GradResult]:
    raise NotImplementedError("V3")


# ---------------------------------------------------------------------------
# V4: Naive vs corrected gradient near criticality + moving lambda* combined
# Setup: two-bounce scene combining TIR-adjacent sampling with emission band
#        straddling lambda*.
# Claim: FD oracle on full re-render -- tests variance behavior under MC
#        sampling, which T12 cannot test without a real estimator.
# ---------------------------------------------------------------------------

def test_V4() -> GradResult:
    raise NotImplementedError("V4")


# ---------------------------------------------------------------------------
# V5: Substrate confound and B-bottleneck in real gradient-based recovery
# Setup: render ground truth, recover via gradient descent, multiple random inits.
# Claim: recovery variance across seeds rank-orders parameters the same way
#        the Jacobian SVD predicts.
# ---------------------------------------------------------------------------

def test_V5() -> StructResult:
    raise NotImplementedError("V5")


# ---------------------------------------------------------------------------
# V6: Inverse-crime check (cross-cutting)
# Setup: any recovery test (V5), but generate ground truth with a deliberately
#        mismatched forward model.
# Claim: recovery quality must degrade under mismatch -- else the "successful"
#        recovery was partly circular.
# ---------------------------------------------------------------------------

def test_V6() -> StructResult:
    raise NotImplementedError("V6")


# ---------------------------------------------------------------------------
# V7: Three-estimator bias test (Zeltner taxonomy)
# Setup: scene exercising both K_x and TIR-adjacent sampling.
# Claim: all three estimators vs V3's FD oracle -- catch an estimator that is
#        unbiased in expectation but with variance blowing up near the singular
#        manifold.
# ---------------------------------------------------------------------------

def test_V7() -> list[StructResult]:
    raise NotImplementedError("V7")


# ---------------------------------------------------------------------------
# V8: Near-rational FSR aliasing (rendered)
# Setup: thin-film scene, d swept so FSR/delta_lambda_sample crosses near-integer
#        ratios, actually rendered.
# Claim: oversampled-lambda reference render vs production sampling rate;
#        any spurious beat pattern in the rendered image is a failure.
# ---------------------------------------------------------------------------

def test_V8() -> StructResult:
    raise NotImplementedError("V8")


# ---------------------------------------------------------------------------
# V9: C10 at full scene complexity
# Setup: full two-bounce scene: fluorescent emission inside dispersive medium,
#        escaping past TIR-bounded interface, lambda* swept through emission band.
# Claim: FD on full re-render at A +/- eps -- the explicit named falsifier
#        from the project's own list.
# ---------------------------------------------------------------------------

def test_V9() -> GradResult:
    raise NotImplementedError("V9")


# ---------------------------------------------------------------------------
# V10-V12: Phase 2 only (require C++ path tracer)
# ---------------------------------------------------------------------------

def test_V10() -> StructResult:
    raise NotImplementedError("V10 -- Phase 2: requires C++ path tracer")


def test_V11() -> StructResult:
    raise NotImplementedError("V11 -- Phase 2: requires C++ path tracer")


def test_V12() -> StructResult:
    raise NotImplementedError("V12 -- Phase 2: requires C++ path tracer")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

PHASE1 = [test_V1, test_V2, test_V3, test_V4, test_V5, test_V6, test_V7, test_V8, test_V9]
PHASE2 = [test_V10, test_V11, test_V12]
ALL    = PHASE1 + PHASE2


def main(phase2: bool = False) -> None:
    tests = ALL if phase2 else PHASE1
    rep = Reporter()
    for fn in tests:
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", action="store_true",
                        help="Also run Phase 2 tests (require C++ tracer)")
    args = parser.parse_args()
    main(phase2=args.phase2)
