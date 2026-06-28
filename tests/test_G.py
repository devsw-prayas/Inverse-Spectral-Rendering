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

def test_G1() -> StructResult:
    raise NotImplementedError("G1")


# ---------------------------------------------------------------------------
# G2: theta_i sweep Jacobians
# X-axis: theta_i, 0 -> grazing
# Y-axis: J_||, J_perp, det -- several eta values
# Predicted shape: J_perp flat; J_||, det diverge together near critical theta;
#                  all curves touch at theta_i=0
# Failure: curves don't touch at 0; J_perp not flat
# ---------------------------------------------------------------------------

def test_G2() -> StructResult:
    raise NotImplementedError("G2")


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
            rep.add(fn())
        except NotImplementedError as e:
            print(f"SKIP  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e}", file=sys.stderr)
    rep.print_all()


if __name__ == "__main__":
    main()
