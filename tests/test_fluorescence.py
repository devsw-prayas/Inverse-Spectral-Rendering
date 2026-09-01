"""Fluorescence: rank-1 kernel unit checks, plus the cross-check against the Vera
GPU renderer's fluorescent Lambertian (Mojzik-Fichet-Wilkie 2018, DBR track).

The kernel tests are pure oracle. The last test renders with Vera's `_vera`
extension (from ../../Prototype/DBR-x-ReSTIR/build/<config>/) and compares the
fluor/elastic XYZ ratio to the same ratio from `kernel_fluorescence`; it skips if
that build is not present. Run manually:

    conda activate Spectral
    python -m pytest tests/test_fluorescence.py -v

Tests
-----
F1  rank-1 Phi is Stokes / lower-triangular   -- energy only moves to longer lambda
F2  rank-1 bounce conserves energy            -- re-emits exactly QY of what it absorbed
F3  Vera fluor/elastic XYZ ratio vs oracle    -- single bounce, rel err < 5%
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
torch.set_default_dtype(torch.float64)

from src.kernels import kernel_fluorescence
from src.spectral_grid import make_grid

LAMBDA_MIN, LAMBDA_MAX = 360.0, 700.0


# ---------------------------------------------------------------------------
# F1: rank-1 Phi is Stokes / lower-triangular
# Oracle convention T[i,j] = QY * e(lambda_i) * a(lambda_j) * w_j, with e centred
# at lambda_em, a at lambda_ex, lambda_em > lambda_ex. Row i = output, column j =
# input, so energy to longer lambda fills the lower triangle; the strict upper
# triangle (output well below input) must be ~0.
# ---------------------------------------------------------------------------

def test_rank1_kernel_is_stokes_and_lower_triangular():
    grid = make_grid(lam_min=LAMBDA_MIN, lam_max=LAMBDA_MAX)
    T = kernel_fluorescence(grid.lam, lam_ex=380.0, lam_em=520.0, sigma_f=15.0,
                            weights=grid.weights, quantum_yield=0.9)
    assert T.shape == (grid.N, grid.N)
    lam = grid.lam
    upper = T[lam.unsqueeze(1) < lam.unsqueeze(0) - 3 * 15.0]  # output well below input
    assert upper.abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# F2: rank-1 bounce conserves energy
# With e normalised to sum_i e_i w_i = 1, applying T to a field L gives
#   re-emitted = sum_i w_i (T @ L)_i = QY * sum_j a_j w_j L_j = QY * absorbed,
# so the bounce is strictly lossy for QY < 1 and never amplifies.
# ---------------------------------------------------------------------------

def test_rank1_kernel_energy_conserving():
    grid = make_grid(lam_min=LAMBDA_MIN, lam_max=LAMBDA_MAX)
    qy = 0.75
    a = torch.exp(-0.5 * ((grid.lam - 400.0) / 20.0) ** 2)  # peak-1 absorption
    T = kernel_fluorescence(grid.lam, lam_ex=400.0, lam_em=560.0, sigma_f=20.0,
                            weights=grid.weights, quantum_yield=qy)
    L = torch.ones_like(grid.lam)
    absorbed = (a * grid.weights * L).sum()
    reemitted = (grid.weights * (T @ L)).sum()
    assert reemitted.item() <= absorbed.item() + 1e-9
    assert reemitted.item() == pytest.approx(qy * absorbed.item(), rel=1e-9)


# ---------------------------------------------------------------------------
# F3: Vera fluor/elastic XYZ ratio vs oracle
# One fluorescent bounce (max_bounces=2), rendered elastic and fluorescent with
# the same albedo. The XYZ ratio fluor/elastic cancels geometry / pi / exposure /
# CIE-norm, so it can be compared directly to the same ratio from the rank-1
# operator on a flat unit illuminant.
# ---------------------------------------------------------------------------

def _load_vera():
    # .../Graphics Programming/R&D/Inverse Spectral Rendering/tests/<this>
    root = Path(__file__).resolve().parents[3] / "Prototype" / "DBR-x-ReSTIR" / "build"
    for cfg in ("Release", "Debug"):
        p = root / cfg
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        import _vera  # noqa
        return _vera
    except ImportError:
        return None


# analytic CIE 1931 fit, verbatim from Vera's src/HWSS/Public/CIE.h
def _cie_xyz(lam):
    lam = np.asarray(lam, dtype=np.float64)
    def g(mu, s1, s2):
        s = np.where(lam < mu, s1, s2)
        return np.exp(-0.5 * ((lam - mu) / s) ** 2)
    x = 1.056 * g(599.8, 37.9, 31.0) + 0.362 * g(442.0, 16.0, 26.7) - 0.065 * g(501.1, 20.4, 26.2)
    y = 0.821 * g(568.8, 46.9, 40.5) + 0.286 * g(530.9, 16.3, 31.1)
    z = 1.217 * g(437.0, 11.8, 36.0) + 0.681 * g(459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def test_vera_single_bounce_ratio_matches_oracle():
    _vera = _load_vera()
    if _vera is None:
        pytest.skip("Vera _vera extension not built at ../../Prototype/DBR-x-ReSTIR/build")

    R, QY = 0.5, 0.8
    lam_ex, lam_em, sigma = 440.0, 610.0, 16.0

    def render_wall(mat_fn):
        scene = _vera.Scene()
        wall = scene.add_material(mat_fn(scene))
        emit = scene.add_material(_vera.emissive([1.0, 1.0, 1.0], 8.0))
        s = 1.0
        scene.add_quad([-s, -s, -1.0], [s, -s, -1.0], [s, s, -1.0], [-s, s, -1.0],
                       [0.0, 0.0, 1.0], wall)
        e = 2.0
        scene.add_quad([-e, -e, 1.6], [e, -e, 1.6], [e, e, 1.6], [-e, e, 1.6],
                       [0.0, 0.0, -1.0], emit)
        cam = _vera.camera_look_at(eye=[0.0, 0.0, 1.2], target=[0.0, 0.0, -1.0],
                                   up=[0.0, 1.0, 0.0], fov_y_deg=30.0, width=48, height=48)
        img = _vera.render_xyz(scene, cam, spp=256, max_bounces=2)
        return np.asarray(img, np.float64)[16:32, 16:32].reshape(-1, 3).mean(axis=0)

    xyz_elastic = render_wall(lambda sc: _vera.lambertian([R, R, R]))
    xyz_fluor = render_wall(lambda sc: _vera.fluorescent_lambertian([R, R, R], lam_ex, lam_em, sigma, QY))
    vera_ratio = xyz_fluor / xyz_elastic

    grid = make_grid(lam_min=LAMBDA_MIN, lam_max=LAMBDA_MAX)
    lam, w = grid.lam.numpy(), grid.weights.numpy()
    L_flat = np.ones_like(lam)
    Kfl = kernel_fluorescence(grid.lam, lam_ex, lam_em, sigma, grid.weights, QY).numpy()
    cie = _cie_xyz(lam)
    proj = lambda L: (L[:, None] * cie * w[:, None]).sum(axis=0)
    oracle_ratio = proj(R * L_flat + Kfl @ L_flat) / proj(R * L_flat)

    rel = np.abs(vera_ratio - oracle_ratio) / np.abs(oracle_ratio)
    assert rel.max() < 0.05, f"vera {vera_ratio} vs oracle {oracle_ratio} (rel {rel})"
