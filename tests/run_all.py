"""Tier 1-2 gradient and structural test suite.

Run:
    conda activate Spectral
    python -m tests.run_all

Outputs:
    - Console table (Section 3 gradient validation + structural invariants)
    - results/validation_table.csv / .tex
    - results/structural_table.csv

Tests
-----
A0  dloss/dd          single_bounce_flat   analytic fabry_airy_dR_dd         (3-way)
A1  dloss/dA          single_bounce_flat   FD vs autograd only -- AN uses      (2-way)
                                           torch.autograd.jacobian internally
A2  dloss/dB          single_bounce_flat   same as A1                         (2-way)
B7  dloss/dlam_em     two_bounce           analytic fluorescence_dK_dlam_em   (3-way)
B8  dloss/dlam_ex     two_bounce           analytic fluorescence_dK_dlam_ex   (3-way)
B9  dloss/dsigma_f    two_bounce           analytic fluorescence_dK_dsigma_f  (3-way)
C0  Snell Jacobian     Lemma 1 vs FD Jacobian
C1  TIR-safe F(v)      Theorem 3: F(v)=J(v)*|dcos_i/dv|, glass->air, N=10
C2  Jacobian compose   compose_jacobians == manual bmm
C3  Adjoint residual   ||(I-T^T)G - g|| / ||g||
C4  dloss/dd           two_bounce   (thin-film + fluorescence)
C5  dloss/dd           per-lam cos_i -> dense kernel_thinfilm (dispersive)    (2-way: FD+AG)
C6  Neumann convergence ||L_32 - L_exact|| / ||L_exact||
C7  Spectral radius     rho(K_TF) and rho(K_FL) separately, both < 1
C8  Half-attached bias  measured vs analytic missing-term formula
C9  Dense vs diag       kernel_thinfilm: scalar cos_i -> diagonal matrix

Known gaps (deferred to Phase 2 / C++ tracer)
----------------------------------------------
V_theta / reparameterization (Eq. 3, Section 2-3): `propagate_velocity` and
`compose_jacobians` appear nowhere in a validated gradient test. The paper's
boundary term grad_x f * V_theta * J_theta is UNVALIDATED until a C++ path-tracer
can exercise a moving-domain path integral. Flag explicitly in Phase 2 plan.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on the path when run as a module or directly.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch
torch.set_default_dtype(torch.float64)

from src.spectral_grid import make_grid
from src.sensor import Sensor, hyperspectral_fx10
from src.scenes import single_bounce_flat, two_bounce, d65_on_grid
from src.forward import neumann_forward, fredholm_solve_exact
from src.kernels import (
    kernel_thinfilm, kernel_fluorescence,
    fabry_airy_dR_dd, fabry_airy_dR_dA, fabry_airy_dR_dB,
)
from src.snell_jacobian import (
    snell_jacobian, snell_jacobian_tir_safe,
    refracted_direction, compose_jacobians,
)
from src.cauchy_ior import n_cauchy, cos_theta_t, is_tir
from src.gradient import (
    fd_gradient, neumann_adjoint, kernel_gradient,
    fluorescence_dK_dlam_em, fluorescence_dK_dlam_ex, fluorescence_dK_dsigma_f,
    kernel_fluorescence_half_attached,
)
from tests.harness import GradResult, StructResult, Reporter, run_three_way, _rel

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Shared scene configuration
# ---------------------------------------------------------------------------

GRID   = make_grid()
SENSOR = hyperspectral_fx10(GRID.lam)
MAX_D  = 32

D_NM   = 120.0
A_CAU  = 1.45
B_CAU  = 8000.0
COS_I  = 0.8

LAM_EX = 450.0
LAM_EM = 530.0
SIG_F  = 20.0
QY     = 0.9

_L_E = d65_on_grid(GRID)          # shared source spectrum, no grad


def _loss(image: torch.Tensor) -> torch.Tensor:
    return image.sum()


def _g() -> torch.Tensor:
    return SENSOR.S.T @ torch.ones(SENSOR.S.shape[0], dtype=torch.float64)


# ---------------------------------------------------------------------------
# A-series: thin-film parameter gradients (single_bounce_flat)
# ---------------------------------------------------------------------------

def test_A0() -> GradResult:
    d = torch.tensor(D_NM, requires_grad=True)

    def fn():
        r = single_bounce_flat(GRID, SENSOR, d=d, A=A_CAU, B=B_CAU,
                               cos_i=COS_I, max_depth=MAX_D, L_e=_L_E)
        return _loss(r.image)

    with torch.no_grad():
        K0     = kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
        dR_dd  = fabry_airy_dR_dd(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
        an     = kernel_gradient(K0, dR_dd, _L_E, _g(), MAX_D).item()

    return run_three_way(fn, d, an, "A0", "single_bounce_flat", "d")


def test_A1() -> GradResult:
    """∂loss/∂A: FD vs autograd (two-way only).

    fabry_airy_dR_dA calls torch.autograd.functional.jacobian internally, so the
    'analytic' column is the same computation as 'autograd'. FD is the only
    independent oracle. Both AN and AG will agree to float64 machine precision.
    """
    A = torch.tensor(A_CAU, requires_grad=True)

    def fn():
        r = single_bounce_flat(GRID, SENSOR, d=D_NM, A=A, B=B_CAU,
                               cos_i=COS_I, max_depth=MAX_D, L_e=_L_E)
        return _loss(r.image)

    with torch.no_grad():
        K0     = kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
        dR_dA  = fabry_airy_dR_dA(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
        an     = kernel_gradient(K0, dR_dA, _L_E, _g(), MAX_D).item()

    # scene label notes two-way nature: AN≡AG (both use autograd through fabry_airy_R)
    return run_three_way(fn, A, an, "A1", "sbf[AN=AG,FD-only]", "A")


def test_A2() -> GradResult:
    """∂loss/∂B: FD vs autograd (two-way only).

    Same as A1: fabry_airy_dR_dB uses torch.autograd.functional.jacobian
    internally; AN and AG are the same computation. FD is the only independent
    oracle.
    """
    B = torch.tensor(B_CAU, requires_grad=True)

    def fn():
        r = single_bounce_flat(GRID, SENSOR, d=D_NM, A=A_CAU, B=B,
                               cos_i=COS_I, max_depth=MAX_D, L_e=_L_E)
        return _loss(r.image)

    with torch.no_grad():
        K0     = kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
        dR_dB  = fabry_airy_dR_dB(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
        an     = kernel_gradient(K0, dR_dB, _L_E, _g(), MAX_D).item()

    # scene label notes two-way nature: AN≡AG (both use autograd through fabry_airy_R)
    return run_three_way(fn, B, an, "A2", "sbf[AN=AG,FD-only]", "B")


# ---------------------------------------------------------------------------
# B-series: fluorescence parameter gradients (two_bounce)
# ---------------------------------------------------------------------------

def test_B7() -> GradResult:
    lam_em = torch.tensor(LAM_EM, requires_grad=True)

    def fn():
        r = two_bounce(GRID, SENSOR, d=D_NM, A=A_CAU, B=B_CAU, cos_i=COS_I,
                       lam_ex=LAM_EX, lam_em=lam_em, sigma_f=SIG_F,
                       max_depth=MAX_D, quantum_yield=QY, L_e=_L_E)
        return _loss(r.image)

    with torch.no_grad():
        K0     = (kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
                  + kernel_fluorescence(GRID.lam, LAM_EX, LAM_EM, SIG_F,
                                        GRID.weights, QY))
        dK     = fluorescence_dK_dlam_em(GRID.lam, LAM_EX, LAM_EM, SIG_F,
                                          GRID.weights, QY)
        an     = kernel_gradient(K0, dK, _L_E, _g(), MAX_D).item()

    return run_three_way(fn, lam_em, an, "B7", "two_bounce", "lam_em")


def test_B8() -> GradResult:
    lam_ex = torch.tensor(LAM_EX, requires_grad=True)

    def fn():
        r = two_bounce(GRID, SENSOR, d=D_NM, A=A_CAU, B=B_CAU, cos_i=COS_I,
                       lam_ex=lam_ex, lam_em=LAM_EM, sigma_f=SIG_F,
                       max_depth=MAX_D, quantum_yield=QY, L_e=_L_E)
        return _loss(r.image)

    with torch.no_grad():
        K0     = (kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
                  + kernel_fluorescence(GRID.lam, LAM_EX, LAM_EM, SIG_F,
                                        GRID.weights, QY))
        dK     = fluorescence_dK_dlam_ex(GRID.lam, LAM_EX, LAM_EM, SIG_F,
                                          GRID.weights, QY)
        an     = kernel_gradient(K0, dK, _L_E, _g(), MAX_D).item()

    return run_three_way(fn, lam_ex, an, "B8", "two_bounce", "lam_ex")


def test_B9() -> GradResult:
    sigma_f = torch.tensor(SIG_F, requires_grad=True)

    def fn():
        r = two_bounce(GRID, SENSOR, d=D_NM, A=A_CAU, B=B_CAU, cos_i=COS_I,
                       lam_ex=LAM_EX, lam_em=LAM_EM, sigma_f=sigma_f,
                       max_depth=MAX_D, quantum_yield=QY, L_e=_L_E)
        return _loss(r.image)

    with torch.no_grad():
        K0     = (kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
                  + kernel_fluorescence(GRID.lam, LAM_EX, LAM_EM, SIG_F,
                                        GRID.weights, QY))
        dK     = fluorescence_dK_dsigma_f(GRID.lam, LAM_EX, LAM_EM, SIG_F,
                                           GRID.weights, QY)
        an     = kernel_gradient(K0, dK, _L_E, _g(), MAX_D).item()

    return run_three_way(fn, sigma_f, an, "B9", "two_bounce", "sigma_f")


# ---------------------------------------------------------------------------
# C-series: structural invariants
# ---------------------------------------------------------------------------

def test_C0() -> StructResult:
    """Snell Jacobian (Lemma 1): analytic J vs FD ∂ω_t/∂ω_i."""
    N = 8
    lam_sub = GRID.lam[:N]
    n_i = torch.ones(N, dtype=torch.float64)
    n_t = n_cauchy(lam_sub, A_CAU, B_CAU)
    cos_i_vec = torch.full((N,), COS_I, dtype=torch.float64)
    cos_t_vec = cos_theta_t(COS_I, 1.0, n_t)
    tir = is_tir(COS_I, 1.0, n_t)

    # n_hat points toward incident medium (+z when ray travels in -z direction)
    n_hat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    sin_i   = (1.0 - COS_I ** 2) ** 0.5
    omega_i = torch.tensor([sin_i, 0.0, -COS_I], dtype=torch.float64)

    J_an = snell_jacobian(n_i, n_t, cos_i_vec, cos_t_vec, n_hat)  # (N,3,3)

    # FD Jacobian: per-wavelength, recompute cos_i and cos_t from perturbed omega_i.
    # Lemma 1 treats omega_i as a free vector (not constrained to unit sphere), with
    # cos_i = -omega_i · n_hat, so ∂cos_i/∂omega_i = -n_hat (enters the formula).
    eps = 1e-5
    J_fd = torch.zeros(N, 3, 3, dtype=torch.float64)
    for k in range(N):
        if tir[k]:
            continue
        nt_k = n_t[k]
        for j in range(3):
            e = torch.zeros(3, dtype=torch.float64); e[j] = eps
            for sign, dest_j in [(+1, 'p'), (-1, 'm')]:
                oi = omega_i + sign * e
                ci = -(oi * n_hat).sum()            # cos_i recomputed from perturbed dir
                ct = cos_theta_t(ci, 1.0, nt_k)
                ot = refracted_direction(oi, ci, ct, 1.0, nt_k, n_hat)
                if sign == 1:
                    ot_p = ot
                else:
                    ot_m = ot
            J_fd[k, :, j] = (ot_p - ot_m) / (2.0 * eps)

    # Skip TIR wavelengths (J diverges there)
    mask = ~tir
    rel = ((J_an[mask] - J_fd[mask]).abs()
           / J_fd[mask].abs().clamp(min=1e-30)).max().item()

    return StructResult(
        test_id="C0", quantity="max|J_an - J_fd| / |J_fd|",
        measured=rel, expected=0.0, rel_err=rel, tol=1e-6,
        passed=rel < 1e-6,
        note=f"Lemma 1; {mask.sum().item()}/{N} non-TIR wavelengths",
    )


def test_C1() -> StructResult:
    """TIR-safe F(v) = J(v)·|∂cosθ_i/∂v|: numerical verification (glass→air).

    Uses n_i=1.5 > n_t=1.0 so TIR is geometrically possible. Verifies that the
    combined factor F from snell_jacobian_tir_safe matches J(v) × |∂cos_i/∂v|
    element-wise over v ∈ [0.05, 0.95]. This test would have caught the η bug
    (β = n_i/n_t ≈ 1.5 instead of β = 1), which gave max rel-err ≈ 0.33.

    The old C1 used n_i=air < n_t=glass (never TIR) and only tested finiteness
    at v=0 — both trivially pass even with the wrong formula.
    """
    N = 10
    n_i  = torch.full((N,), 1.5, dtype=torch.float64)   # glass
    n_t  = torch.full((N,), 1.0, dtype=torch.float64)   # air
    v    = torch.linspace(0.05, 0.95, N, dtype=torch.float64)
    n_hat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    F_an  = snell_jacobian_tir_safe(v, n_i, n_t, n_hat)  # (N,3,3)

    # Reference: J(v) × |∂cosθ_i/∂v| (FD derivative)
    cos_i_v = torch.sqrt((n_i**2 - n_t**2 * (1.0 - v**2)).clamp(min=0.0)) / n_i
    J_v     = snell_jacobian(n_i, n_t, cos_i_v, v, n_hat)  # (N,3,3)

    eps = 1e-5
    cos_i_p = torch.sqrt((n_i**2 - n_t**2 * (1.0 - (v + eps)**2)).clamp(min=0.0)) / n_i
    cos_i_m = torch.sqrt((n_i**2 - n_t**2 * (1.0 - (v - eps)**2)).clamp(min=0.0)) / n_i
    dcos_dv = (cos_i_p - cos_i_m) / (2.0 * eps)          # (N,)

    F_ref = J_v * dcos_dv.abs().view(N, 1, 1)             # (N,3,3)

    rel = ((F_an - F_ref).abs() / F_ref.abs().clamp(min=1e-10)).max().item()

    return StructResult(
        test_id="C1", quantity="max|F_tir_safe - J*|dcos_i/dv|| / |ref|",
        measured=rel, expected=0.0, rel_err=rel, tol=1e-5,
        passed=rel < 1e-5,
        note=f"glass(1.5)->air(1.0), N={N}, v in [0.05,0.95]; catches eta bug",
    )


def test_C2() -> StructResult:
    """Jacobian composition: compose_jacobians([J1,J2]) == bmm(J2, J1)."""
    N = 16
    lam_sub = GRID.lam[:N]
    n1 = n_cauchy(lam_sub, A_CAU, B_CAU)
    n2 = n_cauchy(lam_sub, 1.35, 5000.0)
    n_air = torch.ones(N, dtype=torch.float64)
    cos_i1 = torch.full((N,), COS_I)
    cos_t1 = cos_theta_t(COS_I, 1.0, n1)
    cos_t2 = cos_theta_t(cos_t1, n1, n2)
    n_hat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    J1 = snell_jacobian(n_air, n1, cos_i1, cos_t1, n_hat)
    J2 = snell_jacobian(n1, n2, cos_t1, cos_t2, n_hat)

    composed   = compose_jacobians([J1, J2])
    manual_bmm = torch.bmm(J2, J1)

    rel = ((composed - manual_bmm).abs()
           / manual_bmm.abs().clamp(min=1e-30)).max().item()

    return StructResult(
        test_id="C2", quantity="max|compose - bmm(J2,J1)| / |bmm|",
        measured=rel, expected=0.0, rel_err=rel, tol=1e-12,
        passed=rel < 1e-12,
        note="two-bounce, N=16",
    )


def test_C3() -> StructResult:
    """Adjoint Neumann residual: ||(I − T^T) G − g|| / ||g||."""
    K = kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
    g = _g()
    G = neumann_adjoint(K, g, MAX_D)

    I = torch.eye(GRID.lam.shape[0], dtype=torch.float64)
    residual = ((I - K.T) @ G - g).norm() / g.norm()
    rel = residual.item()

    return StructResult(
        test_id="C3", quantity="||(I-T^T)G - g|| / ||g||",
        measured=rel, expected=0.0, rel_err=rel, tol=1e-6,
        passed=rel < 1e-6,
        note=f"D={MAX_D}",
    )


def test_C4() -> GradResult:
    """∂loss/∂d in the two_bounce scene (dispersive coupling)."""
    d = torch.tensor(D_NM, requires_grad=True)

    def fn():
        r = two_bounce(GRID, SENSOR, d=d, A=A_CAU, B=B_CAU, cos_i=COS_I,
                       lam_ex=LAM_EX, lam_em=LAM_EM, sigma_f=SIG_F,
                       max_depth=MAX_D, quantum_yield=QY, L_e=_L_E)
        return _loss(r.image)

    with torch.no_grad():
        K0    = (kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
                 + kernel_fluorescence(GRID.lam, LAM_EX, LAM_EM, SIG_F,
                                       GRID.weights, QY))
        dR_dd = fabry_airy_dR_dd(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
        an    = kernel_gradient(K0, dR_dd, _L_E, _g(), MAX_D).item()

    return run_three_way(fn, d, an, "C4", "two_bounce", "d")


def test_C5() -> GradResult:
    """dloss/dd with per-wavelength cos_i -> dense kernel_thinfilm (dispersive).

    Exercises the cos_i_is_vec branch in kernel_thinfilm, which produces a full
    (N,N) K matrix (off-diagonal entries non-zero). cos_i varies per wavelength
    because the ray exited Cauchy glass at a fixed angle -- dispersive IOR makes
    the refracted angle wavelength-dependent.

    This is a two-way test: FD vs autograd. The 'analytic' path uses
    kernel_gradient with dK/dd obtained via torch.autograd.jacobian through
    kernel_thinfilm only (not the full scene), so it is a genuinely different
    computation from the full-scene autograd path. FD remains the ground truth.

    Uses d=5nm so rho(K_dense) ~ 0.2 < 1 (at d=120nm rho ~ 16, Neumann diverges).
    """
    # Per-wavelength cos_i in air: an angular sweep from 35 to 55 degrees across the
    # spectral band (models a grating-dispersed beam arriving at the film at
    # wavelength-dependent angles). All values are valid non-TIR air incidence.
    C5_D = 5.0   # thin film: rho(K_dense) ~ 0.2 at this thickness
    N = GRID.lam.shape[0]
    cos_i_per_lam = torch.cos(torch.linspace(0.611, 0.960, N, dtype=torch.float64))

    d = torch.tensor(C5_D, requires_grad=True)

    def fn():
        K = kernel_thinfilm(GRID.lam, cos_i_per_lam, d, A_CAU, B_CAU)
        L = neumann_forward(K, _L_E, MAX_D)
        return _loss(SENSOR.measure(L))

    # Analytic: adjoint formula with dK/dd from jacobian through kernel only.
    # dK_dd[i,j] = dK_dense[i,j]/dd -- full (N,N) matrix (dense off-diagonals).
    with torch.no_grad():
        K0 = kernel_thinfilm(GRID.lam, cos_i_per_lam, C5_D, A_CAU, B_CAU)

    d_t   = torch.tensor(C5_D, requires_grad=True, dtype=torch.float64)
    dK_dd = torch.autograd.functional.jacobian(
        lambda dv: kernel_thinfilm(GRID.lam, cos_i_per_lam, dv, A_CAU, B_CAU),
        d_t,
    ).detach()  # (N, N)

    an = kernel_gradient(K0, dK_dd, _L_E, _g(), MAX_D).item()

    return run_three_way(fn, d, an, "C5", "single_bounce_dense_K", "d")


def test_C6() -> StructResult:
    """Neumann convergence: ||L_32 − L_exact|| / ||L_exact||."""
    K = kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
    L_neu  = neumann_forward(K, _L_E, MAX_D)
    L_ex   = fredholm_solve_exact(K, _L_E)
    rel    = ((L_neu - L_ex).norm() / L_ex.norm()).item()

    return StructResult(
        test_id="C6", quantity="||L_32 - L_exact|| / ||L_exact||",
        measured=rel, expected=0.0, rel_err=rel, tol=1e-6,
        passed=rel < 1e-6,
        note=f"D={MAX_D}, single_bounce_flat",
    )


def test_C7() -> StructResult:
    """Spectral radius ρ < 1 for K_TF and K_FL separately (Neumann convergence).

    Checks each kernel independently. K_FL is rank-1 with ρ = Φ Σ_i e_i a_i w_i ≤ QY.
    The previous test also checked ρ(K_TF + K_FL) but that differed from ρ(K_TF)
    by float64 noise (~1e-16), so it added no information.
    """
    K_TF = kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
    K_FL = kernel_fluorescence(GRID.lam, LAM_EX, LAM_EM, SIG_F, GRID.weights, QY)
    rho_tf = torch.linalg.eigvals(K_TF).abs().max().item()
    rho_fl = torch.linalg.eigvals(K_FL).abs().max().item()
    rho    = max(rho_tf, rho_fl)

    return StructResult(
        test_id="C7", quantity="max rho(K_TF), rho(K_FL)",
        measured=rho, expected=1.0, rel_err=None,
        tol=1.0, passed=rho < 1.0,
        note=f"rho_TF={rho_tf:.4f}, rho_FL={rho_fl:.4f} (<=QY={QY})",
    )


def test_C8() -> StructResult:
    """Half-attached bias (B7-spectral): measured bias vs analytic missing term.

    Uses lam_em near the blue grid edge so the emission Gaussian is
    asymmetrically truncated, making dC/dlam_em measurably nonzero.
    (At lam_em=530nm the grid is symmetric → dC≈0 → unmeasurable bias.)
    """
    # lam_em near edge: grid runs 360-830nm, lam_em=390 is 1.5σ from edge
    C8_LAM_EX = 370.0
    C8_LAM_EM = 400.0
    C8_SIG_F  = 20.0
    lam_em_t = torch.tensor(C8_LAM_EM, requires_grad=True)

    # Correct gradient
    def fn_correct():
        K  = (kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
              + kernel_fluorescence(GRID.lam, C8_LAM_EX, lam_em_t, C8_SIG_F,
                                    GRID.weights, QY))
        return _loss(SENSOR.measure(neumann_forward(K, _L_E, MAX_D)))

    lam_em_t.grad = None
    fn_correct().backward()
    grad_correct = lam_em_t.grad.item()
    lam_em_t.grad.zero_()

    # Half-attached gradient
    def fn_half():
        K  = (kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
              + kernel_fluorescence_half_attached(GRID.lam, C8_LAM_EX, lam_em_t,
                                                  C8_SIG_F, GRID.weights, QY))
        return _loss(SENSOR.measure(neumann_forward(K, _L_E, MAX_D)))

    fn_half().backward()
    grad_half = lam_em_t.grad.item()
    lam_em_t.grad.zero_()

    bias_measured = grad_correct - grad_half

    # Analytic missing term: G^T · ΔdK · L
    # ΔdK = dK_correct/dlam_em - dK_half/dlam_em
    #      = −(e × ∂C/∂lam_em / C).unsqueeze(1) × (a·w).unsqueeze(0)
    with torch.no_grad():
        sigma2   = C8_SIG_F ** 2
        a        = torch.exp(-0.5 * ((GRID.lam - C8_LAM_EX) / C8_SIG_F) ** 2)
        e_raw    = torch.exp(-0.5 * ((GRID.lam - C8_LAM_EM) / C8_SIG_F) ** 2)
        C        = (e_raw * GRID.weights).sum()
        e        = e_raw / C
        de_raw   = e_raw * (GRID.lam - C8_LAM_EM) / sigma2
        dC       = (de_raw * GRID.weights).sum()
        delta_de = -e * dC / C                                   # missing quotient-rule term
        aw       = a * GRID.weights
        delta_dK = QY * delta_de.unsqueeze(1) * aw.unsqueeze(0) # (N,N)

        K0  = (kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
               + kernel_fluorescence(GRID.lam, C8_LAM_EX, C8_LAM_EM, C8_SIG_F,
                                     GRID.weights, QY))
        bias_analytic = kernel_gradient(K0, delta_dK, _L_E, _g(), MAX_D).item()

    rel = _rel(bias_measured, bias_analytic)
    bias_frac = abs(bias_measured / max(abs(grad_correct), 1e-30))

    return StructResult(
        test_id="C8", quantity="half-attached bias",
        measured=bias_measured, expected=bias_analytic,
        rel_err=rel, tol=1e-4,
        passed=rel < 1e-4,
        note=f"bias/grad={bias_frac:.1%}",
    )


def test_C9() -> StructResult:
    """Dense vs diagonal: scalar cos_i → kernel_thinfilm is diagonal."""
    K = kernel_thinfilm(GRID.lam, COS_I, D_NM, A_CAU, B_CAU)
    off_diag = K.clone()
    off_diag.fill_diagonal_(0.0)
    max_off = off_diag.abs().max().item()
    diag_scale = K.diagonal().abs().max().item()
    rel = max_off / max(diag_scale, 1e-30)

    return StructResult(
        test_id="C9", quantity="max off-diag / max diag",
        measured=rel, expected=0.0, rel_err=rel, tol=1e-14,
        passed=rel < 1e-14,
        note="scalar cos_i must give diagonal K",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rep = Reporter()

    print("Running A-series (thin-film gradient tests)...")
    for fn in (test_A0, test_A1, test_A2):
        rep.add(fn())

    print("Running B-series (fluorescence gradient tests)...")
    for fn in (test_B7, test_B8, test_B9):
        rep.add(fn())

    print("Running C-series (structural invariants + cross-scene gradients)...")
    for fn in (test_C0, test_C1, test_C2, test_C3,
               test_C4, test_C5,
               test_C6, test_C7, test_C8, test_C9):
        rep.add(fn())

    rep.print_all()

    out = Path("results")
    rep.save_csv(out)
    rep.save_latex(out)
    print(f"\nResults saved to {out}/")
    print("\nOverall:", "ALL PASS" if rep.all_passed() else "SOME FAILURES")


if __name__ == "__main__":
    main()
