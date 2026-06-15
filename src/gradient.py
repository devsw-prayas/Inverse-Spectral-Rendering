"""Gradient utilities for the bispectral forward oracle.

Three gradient paths for Tier 1-2 three-way comparison tests (A0-C9):
  1. Autograd  — .backward() through neumann_forward, always available
  2. Analytic  — kernel_gradient() using explicit ∂T/∂θ formulas
  3. FD        — fd_gradient(), ground truth at ~h=1e-6|θ|

For thin-film ∂/∂d, the analytic ∂T/∂d is diagonal (fabry_airy_dR_dd).
For fluorescence ∂/∂lam_em, the analytic ∂T/∂lam_em is rank-1.

Wrong-gradient oracle (Test B7-spectral):
  kernel_fluorescence_half_attached — detaches the normalization constant C
  so autograd misses the ∂C/∂lam_em quotient-rule term, giving a biased gradient.
"""

from __future__ import annotations

import torch

from .forward import neumann_forward
from .kernels import kernel_fluorescence


# ---------------------------------------------------------------------------
# Finite-difference ground truth
# ---------------------------------------------------------------------------

def fd_gradient(
    fn:    "callable[[], torch.Tensor]",
    param: torch.Tensor,
    h:     float | None = None,
) -> torch.Tensor:
    """Central finite differences: ∂fn() / ∂param.

    fn:    callable returning a scalar tensor; called with param at its
           current value. Must not close over any other requires_grad tensors
           that depend on param — use torch.no_grad() internally if needed.
    param: 0-dim (scalar) tensor — temporarily perturbed in place.
    h:     step; defaults to max(1e-6 |param|, 1e-6). At float64 this gives
           an FD floor of ~h² ≈ 1e-12 relative error (expected FD floor).

    Returns a detached scalar tensor.
    """
    if h is None:
        h = max(1e-6 * abs(param.item()), 1e-6)

    orig = param.item()
    with torch.no_grad():
        param.fill_(orig + h)
        fp = fn().detach().clone()
        param.fill_(orig - h)
        fm = fn().detach().clone()
        param.fill_(orig)                   # restore unconditionally

    return (fp - fm) / (2.0 * h)


# ---------------------------------------------------------------------------
# Adjoint Neumann sum
# ---------------------------------------------------------------------------

def neumann_adjoint(
    T:         torch.Tensor,
    g:         torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Adjoint Neumann sum: G = Σ_{k=0}^{max_depth} (T^T)^k g.

    G is the adjoint radiance — converges to (I − T^T)^{-1} g for ρ(T) < 1.
    Must match the max_depth used in the primal neumann_forward call.

    T:         (N, N) operator matrix (same as primal)
    g:         (N,)   ∂loss / ∂L  — gradient of the scalar loss w.r.t. L
    max_depth: D — number of adjoint bounces
    Returns G  (N,)
    """
    G    = g
    term = g
    T_T  = T.T
    for _ in range(max_depth):
        term = T_T @ term
        G    = G + term
    return G


# ---------------------------------------------------------------------------
# Analytic kernel gradient
# ---------------------------------------------------------------------------

def kernel_gradient(
    T:         torch.Tensor,
    dT_dθ:    torch.Tensor,
    L_e:       torch.Tensor,
    g:         torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Analytic ∂loss/∂θ = G^T · (∂T/∂θ) · L.

    Exact in the limit max_depth → ∞; error ∝ ρ(T)^max_depth.
    At max_depth = 32 and ρ ≲ 0.4, error < 10^{-18} — below float64 floor.

    T:        (N, N) operator matrix
    dT_dθ:   (N,) or (N, N)
              Pass (N,) for diagonal ∂T/∂θ (thin-film ∂/∂d, ∂/∂A, ∂/∂B).
              Pass (N, N) for dense ∂T/∂θ (fluorescence rank-1 matrices).
    L_e:      (N,) source radiance
    g:        (N,) ∂loss/∂L  (e.g. S^T @ ones(M) for loss = image.sum())
    max_depth: D — must match the primal forward call

    Returns a detached scalar.

    Typical usage (thin film, loss = image.sum()):
        dR_dd = fabry_airy_dR_dd(grid.lam, cos_i, d, A, B)   # (N,)
        g     = sensor.S.T @ torch.ones(M)                    # (N,)
        grad  = kernel_gradient(K, dR_dd, L_e, g, max_depth)
    """
    L = neumann_forward(T, L_e, max_depth)   # (N,) primal radiance
    G = neumann_adjoint(T, g, max_depth)     # (N,) adjoint radiance

    if dT_dθ.dim() == 1:
        # Diagonal: G^T diag(dT_dθ) L = (G ⊙ dT_dθ ⊙ L).sum()
        return (G * dT_dθ * L).sum().detach()
    else:
        # Dense: G^T (dT_dθ @ L)
        return (G @ (dT_dθ @ L)).detach()


# ---------------------------------------------------------------------------
# Wrong-gradient oracle — Test B7-spectral
# ---------------------------------------------------------------------------

def kernel_fluorescence_half_attached(
    lam:          torch.Tensor,
    lam_ex:       torch.Tensor | float,
    lam_em:       torch.Tensor | float,
    sigma_f:      torch.Tensor | float,
    weights:      torch.Tensor,
    quantum_yield: float = 1.0,
) -> torch.Tensor:
    """Half-attached fluorescence kernel — biased gradient oracle for B7.

    Identical to kernel_fluorescence except the normalization constant
    C = (e_raw · w).sum() is detached before dividing. Autograd therefore
    misses the quotient-rule term -e · ∂C/∂lam_em / C and returns a biased
    ∂loss/∂lam_em. The discrepancy vs the correct gradient isolates the
    'position-velocity' bias in Zeltner-style half-attached estimators.

    All parameters and return shape are the same as kernel_fluorescence.
    """
    a     = torch.exp(-0.5 * ((lam - lam_ex) / sigma_f) ** 2)    # (N,)
    e_raw = torch.exp(-0.5 * ((lam - lam_em) / sigma_f) ** 2)    # (N,)

    # Detach C: gradient treats normalization as a constant — the bias source.
    C = (e_raw * weights).sum().detach()
    e = e_raw / C                                                   # norm missing from graph

    aw = a * weights
    return quantum_yield * e.unsqueeze(1) * aw.unsqueeze(0)        # (N, N)


# ---------------------------------------------------------------------------
# Convenience: ∂loss/∂lam_em for fluorescence kernel via analytic formula
# ---------------------------------------------------------------------------

def fluorescence_dK_dlam_em(
    lam:     torch.Tensor,
    lam_ex:  torch.Tensor | float,
    lam_em:  torch.Tensor | float,
    sigma_f: torch.Tensor | float,
    weights: torch.Tensor,
    quantum_yield: float = 1.0,
) -> torch.Tensor:
    """Analytic ∂K_fl/∂lam_em — (N, N) matrix, quotient-rule correct.

    ∂e/∂lam_em = (1/C) [∂e_raw/∂lam_em - e · ∂C/∂lam_em]
    where ∂e_raw_i/∂lam_em = e_raw_i · (λ_i - lam_em) / σ²

    Returns (N, N) operator matrix (ready for kernel_gradient).
    """
    sigma2 = sigma_f ** 2

    a     = torch.exp(-0.5 * ((lam - lam_ex) / sigma_f) ** 2)    # (N,)
    e_raw = torch.exp(-0.5 * ((lam - lam_em) / sigma_f) ** 2)    # (N,)
    C     = (e_raw * weights).sum()
    e     = e_raw / C                                              # normalized

    de_raw = e_raw * (lam - lam_em) / sigma2                      # ∂e_raw_i/∂lam_em
    dC     = (de_raw * weights).sum()                              # ∂C/∂lam_em
    de     = (de_raw - e * dC) / C                                 # quotient rule (N,)

    aw = a * weights
    return quantum_yield * de.unsqueeze(1) * aw.unsqueeze(0)      # (N, N)
