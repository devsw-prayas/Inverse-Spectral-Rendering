"""Gradient utilities for the bispectral forward oracle."""

from __future__ import annotations

import torch

from .cauchy_ior import n_cauchy, cos_theta_t
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
    h:     step; defaults to max(1e-6 |param|, 1e-6). At float64, roundoff
           error ε/h dominates at this step size, giving a realistic FD
           floor of ~1e-9 relative error — not ~h² ≈ 1e-12 (truncation only).

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
    dT_dtheta:    torch.Tensor,
    L_e:       torch.Tensor,
    g:         torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Analytic ∂loss/∂θ = G^T · (∂T/∂θ) · L.

    Exact in the limit max_depth → ∞; error ∝ ρ(T)^max_depth.
    At max_depth = 32 and ρ ≲ 0.4, error < 10^{-18} — below float64 floor.

    T:        (N, N) operator matrix
    dT_dtheta:   (N,) or (N, N)
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

    if dT_dtheta.dim() == 1:
        # Diagonal: G^T diag(dT_dtheta) L = (G ⊙ dT_dtheta ⊙ L).sum()
        return (G * dT_dtheta * L).sum().detach()
    else:
        # Dense: G^T (dT_dtheta @ L)
        return (G @ (dT_dtheta @ L)).detach()


def kernel_gradient_wrong_adjoint(
    T:         torch.Tensor,
    dT_dtheta:    torch.Tensor,
    L_e:       torch.Tensor,
    g:         torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Wrong gradient: adjoint sourced by (∂T/∂θ)L instead of S.

    Solves (I − T^T) G_wrong = (∂T/∂θ)L, then returns g · G_wrong.
    Equals the correct gradient only when T = T^T (zero Stokes shift).
    For fluorescence with a Stokes shift, gives ~half the correct value.

    Same signature as kernel_gradient — swap in to isolate the bug.
    """
    L = neumann_forward(T, L_e, max_depth)
    if dT_dtheta.dim() == 1:
        source = dT_dtheta * L
    else:
        source = dT_dtheta @ L
    G_wrong = neumann_adjoint(T, source, max_depth)
    return (g @ G_wrong).detach()


# ---------------------------------------------------------------------------
# Wrong-gradient oracle — Test B7-spectral / C8
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
# Analytic fluorescence kernel derivatives
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


def fluorescence_dK_dlam_ex(
    lam:     torch.Tensor,
    lam_ex:  torch.Tensor | float,
    lam_em:  torch.Tensor | float,
    sigma_f: torch.Tensor | float,
    weights: torch.Tensor,
    quantum_yield: float = 1.0,
) -> torch.Tensor:
    """Analytic ∂K_fl/∂lam_ex — (N, N) matrix.

    Only the absorption profile a(λ') changes; emission e(λ) is independent.
    ∂a_j/∂lam_ex = a_j · (λ_j − lam_ex) / σ²
    """
    sigma2 = sigma_f ** 2
    a      = torch.exp(-0.5 * ((lam - lam_ex) / sigma_f) ** 2)
    e_raw  = torch.exp(-0.5 * ((lam - lam_em) / sigma_f) ** 2)
    C      = (e_raw * weights).sum()
    e      = e_raw / C

    da     = a * (lam - lam_ex) / sigma2                          # ∂a/∂lam_ex  (N,)
    daw    = da * weights
    return quantum_yield * e.unsqueeze(1) * daw.unsqueeze(0)      # (N, N)


def fluorescence_dK_dsigma_f(
    lam:     torch.Tensor,
    lam_ex:  torch.Tensor | float,
    lam_em:  torch.Tensor | float,
    sigma_f: torch.Tensor | float,
    weights: torch.Tensor,
    quantum_yield: float = 1.0,
) -> torch.Tensor:
    """Analytic ∂K_fl/∂sigma_f — (N, N) matrix.

    Both a and e (through C) change.
    ∂a/∂σ  = a · (λ−lam_ex)² / σ³
    ∂e/∂σ  = (1/C)[∂e_raw/∂σ − e · ∂C/∂σ]   (quotient rule, same pattern as ∂/∂lam_em)
    """
    sigma2 = sigma_f ** 2
    sigma3 = sigma_f ** 3

    a      = torch.exp(-0.5 * ((lam - lam_ex) / sigma_f) ** 2)
    e_raw  = torch.exp(-0.5 * ((lam - lam_em) / sigma_f) ** 2)
    C      = (e_raw * weights).sum()
    e      = e_raw / C

    da_ds      = a * (lam - lam_ex) ** 2 / sigma3
    de_raw_ds  = e_raw * (lam - lam_em) ** 2 / sigma3
    dC_ds      = (de_raw_ds * weights).sum()
    de_ds      = (de_raw_ds - e * dC_ds) / C

    aw  = a * weights
    daw = da_ds * weights
    return quantum_yield * (
        de_ds.unsqueeze(1) * aw.unsqueeze(0)
        + e.unsqueeze(1) * daw.unsqueeze(0)
    )                                                              # (N, N)


# ---------------------------------------------------------------------------
# Moving-boundary gradient (§13) — scope {A, B} only
# ---------------------------------------------------------------------------

def lambda_star(
    A:     torch.Tensor | float,
    B:     torch.Tensor | float,
    cos_i: torch.Tensor | float,
) -> torch.Tensor:
    """Critical wavelength λ*(A,B) = sqrt(B / (κ − A)),  κ = 1/sinθᵢ.

    The TIR critical condition n(λ*)sinθᵢ = 1 with n(λ) = A + B/λ² gives this
    closed form. Returns a scalar tensor.
    """
    sin_i = (1.0 - cos_i ** 2) ** 0.5
    kappa = 1.0 / sin_i
    return (B / (kappa - A)) ** 0.5


def dlambda_star_dA(
    lam_star: torch.Tensor | float,
    B:        torch.Tensor | float,
) -> torch.Tensor:
    """dλ*/dA = λ*³ / (2B)."""
    return lam_star ** 3 / (2.0 * B)


def dlambda_star_dB(
    lam_star: torch.Tensor | float,
    B:        torch.Tensor | float,
) -> torch.Tensor:
    """dλ*/dB = λ* / (2B)."""
    return lam_star / (2.0 * B)


def moving_boundary_grad(
    e_at_star:       torch.Tensor | float,
    dlam_star_dtheta: torch.Tensor | float,
) -> torch.Tensor:
    """Moving-boundary contribution to ∂I/∂θ: −e(λ*) · dλ*/dθ.

    Scope: θ ∈ {A, B} only — the only parameters that move the critical
    wavelength λ*. Returns a scalar. e_at_star is the (normalized) emission
    profile evaluated at λ*; dlam_star_dtheta from dlambda_star_dA/dB.
    """
    return -e_at_star * dlam_star_dtheta


# ---------------------------------------------------------------------------
# V9: moving-boundary Leibniz term at full multi-bounce scene complexity
# ---------------------------------------------------------------------------
#
# Scene: fluorescence (K_fl, rank-1) living inside a dispersive medium behind
# a single TIR-bounded interface (K_e = diag(R(lam)), plain Fresnel R -- no
# thin-film oscillation here). Solved exactly (Sherman-Morrison, not a
# truncated Neumann series) on the propagating-only subset Omega =
# {lam : lam > lambda_star(A,B)}, matching V1 Check 5/6's precedent that this
# restriction is well-posed even as R -> 1 at the boundary.
#
# Since G0 := (I - diag(R))^-1 is diagonal, (1-R)*G0 = 1 exactly, so the
# escaping-flux density collapses to phi(lam) = L_e(lam) + e(lam)*c, with the
# feedback amplitude c = qy*A_int/(1 - qy*B_int) (Sherman-Morrison scalar for
# the rank-1 K_fl = e (x) v term). The measured quantity is
#     I(theta) = integral_{lambda_star(theta)}^{lam_max} phi(lam) dlam = P + c*Q
# P, Q (bare L_e, e integrals) are smooth/bounded -- their moving-boundary
# term is exactly the already-locked lambda_star/moving_boundary_grad lemma
# above. But A_int, B_int (integrals of a(lam')/(1-R(lam')) against L_e, e)
# have an INTEGRABLE 1/sqrt(lam'-lambda_star) singularity right at the moving
# boundary itself (1-R ~ v ~ sqrt(lam'-lambda_star) near the critical
# wavelength) -- the moving-boundary lemma assumes a BOUNDED integrand there,
# so it only applies directly to P and Q, not A_int/B_int.
#
# Fix: substitute w = sqrt(lam - lambda_star(theta)), then w = W(theta)*t,
# t in [0,1], W(theta) = sqrt(lam_max - lambda_star(theta)). dlam = 2w dw
# cancels the leading 1/w blowup exactly, AND fixes the integration domain to
# [0,1] so autograd differentiates straight through
# theta -> lambda_star(theta) -> the integrand, recovering both the boundary
# term and the interior R(lam;theta) dependence automatically -- no
# hand-assembled Leibniz term needed for A_int/B_int. Quadrature uses
# Gauss-Legendre nodes on the OPEN interval (0,1) (via Golub-Welsch
# eigendecomposition, pure torch, no numpy) -- critically, none of the nodes
# land on the endpoint t=0, which is a removable-but-literal 0/0 at the exact
# critical wavelength (1-R and dlam/dw both vanish there).
# ---------------------------------------------------------------------------

def gauss_legendre_01(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre quadrature nodes/weights on the open interval (0, 1).

    Nodes are eigenvalues of the tridiagonal Jacobi matrix for Legendre
    polynomials (Golub-Welsch); weights come from the first component of each
    eigenvector. All nodes are strictly interior to (0, 1) -- unlike a
    Simpson/trapezoid rule, none lands on an endpoint, which matters when
    t=0 is a removable-but-literal singularity (see module comment above).
    """
    k = torch.arange(1, n, dtype=torch.float64)
    beta = k / torch.sqrt(4.0 * k ** 2 - 1.0)
    J = torch.diag(beta, 1) + torch.diag(beta, -1)
    nodes, vecs = torch.linalg.eigh(J)
    weights = 2.0 * vecs[0, :] ** 2
    return 0.5 * (nodes + 1.0), 0.5 * weights


def interface_T_stable(
    v:            torch.Tensor,
    eta:          torch.Tensor,
    cos_i:        torch.Tensor | float,
    polarization: str = "unpolarized",
) -> torch.Tensor:
    """Stable T(v) = 1 - R for a single dielectric interface, v = cosθ_t.

    Same cancellation landmine as G1/tir_jacobian: naive 1 - r² loses several
    orders of magnitude near v -> 0 (r -> 1, r² computed near 1, subtracted
    from 1). Uses the rational form directly instead (numerator explicitly
    O(v), no subtraction of near-equal quantities):
        T_s(v) = 4ηcv / (ηc + v)²,   T_p(v) = 4ηcv / (c + ηv)²
    """
    c   = cos_i
    T_s = 4.0 * eta * c * v / (eta * c + v) ** 2
    T_p = 4.0 * eta * c * v / (c + eta * v) ** 2
    if polarization == "s":
        return T_s
    if polarization == "p":
        return T_p
    return 0.5 * (T_s + T_p)


def v9_escaping_flux(
    A:             torch.Tensor,
    B:             torch.Tensor,
    cos_i:         torch.Tensor | float,
    lam_max:       float,
    lam_ex:        float,
    lam_em:        float,
    sigma_f:       float,
    quantum_yield: float,
    L0:            float,
    t_nodes:       torch.Tensor,
    t_weights:     torch.Tensor,
    polarization:  str = "unpolarized",
) -> tuple[torch.Tensor, dict]:
    """I(θ) = ∫_{λ*(A,B)}^{lam_max} (1−R(λ))·L(λ) dλ, fixed-domain reformulation.

    θ ∈ {A, B} enters only through λ*(θ); w = √(λ−λ*), w = W(θ)·t maps the
    moving domain to a fixed t ∈ [0,1] (see module comment). Differentiable
    end to end via plain torch autograd — both the λ_max boundary term (via
    W(θ)) and the interior R(λ;θ) dependence come out automatically.

    L_e is a flat illuminant (constant L0); a, e are the usual peak-1 /
    unit-integral Gaussians (continuum normalization √(2π)σ_f — exact, no
    grid needed).

    Returns (I, info); info carries lam_star, W, and the A_int/B_int/P/Q/c
    intermediates (used for the naive-boundary-only comparison in test_V9).
    """
    lam_star = lambda_star(A, B, cos_i)
    W = torch.sqrt(lam_max - lam_star)
    w = W * t_nodes
    lam = lam_star + w ** 2

    n = n_cauchy(lam, A, B)
    v = cos_theta_t(cos_i, n, 1.0)
    Tv = interface_T_stable(v, n, cos_i, polarization)

    a     = torch.exp(-0.5 * ((lam - lam_ex) / sigma_f) ** 2)
    e_raw = torch.exp(-0.5 * ((lam - lam_em) / sigma_f) ** 2)
    e     = e_raw / ((2.0 * torch.pi) ** 0.5 * sigma_f)
    L_e   = torch.full_like(lam, L0)

    dlam = 2.0 * w * t_weights * W   # dλ = 2w dw = 2w·W dt

    A_int = (dlam * a * L_e / Tv).sum()
    B_int = (dlam * a * e / Tv).sum()
    P_val = (dlam * L_e).sum()
    Q_val = (dlam * e).sum()

    c_val = quantum_yield * A_int / (1.0 - quantum_yield * B_int)
    I_val = P_val + c_val * Q_val
    return I_val, dict(lam_star=lam_star, W=W, A_int=A_int, B_int=B_int,
                        P_val=P_val, Q_val=Q_val, c_val=c_val)


# ---------------------------------------------------------------------------
# Levenberg-Marquardt recovery (V5, V6) — nonlinear least-squares over a
# differentiable forward model, using the same Jacobian machinery the
# G/T-series conditioning tests already build.
# ---------------------------------------------------------------------------

def levenberg_marquardt(
    residual_fn: "callable[[torch.Tensor], torch.Tensor]",
    theta0:      torch.Tensor,
    max_iter:    int = 80,
    lam_init:    float = 1e-3,
    tol:         float = 1e-14,
) -> tuple[torch.Tensor, float, int]:
    """Marquardt-damped Gauss-Newton: minimize ||residual_fn(theta)||^2.

    Marquardt scaling (J^T J + lam * diag(J^T J)) delta = -J^T r — the
    diag(J^T J) damping (rather than plain lam*I) keeps steps well-scaled
    when parameters span very different physical units (e.g. d ~ 1e2 nm vs
    B ~ 1e3 nm^2), without any manual per-parameter normalization.

    residual_fn: theta (P,) -> residual (M,), differentiable.
    theta0:      (P,) initial guess.
    Returns (theta_hat (P,) detached, final loss, iterations used).
    """
    theta = theta0.clone().detach()
    lam = lam_init
    loss = residual_fn(theta).detach().pow(2).sum().item()
    delta = torch.zeros_like(theta)
    it = 0
    for it in range(max_iter):
        theta_req = theta.clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(residual_fn, theta_req).detach()
        r = residual_fn(theta_req).detach()
        JTJ = J.T @ J
        JTr = J.T @ r
        diag_JTJ = torch.diag(JTJ).clamp(min=1e-30)
        accepted = False
        for _ in range(40):
            A_damped = JTJ + lam * torch.diag(diag_JTJ)
            try:
                delta = torch.linalg.solve(A_damped, -JTr)
            except torch._C._LinAlgError:
                lam *= 10.0
                continue
            theta_new = theta + delta
            try:
                loss_new = residual_fn(theta_new).detach().pow(2).sum().item()
            except ValueError:
                # Trial step landed outside the physically valid region
                # (e.g. H_wp violated) -- treat exactly like a rejected step.
                lam *= 10.0
                continue
            if loss_new < loss:
                theta, loss = theta_new, loss_new
                lam = max(lam / 10.0, 1e-12)
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            break
        if delta.norm().item() < tol * max(theta.norm().item(), 1.0):
            break
    return theta, loss, it
