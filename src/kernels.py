import torch
from .cauchy_ior import n_cauchy, cos_theta_t, is_tir
from .fresnel import fresnel_rs, fresnel_rp


# ---------------------------------------------------------------------------
# Diagonal kernel
# ---------------------------------------------------------------------------

def kernel_diagonal(fr: torch.Tensor) -> torch.Tensor:
    """K(λ,λ') = f_r(λ) δ(λ−λ'). Returns (N, N) diagonal matrix."""
    return torch.diag(fr)


# ---------------------------------------------------------------------------
# Rank-1 fluorescence kernel
# ---------------------------------------------------------------------------

def kernel_fluorescence(
    lam: torch.Tensor,
    lam_ex: torch.Tensor | float,
    lam_em: torch.Tensor | float,
    sigma_f: torch.Tensor | float,
    weights: torch.Tensor,
    quantum_yield: float = 1.0,
) -> torch.Tensor:
    """Fluorescence operator matrix T[i,j] = quantum_yield · e(λ_i) · a(λ_j) · w_j.

    Returns the operator matrix directly (w baked into j-dimension) so it can
    be passed straight to fredholm_solve() alongside diagonal/thin-film operators.

    a(λ') = Gaussian absorption centered at lam_ex,  peak = 1
    e(λ)  = Gaussian emission centered at lam_em, normalized so Σ_i e_i w_i = 1
    weights: (N,) quadrature weights (λ²Δν̃ trapezoid)

    Convention: lam_em > lam_ex (Stokes — emission is red-shifted).
    Returns (N, N).
    """
    a = torch.exp(-0.5 * ((lam - lam_ex) / sigma_f) ** 2)           # (N,)
    e = torch.exp(-0.5 * ((lam - lam_em) / sigma_f) ** 2)           # (N,)
    e = e / (e * weights).sum()                                        # normalize
    aw = a * weights                                                    # a_j · w_j
    return quantum_yield * e.unsqueeze(1) * aw.unsqueeze(0)           # (N, N)


# ---------------------------------------------------------------------------
# Thin-film (Fabry-Airy) kernel — internal helpers
# ---------------------------------------------------------------------------

def _fabry_airy_pol(r12: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Fabry-Airy power reflectance for one polarization, air-film-air (r23 = -r12).

    Uses complex exponential for autograd through d, A, B.
    r12, phi: real tensors of identical shape.
    Returns real R of same shape.
    """
    eiphi = torch.polar(torch.ones_like(phi), phi)     # complex, |·|=1, arg=φ
    A = r12 - r12 * eiphi                              # r12 + r23·exp(iφ), r23=-r12
    B = 1.0 - r12 ** 2 * eiphi                         # 1 + r12·r23·exp(iφ)
    return (A.abs() ** 2) / (B.abs() ** 2)


def _r12_and_phi(
    lam: torch.Tensor,
    cos_i,
    d,
    A_cauchy,
    B_cauchy,
    polarization: str,
):
    """Shared computation: Fresnel r12 and round-trip phase φ."""
    n = n_cauchy(lam, A_cauchy, B_cauchy)
    cos_t = cos_theta_t(cos_i, 1.0, n)
    phi = 4.0 * torch.pi * n * cos_t * d / lam         # round-trip phase

    rs = fresnel_rs(1.0, n, cos_i, cos_t)
    rp = fresnel_rp(1.0, n, cos_i, cos_t)
    return rs, rp, phi, n, cos_t


# ---------------------------------------------------------------------------
# Public Fabry-Airy API
# ---------------------------------------------------------------------------

def fabry_airy_R(
    lam: torch.Tensor,
    cos_i,
    d,
    A: torch.Tensor | float,
    B: torch.Tensor | float,
    polarization: str = "unpolarized",
) -> torch.Tensor:
    """Fabry-Airy power reflectance for a free-standing thin film (air-film-air).

    lam:   (N,) wavelengths [nm]
    cos_i: scalar or (N,) cosine of incidence angle in air
    d:     film thickness [nm]  (may require_grad)
    A, B:  Cauchy coefficients  (may require_grad)
    Returns R (N,) ∈ [0, 1].  TIR wavelengths → R = 1.0.

    Unpolarized = 0.5 · (R_s + R_p) on power, not averaged amplitudes.
    """
    n = n_cauchy(lam, A, B)
    tir = is_tir(cos_i, 1.0, n)
    rs, rp, phi, _, _ = _r12_and_phi(lam, cos_i, d, A, B, polarization)

    if polarization == "s":
        R = _fabry_airy_pol(rs, phi)
    elif polarization == "p":
        R = _fabry_airy_pol(rp, phi)
    else:
        R = 0.5 * (_fabry_airy_pol(rs, phi) + _fabry_airy_pol(rp, phi))

    return torch.where(tir, torch.ones_like(R), R)


def fabry_airy_dR_dd(
    lam: torch.Tensor,
    cos_i,
    d,
    A: torch.Tensor | float,
    B: torch.Tensor | float,
    polarization: str = "unpolarized",
) -> torch.Tensor:
    """Analytic ∂R/∂d for air-film-air thin film (Eq. 15).

    Derivation:
        φ = 4π n cosθ_t d / λ  →  ∂φ/∂d = 4π n cosθ_t / λ
        ∂R/∂φ = −2 r12 r23 sinφ (1−R) / |B|²,   r23 = −r12
               = 2 r12² sinφ (1−R) / |B|²
        ∂R/∂d = ∂R/∂φ · ∂φ/∂d

    TIR wavelengths → ∂R/∂d = 0 (gradient dead zone).
    Returns (N,).
    """
    n = n_cauchy(lam, A, B)
    cos_t = cos_theta_t(cos_i, 1.0, n)
    tir = is_tir(cos_i, 1.0, n)
    rs, rp, phi, _, _ = _r12_and_phi(lam, cos_i, d, A, B, polarization)

    dphidd = 4.0 * torch.pi * n * cos_t / lam

    def _dR_dd_pol(r12):
        denom2 = (1.0 + r12 ** 4 - 2.0 * r12 ** 2 * torch.cos(phi))   # |B|²
        R = _fabry_airy_pol(r12, phi)
        dR_dphi = 2.0 * r12 ** 2 * torch.sin(phi) * (1.0 - R) / denom2
        return dR_dphi * dphidd

    if polarization == "s":
        dR_dd = _dR_dd_pol(rs)
    elif polarization == "p":
        dR_dd = _dR_dd_pol(rp)
    else:
        dR_dd = 0.5 * (_dR_dd_pol(rs) + _dR_dd_pol(rp))

    return torch.where(tir, torch.zeros_like(dR_dd), dR_dd)


# ---------------------------------------------------------------------------
# Thin-film kernel matrix
# ---------------------------------------------------------------------------

def kernel_thinfilm(
    lam: torch.Tensor,
    cos_i,
    d,
    A,
    B,
    polarization: str = "unpolarized",
) -> torch.Tensor:
    """Thin-film bispectral kernel as (N, N) matrix.

    cos_i scalar  → diagonal K[i,i] = R(λ_i)  (single-bounce oracle)
    cos_i (N,)    → dense   K[i,j] = R(λ_i; θ_incidence at λ_j)
                    Used in multi-bounce: incoming geometry is set by wavelength λ_j
                    (dispersive upstream refraction), evaluated at output wavelength λ_i.

    Returns (N, N).
    """
    N = lam.shape[0]
    cos_i_is_vec = isinstance(cos_i, torch.Tensor) and cos_i.shape == (N,)

    if cos_i_is_vec:
        lam_i = lam.unsqueeze(1).expand(N, N)          # (N, N) — output wavelength
        cos_i_j = cos_i.unsqueeze(0).expand(N, N)      # (N, N) — incidence from λ_j path

        n_i = n_cauchy(lam_i, A, B)
        cos_t = cos_theta_t(cos_i_j, 1.0, n_i)
        tir = is_tir(cos_i_j, 1.0, n_i)
        phi = 4.0 * torch.pi * n_i * cos_t * d / lam_i

        rs = fresnel_rs(1.0, n_i, cos_i_j, cos_t)
        rp = fresnel_rp(1.0, n_i, cos_i_j, cos_t)

        if polarization == "s":
            R = _fabry_airy_pol(rs, phi)
        elif polarization == "p":
            R = _fabry_airy_pol(rp, phi)
        else:
            R = 0.5 * (_fabry_airy_pol(rs, phi) + _fabry_airy_pol(rp, phi))

        return torch.where(tir, torch.ones_like(R), R)

    else:
        return torch.diag(fabry_airy_R(lam, cos_i, d, A, B, polarization))
