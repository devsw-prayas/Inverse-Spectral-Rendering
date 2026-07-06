import torch
from .cauchy_ior import n_cauchy, cos_theta_t, is_tir
from .fresnel import fresnel_rs, fresnel_rp


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

def _fabry_airy_pol(r12: torch.Tensor, phi: torch.Tensor, r23: torch.Tensor | None = None) -> torch.Tensor:
    """Fabry-Airy power reflectance for one polarization.

    r23 defaults to -r12 (free-standing air-film-air, r2=-r1 by symmetry).
    Pass an explicit r23 for a film-on-substrate stack where the second
    interface has its own index (see fabry_airy_R_substrate).

    Uses complex exponential for autograd through d, A, B.
    r12, phi: real tensors of identical shape.
    Returns real R of same shape.
    """
    if r23 is None:
        r23 = -r12
    eiphi = torch.polar(torch.ones_like(phi), phi)     # complex, |·|=1, arg=φ
    num = r12 + r23 * eiphi
    den = 1.0 + r12 * r23 * eiphi
    return (num.abs() ** 2) / (den.abs() ** 2)


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


def fabry_airy_R_substrate(
    lam: torch.Tensor,
    cos_i,
    d,
    A: torch.Tensor | float,
    B: torch.Tensor | float,
    C_sub: torch.Tensor | float,
    D_sub: torch.Tensor | float,
    polarization: str = "unpolarized",
) -> torch.Tensor:
    """Fabry-Airy power reflectance for a thin film on a substrate (air/film/substrate).

    Generalizes fabry_airy_R (free-standing, r23 = -r12) to a real film-substrate
    interface with its own Cauchy index n_s = C_sub + D_sub/λ². Substrate-confound
    regime (T6): as C_sub -> A, r23 -> 0 and d becomes unobservable in R(λ).

    TIR is checked at both interfaces (air->film and film->substrate).
    Returns R (N,) in [0, 1].
    """
    n_f = n_cauchy(lam, A, B)
    cos_f = cos_theta_t(cos_i, 1.0, n_f)
    n_s = n_cauchy(lam, C_sub, D_sub)
    cos_s = cos_theta_t(cos_f, n_f, n_s)

    tir = is_tir(cos_i, 1.0, n_f) | is_tir(cos_f, n_f, n_s)
    phi = 4.0 * torch.pi * n_f * cos_f * d / lam

    rs12 = fresnel_rs(1.0, n_f, cos_i, cos_f)
    rp12 = fresnel_rp(1.0, n_f, cos_i, cos_f)
    rs23 = fresnel_rs(n_f, n_s, cos_f, cos_s)
    rp23 = fresnel_rp(n_f, n_s, cos_f, cos_s)

    if polarization == "s":
        R = _fabry_airy_pol(rs12, phi, rs23)
    elif polarization == "p":
        R = _fabry_airy_pol(rp12, phi, rp23)
    else:
        R = 0.5 * (_fabry_airy_pol(rs12, phi, rs23) + _fabry_airy_pol(rp12, phi, rp23))

    return torch.where(tir, torch.ones_like(R), R)


def _dR_dparam(
    rs:         torch.Tensor,
    rp:         torch.Tensor,
    phi:        torch.Tensor,
    tir:        torch.Tensor,
    dphi_dp:    torch.Tensor,
    polarization: str,
) -> torch.Tensor:
    """∂R/∂param = ∂R/∂φ · ∂φ/∂param, TIR-zeroed.  Internal helper."""
    def _pol(r12):
        denom2 = 1.0 + r12 ** 4 - 2.0 * r12 ** 2 * torch.cos(phi)   # |B|²
        R      = _fabry_airy_pol(r12, phi)
        dR_dphi = 2.0 * r12 ** 2 * torch.sin(phi) * (1.0 - R) / denom2
        return dR_dphi * dphi_dp

    if polarization == "s":
        out = _pol(rs)
    elif polarization == "p":
        out = _pol(rp)
    else:
        out = 0.5 * (_pol(rs) + _pol(rp))
    return torch.where(tir, torch.zeros_like(out), out)


def fabry_airy_dR_dd(
    lam: torch.Tensor,
    cos_i,
    d,
    A: torch.Tensor | float,
    B: torch.Tensor | float,
    polarization: str = "unpolarized",
) -> torch.Tensor:
    """Analytic ∂R/∂d for air-film-air thin film (Eq. 15).

    φ = 4π n cosθ_t d / λ  →  ∂φ/∂d = 4π n cosθ_t / λ
    TIR wavelengths → 0 (gradient dead zone).  Returns (N,).
    """
    n     = n_cauchy(lam, A, B)
    cos_t = cos_theta_t(cos_i, 1.0, n)
    tir   = is_tir(cos_i, 1.0, n)
    rs, rp, phi, _, _ = _r12_and_phi(lam, cos_i, d, A, B, polarization)
    dphi_dd = 4.0 * torch.pi * n * cos_t / lam
    return _dR_dparam(rs, rp, phi, tir, dphi_dd, polarization)


def fabry_airy_dR_dA(
    lam: torch.Tensor,
    cos_i,
    d,
    A: float,
    B: float,
    polarization: str = "unpolarized",
) -> torch.Tensor:
    """∂R_i/∂A for each wavelength via exact Jacobian through fabry_airy_R.

    Uses torch.autograd.functional.jacobian — captures both the phase term
    ∂R/∂φ · ∂φ/∂A and the amplitude term ∂R/∂r12 · ∂r12/∂A. Returns (N,).
    """
    A_t = torch.tensor(float(A), dtype=lam.dtype, requires_grad=True)
    J = torch.autograd.functional.jacobian(
        lambda a: fabry_airy_R(lam, cos_i, d, a, B, polarization),
        A_t, vectorize=True,
    )
    return J.detach()   # (N,)


def fabry_airy_dR_dB(
    lam: torch.Tensor,
    cos_i,
    d,
    A: float,
    B: float,
    polarization: str = "unpolarized",
) -> torch.Tensor:
    """∂R_i/∂B for each wavelength via exact Jacobian through fabry_airy_R.

    Captures both the phase term ∂R/∂φ · ∂φ/∂B and the amplitude term
    ∂R/∂r12 · ∂r12/∂B.  Returns (N,).
    """
    B_t = torch.tensor(float(B), dtype=lam.dtype, requires_grad=True)
    J = torch.autograd.functional.jacobian(
        lambda b: fabry_airy_R(lam, cos_i, d, A, b, polarization),
        B_t, vectorize=True,
    )
    return J.detach()   # (N,)


# ---------------------------------------------------------------------------
# Well-posedness check (H_wp, §3)
# ---------------------------------------------------------------------------

def check_hwp(
    R:       torch.Tensor,
    e:       torch.Tensor,
    a:       torch.Tensor,
    weights: torch.Tensor,
    eps:     float = 1e-3,
) -> None:
    """Assert H_wp: sup_λ R(λ) + ‖e‖₂·‖a‖₂ ≤ 1 − ε.

    Raises ValueError if violated. Call at scene load before any solve.

    R:       (N,) Fabry-Airy reflectance (from fabry_airy_R)
    e:       (N,) normalized emission profile
    a:       (N,) absorption profile (unnormalized peak-1 Gaussian)
    weights: (N,) quadrature weights (for L² norm via ‖f‖₂² = Σ f²·w)
    eps:     minimum margin below 1 (default 1e-3)
    """
    sup_R   = R.max().item()
    norm_e  = ((e ** 2 * weights).sum()) ** 0.5
    norm_a  = ((a ** 2 * weights).sum()) ** 0.5
    total   = sup_R + norm_e.item() * norm_a.item()
    if total > 1.0 - eps:
        raise ValueError(
            f"H_wp violated: sup R + ||e||*||a|| = {total:.6f} > 1 - eps = {1.0 - eps:.6f}. "
            "Reduce reflectance or fluorescence strength."
        )


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
