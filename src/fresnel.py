import torch
from .cauchy_ior import cos_theta_t, is_tir


def fresnel_rs(n_i, n_t, cos_i: torch.Tensor, cos_t: torch.Tensor) -> torch.Tensor:
    """Fresnel amplitude reflectance, s-polarization."""
    return (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t)


def fresnel_rp(n_i, n_t, cos_i: torch.Tensor, cos_t: torch.Tensor) -> torch.Tensor:
    """Fresnel amplitude reflectance, p-polarization."""
    return (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t)


def fresnel_R(
    n_i,
    n_t,
    cos_i: torch.Tensor,
    polarization: str = "unpolarized",
) -> torch.Tensor:
    """Power reflectance R ∈ [0, 1].

    polarization: 's', 'p', or 'unpolarized' (average of s and p).
    TIR wavelengths → R = 1.0 exactly.
    """
    cos_t = cos_theta_t(cos_i, n_i, n_t)
    tir = is_tir(cos_i, n_i, n_t)

    rs = fresnel_rs(n_i, n_t, cos_i, cos_t)
    rp = fresnel_rp(n_i, n_t, cos_i, cos_t)

    if polarization == "s":
        R = rs ** 2
    elif polarization == "p":
        R = rp ** 2
    else:
        R = 0.5 * (rs ** 2 + rp ** 2)

    return torch.where(tir, torch.ones_like(R), R)


def fresnel_T(
    n_i,
    n_t,
    cos_i: torch.Tensor,
    polarization: str = "unpolarized",
) -> torch.Tensor:
    """Power transmittance T = 1 - R (energy conservation).

    TIR wavelengths → T = 0.0 exactly.
    """
    return 1.0 - fresnel_R(n_i, n_t, cos_i, polarization)
