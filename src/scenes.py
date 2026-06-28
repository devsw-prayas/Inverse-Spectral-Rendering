"""Three hardcoded scene archetypes for the Python forward oracle.

All scenes return a ForwardResult containing L, image, and K so that
gradient assembly in gradient.py can reuse them without re-computation.

Scene index
-----------
1  single_bounce_flat       — one thin-film surface, flat illumination
                              Theorem's degenerate case (rank < 6 for joint recovery)
2  two_bounce               — thin film + fluorophore, flat illumination
                              Minimal degeneracy-breaking config (rank 6, D10 baseline)
3  structured_illumination  — archetype 1 or 2 with structured/parameterized L_e
                              (pass the desired L_e directly; no separate function needed)
"""
from dataclasses import dataclass

import torch

from .data.cie_tables import d65_spd
from .forward import neumann_forward
from .kernels import kernel_thinfilm, kernel_fluorescence, check_hwp, fabry_airy_R
from .sensor import Sensor, _torch_interp
from .spectral_grid import SpectralGrid


@dataclass
class ForwardResult:
    L:     torch.Tensor   # (N,) total spectral radiance
    image: torch.Tensor   # (M,) sensor measurements  S @ L
    K:     torch.Tensor   # (N, N) combined kernel (for gradient assembly)


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def d65_on_grid(grid: SpectralGrid) -> torch.Tensor:
    """D65 illuminant resampled onto the ν̃ grid. Returns (N,) float64."""
    lam_tab, spd_tab = d65_spd(device=grid.lam.device)
    return _torch_interp(grid.lam, lam_tab, spd_tab)


# ---------------------------------------------------------------------------
# Archetype 1 — single-bounce flat
# ---------------------------------------------------------------------------

def single_bounce_flat(
    grid:      SpectralGrid,
    sensor:    Sensor,
    *,
    d,
    A,
    B,
    cos_i,
    max_depth: int = 32,
    L_e:       torch.Tensor | None = None,
    polarization: str = "unpolarized",
) -> ForwardResult:
    """One thin-film surface under flat (spectrally uniform angle) illumination.

    d, A, B:   film thickness [nm] and Cauchy coefficients (may require_grad).
    cos_i:     scalar cosine of incidence angle.
    max_depth: bounce depth D — configurable axis per spec §6.1.
    L_e:       (N,) source spectrum; defaults to D65 resampled onto grid.

    This scene is provably rank-deficient for the 6-parameter joint recovery
    (Theorem 7 / D10). Use two_bounce() to break the degeneracy.
    """
    if L_e is None:
        L_e = d65_on_grid(grid)

    with torch.no_grad():
        R = fabry_airy_R(grid.lam, cos_i, d, A, B, polarization)
        check_hwp(R, torch.zeros_like(R), torch.zeros_like(R), grid.weights)

    K = kernel_thinfilm(grid.lam, cos_i, d, A, B, polarization)
    L = neumann_forward(K, L_e, max_depth)
    return ForwardResult(L=L, image=sensor.measure(L), K=K)


# ---------------------------------------------------------------------------
# Archetype 2 — two-bounce: thin film + fluorophore
# ---------------------------------------------------------------------------

def two_bounce(
    grid:      SpectralGrid,
    sensor:    Sensor,
    *,
    d,
    A,
    B,
    cos_i,
    lam_ex,
    lam_em,
    sigma_f,
    max_depth:     int = 32,
    quantum_yield: float = 1.0,
    L_e:           torch.Tensor | None = None,
    polarization:  str = "unpolarized",
) -> ForwardResult:
    """Thin film and fluorophore in the same integrating cavity.

    Combined operator K = K_TF + K_FL. Each Neumann bounce applies both
    materials — K^n includes all interleaved thin-film/fluorophore
    interactions up to depth n. Minimal configuration for rank-6
    identifiability (D10 S8/S10 baseline).

    d, A, B:       thin-film parameters (may require_grad).
    cos_i:         incidence angle cosine.
    lam_ex/em:     fluorophore excitation/emission centres [nm] (may require_grad).
    sigma_f:       fluorescence linewidth [nm] (may require_grad).
    max_depth:     bounce depth D — configurable axis per spec §6.1.
    quantum_yield: fraction of absorbed photons re-emitted (≤ 1).
    L_e:           (N,) source; defaults to D65.
    """
    if L_e is None:
        L_e = d65_on_grid(grid)

    with torch.no_grad():
        R     = fabry_airy_R(grid.lam, cos_i, d, A, B, polarization)
        a_raw = torch.exp(-0.5 * ((grid.lam - lam_ex) / sigma_f) ** 2)
        e_raw = torch.exp(-0.5 * ((grid.lam - lam_em) / sigma_f) ** 2)
        e_n   = e_raw / (e_raw * grid.weights).sum()
        check_hwp(R, e_n, a_raw, grid.weights)

    K_tf = kernel_thinfilm(grid.lam, cos_i, d, A, B, polarization)
    K_fl = kernel_fluorescence(grid.lam, lam_ex, lam_em, sigma_f, grid.weights, quantum_yield)
    K    = K_tf + K_fl
    L    = neumann_forward(K, L_e, max_depth)
    return ForwardResult(L=L, image=sensor.measure(L), K=K)
