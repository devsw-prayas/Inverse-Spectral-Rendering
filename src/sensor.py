import torch
from pathlib import Path
from .data.cie_tables import cie1931_cmf


class Sensor:
    """Spectral sensor defined by a response matrix S (M, N). Image = S @ L."""

    def __init__(self, S: torch.Tensor):
        self.S = S  # (M, N)

    @property
    def M(self) -> int:
        return self.S.shape[0]

    @property
    def N(self) -> int:
        return self.S.shape[1]

    def measure(self, L: torch.Tensor) -> torch.Tensor:
        """L: (..., N) -> (..., M)"""
        return L @ self.S.T

    @classmethod
    def from_file(cls, path: str | Path) -> "Sensor":
        return cls(torch.load(path, weights_only=True))

    def save(self, path: str | Path):
        torch.save(self.S, path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Pure-torch 1D linear interpolation with edge clamping.

    x:  (N,)
    xp: (M,) sorted
    fp: (M,) or (M, K)
    returns: (N,) or (N, K)
    """
    x_c = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp.contiguous(), x_c.contiguous()) - 1
    idx = idx.clamp(0, xp.shape[0] - 2)
    t = (x_c - xp[idx]) / (xp[idx + 1] - xp[idx])
    if fp.dim() == 1:
        return fp[idx] + t * (fp[idx + 1] - fp[idx])
    return fp[idx] + t.unsqueeze(-1) * (fp[idx + 1] - fp[idx])


def _raised_cosine(lam: torch.Tensor, centers: torch.Tensor, fwhm: float) -> torch.Tensor:
    """Raised-cosine band responses. Returns S (M, N)."""
    delta = lam.unsqueeze(0) - centers.unsqueeze(1)  # (M, N)
    half = fwhm / 2.0
    S = 0.5 * (1.0 + torch.cos(torch.pi * delta / half))
    return torch.where(delta.abs() <= half, S, torch.zeros_like(S))


# ---------------------------------------------------------------------------
# Preset constructors  (all take lam in nm, return Sensor)
# ---------------------------------------------------------------------------

def hyperspectral_fx10(lam: torch.Tensor, M: int = 120) -> Sensor:
    """FX10-like hyperspectral: M raised-cosine channels, FWHM = band spacing."""
    lam_min, lam_max = lam[0].item(), lam[-1].item()
    centers = torch.linspace(lam_min, lam_max, M, dtype=lam.dtype, device=lam.device)
    fwhm = (lam_max - lam_min) / M
    return Sensor(_raised_cosine(lam, centers, fwhm))


def hyperspectral_snapshot(lam: torch.Tensor, M: int = 25, fwhm: float = 15.0) -> Sensor:
    """Snapshot-mosaic: M=25 channels, FWHM≈15 nm (D13 ablation lower bound)."""
    lam_min, lam_max = lam[0].item(), lam[-1].item()
    centers = torch.linspace(lam_min, lam_max, M, dtype=lam.dtype, device=lam.device)
    return Sensor(_raised_cosine(lam, centers, fwhm))


def rgb_cie1931(lam: torch.Tensor, weights: torch.Tensor) -> Sensor:
    """CIE 1931 2° CMF sensor. weights are the λ²Δν̃ quadrature weights (N,).

    S[i, k] = CMF_i(λ_k) · w_k  so that  (S @ L)[i] ≈ ∫ CMF_i(λ) L(λ) dλ.
    """
    lam_tab, cmf_tab = cie1931_cmf(device=lam.device)
    cmf = _torch_interp(lam, lam_tab, cmf_tab)       # (N, 3)
    S = (cmf * weights.unsqueeze(-1)).T               # (3, N)
    return Sensor(S)
