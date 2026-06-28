import torch


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
    """Raised-cosine band responses with correct FWHM. Returns S (M, N).

    Half-max lands at delta = +/-fwhm/2: S(fwhm/2) = 0.5(1+cos(pi/2)) = 0.5 (correct)
    Support is |δ| ≤ fwhm (total width = 2×fwhm).
    """
    delta = lam.unsqueeze(0) - centers.unsqueeze(1)  # (M, N)
    S = 0.5 * (1.0 + torch.cos(torch.pi * delta / fwhm))
    return torch.where(delta.abs() <= fwhm, S, torch.zeros_like(S))


# ---------------------------------------------------------------------------
# Preset constructors  (all take lam in nm, return Sensor)
# ---------------------------------------------------------------------------

def hyperspectral_fx10(lam: torch.Tensor, M: int = 120) -> Sensor:
    """FX10-like hyperspectral: M raised-cosine channels, FWHM = band spacing."""
    lam_min, lam_max = lam[0].item(), lam[-1].item()
    centers = torch.linspace(lam_min, lam_max, M, dtype=lam.dtype, device=lam.device)
    fwhm = (lam_max - lam_min) / M
    return Sensor(_raised_cosine(lam, centers, fwhm))


