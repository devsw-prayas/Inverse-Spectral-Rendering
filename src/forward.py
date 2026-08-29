import torch


def transport_step(T: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Apply the bispectral transport operator once: L' = T @ L.

    T: (N, N) operator matrix (kernel functions return this directly)
    L: (N,)   spectral radiance
    Returns (N,), the one-bounce contribution.
    """
    return T @ L


def neumann_forward(
    T:         torch.Tensor,
    L_e:       torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Truncated Neumann series to a fixed bounce depth.

        L = L_e + T L_e + T² L_e + ... + T^max_depth L_e

    Autograd differentiates through the loop directly.

    T:         (N, N) operator matrix
    L_e:       (N,)   emitted / source radiance
    max_depth: number of transport bounces
    Returns L  (N,)   total spectral radiance.
    """
    L    = L_e
    term = L_e
    for _ in range(max_depth):
        term = transport_step(T, term)
        L    = L + term
    return L


# ---------------------------------------------------------------------------
# Validation fixture, not the main forward path
# ---------------------------------------------------------------------------

def fredholm_solve_exact(T: torch.Tensor, L_e: torch.Tensor) -> torch.Tensor:
    """Exact solve of (I - T) L = L_e via a direct linear system.

    Validation fixture only: gives the infinite-bounce solution. Used once at
    startup to confirm the spectral radius is below 1 and that neumann_forward
    converges. Not for use in scenes or gradient tests.
    """
    N = T.shape[0]
    A = torch.eye(N, dtype=T.dtype, device=T.device) - T
    return torch.linalg.solve(A, L_e)
