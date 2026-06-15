import torch


def transport_step(T: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Apply the bispectral transport operator once: L' = T @ L.

    T: (N, N) operator matrix (kernel functions return this directly)
    L: (N,)   spectral radiance
    Returns (N,) — one-bounce contribution.
    """
    return T @ L


def neumann_forward(
    T:         torch.Tensor,
    L_e:       torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Primary forward model: truncated Neumann series to fixed bounce depth.

        L = L_e + T L_e + T² L_e + ... + T^max_depth L_e

    Bounce depth is a configurable axis (spec §6.1). Autograd differentiates
    through the loop naturally — this is also the structural model the C++
    adjoint mirrors bounce by bounce.

    T:         (N, N) operator matrix
    L_e:       (N,)   emitted / source radiance
    max_depth: number of transport bounces (D in the plan)
    Returns L  (N,)   total spectral radiance.
    """
    L    = L_e
    term = L_e
    for _ in range(max_depth):
        term = transport_step(T, term)
        L    = L + term
    return L


# ---------------------------------------------------------------------------
# Validation fixture — not the main forward path
# ---------------------------------------------------------------------------

def fredholm_solve_exact(T: torch.Tensor, L_e: torch.Tensor) -> torch.Tensor:
    """Exact Fredholm solve via direct linear system: (I − T) L = L_e.

    VALIDATION FIXTURE ONLY — gives infinite-bounce solution.
    Used once at startup to confirm spectral radius < 1 and that
    neumann_forward converges. Do not use in scenes or gradient tests.
    """
    N = T.shape[0]
    A = torch.eye(N, dtype=T.dtype, device=T.device) - T
    return torch.linalg.solve(A, L_e)
