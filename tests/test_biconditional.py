"""Biconditional well-posedness conditions -- verification of prewriteup Part VI / F4.

The consolidated reference doc (.claude/ref/dbr_paper_pre_print_prewriteup.md)
states well-posedness in SUFFICIENT form (C-wp: sup R < 1 and rho(Gamma) < 1).
F4 sharpens it to a biconditional, so far only a sketch:

  For sign-definite a*e >= 0, the solve is well-posed (rho(T) < 1) IFF
    sup R < 1  AND  m(z) := integral a e / (R - z) dlambda  !=  -1  for all |z| >= 1.
  Because m is monotone in z for sign-definite profiles, the free-z sweep
  collapses to a single check at z = 1:  B_fl := integral a e / (1 - R) < 1.
  k species:  det(I_k + M(z)) != 0,  M(z)_{sm} = integral a_s e_m / (R - z);
  at z = 1,   M(1) = -Gamma,  so the check is  det(I_k - Gamma) != 0.

This module verifies all three pieces against the three-way oracle style:

  BC1  Weinstein-Aronszajn rank-1 identity (sympy, symbolic N)   -- the +m(z) sign
  BC2  sign disambiguation: it is 1 + m(z), not 1 - m(z)         (sympy)
  BC3  identity holds at the actual eigenvalues of a discrete T  (torch f64)
  BC4  secular structure: m monotone, one root z*, z* = rho(T)   (torch f64)
  BC5  necessity: well-posed IFF sup R < 1 AND B_fl < 1          (torch f64)
  BC6  Weinstein-Aronszajn k-species identity (sympy, k=2 N=3)
  BC7  det(I_k + M(z)) = 0 at the eigenvalues of a discrete T_k  (torch f64)
  BC8  M(1) = -Gamma, and the Woodbury inverse formula           (torch f64)
  BC9  k-species necessity: rho(Gamma) < 1 IFF rho(T_k) < 1      (torch f64)
  BC10 decoupling: species pulled apart -> det(I_k - Gamma) = prod(1 - Gamma_ss)

Run:
    "C:\\Users\\Dell\\anaconda3\\envs\\Spectral\\python.exe" -m tests.test_biconditional
"""
from __future__ import annotations

import sys

import torch

torch.set_default_dtype(torch.float64)

from tests.harness import Reporter, StructResult


# ---------------------------------------------------------------------------
# Discrete operator helpers
#
# Spectral grid: N wavelengths, quadrature weights w_j > 0, inner product
# <f,g>_w = sum_j w_j f_j g_j.  Transport operator T = M_R + K_x with
#   M_R = diag(R),  R_j in [0, 1)
#   (K_x f)_j = e_j <a, f>_w   ->   K_x = e (w * a)^T   (rank 1, single species)
# For k species  K_x = sum_s e_s (w * a_s)^T = E U^T  with U = (w[:,None] * A).
# ---------------------------------------------------------------------------

def build_T(R: torch.Tensor, a: torch.Tensor, e: torch.Tensor,
            w: torch.Tensor) -> torch.Tensor:
    return torch.diag(R) + torch.outer(e, w * a)


def m_scalar(z: float, R: torch.Tensor, a: torch.Tensor, e: torch.Tensor,
             w: torch.Tensor) -> float:
    return (w * a * e / (R - z)).sum().item()


def build_T_k(R: torch.Tensor, A: torch.Tensor, E: torch.Tensor,
              w: torch.Tensor) -> torch.Tensor:
    # A, E are (k, N);  K_x = sum_s e_s (w * a_s)^T  ->  einsum over species s
    return torch.diag(R) + torch.einsum("si,sj->ij", E, w * A)


def M_of_z(z: complex | float, R: torch.Tensor, A: torch.Tensor,
           E: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # M(z)_{sm} = sum_j w_j a_{s,j} e_{m,j} / (R_j - z)      -> (k, k)
    # z may be complex: eigenvalues of a k-species T_k are not all real.
    if isinstance(z, complex) and z.imag != 0.0:
        Rc = R.to(torch.complex128)
        D = w.to(torch.complex128) / (Rc - z)
        return torch.einsum("j,sj,mj->sm", D, A.to(torch.complex128),
                            E.to(torch.complex128))
    D = w / (R - float(z.real if isinstance(z, complex) else z))
    return torch.einsum("j,sj,mj->sm", D, A, E)


def gamma_matrix(R: torch.Tensor, A: torch.Tensor, E: torch.Tensor,
                 w: torch.Tensor) -> torch.Tensor:
    # Gamma_{sm} = sum_j w_j a_{s,j} e_{m,j} / (1 - R_j)
    D = w / (1.0 - R)
    return torch.einsum("j,sj,mj->sm", D, A, E)


def spectral_radius(M: torch.Tensor) -> float:
    return torch.linalg.eigvals(M).abs().max().item()


def secular_root_scalar(R: torch.Tensor, a: torch.Tensor, e: torch.Tensor,
                        w: torch.Tensor, hi: float = 1e6) -> float:
    """Unique root of g(z) = 1 + m(z) above max(R), by bisection."""
    lo = R.max().item() + 1e-9
    g = lambda z: 1.0 + m_scalar(z, R, a, e, w)
    # g -> -inf at lo+, g -> 1- at hi;  strictly increasing between.
    assert g(lo) < 0.0 < g(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# BC1 / BC2  -- Weinstein-Aronszajn rank-1 identity, symbolic
# ---------------------------------------------------------------------------

def test_BC1_BC2() -> list[StructResult]:
    import sympy as sp

    N = 3
    z = sp.Symbol("z")
    R = sp.symbols(f"R1:{N + 1}", real=True)
    a = sp.symbols(f"a1:{N + 1}", real=True)
    e = sp.symbols(f"e1:{N + 1}", real=True)
    w = sp.symbols(f"w1:{N + 1}", positive=True)

    D = sp.diag(*R)
    Kx = sp.Matrix(N, N, lambda i, j: e[i] * w[j] * a[j])
    T = D + Kx

    p = (T - z * sp.eye(N)).det()

    prod = sp.prod([R[j] - z for j in range(N)])
    m = sum(w[j] * a[j] * e[j] / (R[j] - z) for j in range(N))

    resid_plus = sp.simplify(p - prod * (1 + m))
    resid_minus = sp.simplify(p - prod * (1 - m))

    ok_plus = resid_plus == 0
    # 1 - m must NOT reproduce the determinant (would mean the sign is wrong)
    ok_minus_is_wrong = resid_minus != 0

    return [
        StructResult(
            "BC1", "sympy: det(T - zI) == prod(R_j - z) * (1 + m(z))",
            0.0 if ok_plus else 1.0, 0.0, 0.0 if ok_plus else 1.0, 0.0, ok_plus,
            "Weinstein-Aronszajn for T = diag(R) + e (w*a)^T, N=3 symbolic; "
            f"simplify(det - prod*(1+m)) = {resid_plus}",
        ),
        StructResult(
            "BC2", "sympy: sign is 1 + m(z), NOT 1 - m(z)",
            0.0 if ok_minus_is_wrong else 1.0, 0.0,
            0.0 if ok_minus_is_wrong else 1.0, 0.0, ok_minus_is_wrong,
            f"simplify(det - prod*(1-m)) = {sp.simplify(resid_minus)} (nonzero => "
            "eigen-condition is m(z) = -1, confirming the hand-checked sign)",
        ),
    ]


# ---------------------------------------------------------------------------
# BC3  -- identity holds at the actual eigenvalues of a concrete discrete T
# ---------------------------------------------------------------------------

def _rng(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _rand_scene(N: int, seed: int, r_hi: float = 0.9):
    # R laid on a near-uniform grid with bounded jitter: tight R_j clusters make
    # the interlacing roots sit in tiny gaps where m'(z) is enormous, which
    # amplifies eigvals backward error in the 1 + m(z) residual.
    g = _rng(seed)
    base = torch.linspace(0.02, r_hi, N)
    step = (r_hi - 0.02) / (N - 1)
    R = base + (torch.rand(N, generator=g) - 0.5) * step * 0.6
    a = torch.rand(N, generator=g) * 1.3 + 0.2
    e = torch.rand(N, generator=g) * 1.3 + 0.2
    w = torch.rand(N, generator=g) * 0.5 + 0.5
    return R, a, e, w


def test_BC3() -> list[StructResult]:
    R, a, e, w = _rand_scene(N=40, seed=1)
    T = build_T(R, a, e, w)
    ev = torch.linalg.eigvals(T)

    max_im = ev.imag.abs().max().item()
    zs = ev.real.tolist()

    # residual at every eigenvalue: near tight R_j gaps m'(z) is large, so
    # eigvals backward error (~1e-13 * ||T||) shows up amplified here.
    resid_all = max(abs(1.0 + m_scalar(z, R, a, e, w)) for z in zs)
    # residual at the isolated Perron root z* = rho(T): well separated, no gap
    # amplification, so this is a clean machine-precision check.
    z_star = max(zs)
    resid_star = abs(1.0 + m_scalar(z_star, R, a, e, w))
    mean_minus = sum(abs(1.0 - m_scalar(z, R, a, e, w)) for z in zs) / len(zs)

    return [
        StructResult(
            "BC3", "discrete T eigenvalues are all real (sign-definite rank-1)",
            max_im, 0.0, max_im, 1e-10, max_im < 1e-10,
            "N=40; secular eqn sum c_j/(z-R_j)=1 with c_j>0 has N real "
            "interlacing roots",
        ),
        StructResult(
            "BC3", "|1 + m(z*)| at the isolated Perron root z* = rho(T)",
            resid_star, 0.0, resid_star, 1e-10, resid_star < 1e-10,
            "clean machine-precision check away from interlacing gaps",
        ),
        StructResult(
            "BC3", "max |1 + m(z)| over ALL eigenvalues (gap-amplified)",
            resid_all, 0.0, resid_all, 1e-6, resid_all < 1e-6,
            "eigvals backward error amplified by large m'(z) in tight R_j gaps",
        ),
        StructResult(
            "BC3", "mean |1 - m(z)| over eigenvalues (sign check, expect ~2)",
            mean_minus, 2.0, abs(mean_minus - 2.0) / 2.0, 0.15,
            abs(mean_minus - 2.0) / 2.0 < 0.15,
            "1 - m(z) ~ 2 at the roots => the vanishing combination is 1 + m(z)",
        ),
    ]


# ---------------------------------------------------------------------------
# BC4  -- secular structure: m monotone above max R, single root, root = rho(T)
# ---------------------------------------------------------------------------

def test_BC4() -> list[StructResult]:
    R, a, e, w = _rand_scene(N=40, seed=2)
    T = build_T(R, a, e, w)

    z0 = R.max().item()
    z_star = secular_root_scalar(R, a, e, w)
    zs = torch.linspace(z0 + 1e-6, z0 + 1.6 * (z_star - z0), 600).tolist()
    mvals = [m_scalar(z, R, a, e, w) for z in zs]

    diffs = [mvals[i + 1] - mvals[i] for i in range(len(mvals) - 1)]
    monotone_viol = sum(1 for d in diffs if d <= 0)

    gvals = [1.0 + mv for mv in mvals]
    sign_changes = sum(1 for i in range(len(gvals) - 1)
                       if gvals[i] * gvals[i + 1] < 0.0)

    m_at_far = m_scalar(z0 + 1e5, R, a, e, w)  # -> 0^-

    rho_T = spectral_radius(T)
    rel_root = abs(z_star - rho_T) / rho_T

    return [
        StructResult(
            "BC4", "m(z) strictly increasing on (max R, inf): monotonicity violations",
            float(monotone_viol), 0.0, float(monotone_viol), 0.0, monotone_viol == 0,
            "m'(z) = sum w a e /(R-z)^2 > 0 for sign-definite profiles",
        ),
        StructResult(
            "BC4", "g(z) = 1 + m(z) has exactly one sign change above max R",
            float(sign_changes), 1.0, abs(sign_changes - 1.0), 0.0, sign_changes == 1,
            "single root z* => the |z|>=1 sweep collapses to one check",
        ),
        StructResult(
            "BC4", "m(z) -> 0^- as z -> inf",
            m_at_far, 0.0, abs(m_at_far), 1e-3, -1e-3 < m_at_far < 0.0,
            f"m(max R + 1e5) = {m_at_far:.3e} (negative, small)",
        ),
        StructResult(
            "BC4", "secular root z* equals rho(T)",
            z_star, rho_T, rel_root, 1e-9, rel_root < 1e-9,
            f"z* (bisection on 1+m=0) vs max|eig(T)|; both = {rho_T:.6f}",
        ),
    ]


# ---------------------------------------------------------------------------
# BC5  -- necessity: rho(T) < 1  IFF  sup R < 1 AND B_fl < 1
# ---------------------------------------------------------------------------

def _scene_with_Bfl(N: int, seed: int, sup_R: float, target_Bfl: float):
    """Random scene, R rescaled to a fixed supremum, a rescaled so that
    B_fl = sum w a e /(1 - R) hits target_Bfl exactly."""
    g = _rng(seed)
    R = torch.sort(torch.rand(N, generator=g)).values
    R = R / R.max() * sup_R
    a = torch.rand(N, generator=g) * 1.0 + 0.3
    e = torch.rand(N, generator=g) * 1.0 + 0.3
    w = torch.rand(N, generator=g) * 0.5 + 0.5
    Bfl0 = (w * a * e / (1.0 - R)).sum().item()
    a = a * (target_Bfl / Bfl0)
    return R, a, e, w


def _neumann_growth(T: torch.Tensor, n: int = 60) -> float:
    """||T^n|| ratio over the last few powers -- >1 means the series diverges."""
    P = torch.eye(T.shape[0])
    norms = []
    for _ in range(n):
        P = P @ T
        norms.append(torch.linalg.matrix_norm(P, 2).item())
    return norms[-1] / norms[-5]


def test_BC5() -> list[StructResult]:
    out = []

    # (a) well posed: sup R < 1 AND B_fl < 1
    R, a, e, w = _scene_with_Bfl(40, 10, sup_R=0.80, target_Bfl=0.50)
    T = build_T(R, a, e, w)
    rho = spectral_radius(T)
    growth = _neumann_growth(T)
    out.append(StructResult(
        "BC5", "well posed (sup R=0.80, B_fl=0.50): rho(T) < 1",
        rho, None, None, 1.0, rho < 1.0,
        f"rho(T) = {rho:.4f}; ||T^n|| ratio (last 5 powers) = {growth:.3e} < 1",
    ))
    out.append(StructResult(
        "BC5", "well posed: Neumann series contracts",
        growth, 0.0, growth, 1.0, growth < 1.0, "||T^n|| decreasing",
    ))

    # (b) ill posed by fluorescence: sup R < 1 but B_fl > 1
    #     -- the sufficient ||e|| ||a|| bound is beside the point; B_fl is the
    #        sharp quantity (this is the A12 complement).
    R, a, e, w = _scene_with_Bfl(40, 11, sup_R=0.80, target_Bfl=1.50)
    T = build_T(R, a, e, w)
    rho = spectral_radius(T)
    z_star = secular_root_scalar(R, a, e, w)
    growth = _neumann_growth(T)
    Bfl = (w * a * e / (1.0 - R)).sum().item()
    out.append(StructResult(
        "BC5", "ill posed (sup R=0.80 < 1, B_fl=1.50 > 1): rho(T) > 1",
        rho, None, None, None, rho > 1.0,
        f"rho(T) = {rho:.4f} > 1 despite sup R < 1; z* = {z_star:.4f}; "
        f"||T^n|| ratio = {growth:.3e} > 1 (Neumann diverges)",
    ))
    out.append(StructResult(
        "BC5", "ill posed: secular root z* matches rho(T)",
        z_star, rho, abs(z_star - rho) / rho, 1e-8, abs(z_star - rho) / rho < 1e-8,
        f"B_fl (recomputed) = {Bfl:.4f}",
    ))

    # (c) boundary: B_fl = 1 exactly  ->  z* = 1  ->  (I - T) singular
    R, a, e, w = _scene_with_Bfl(40, 12, sup_R=0.80, target_Bfl=1.0)
    T = build_T(R, a, e, w)
    z_star = secular_root_scalar(R, a, e, w)
    sv_min = torch.linalg.svdvals(torch.eye(40) - T).min().item()
    out.append(StructResult(
        "BC5", "boundary B_fl = 1: z* = 1",
        z_star, 1.0, abs(z_star - 1.0), 1e-6, abs(z_star - 1.0) < 1e-6,
        "the biconditional's sharp threshold",
    ))
    out.append(StructResult(
        "BC5", "boundary B_fl = 1: (I - T) genuinely singular",
        sv_min, 0.0, sv_min, 1e-6, sv_min < 1e-6,
        f"sigma_min(I - T) = {sv_min:.3e}",
    ))
    return out


# ---------------------------------------------------------------------------
# BC6  -- Weinstein-Aronszajn k-species identity, symbolic
# ---------------------------------------------------------------------------

def test_BC6() -> StructResult:
    import sympy as sp

    N, k = 3, 2
    z = sp.Symbol("z")
    R = sp.symbols(f"R1:{N + 1}", real=True)
    w = sp.symbols(f"w1:{N + 1}", positive=True)
    A = sp.Matrix(k, N, lambda s, j: sp.Symbol(f"a{s}_{j}", real=True))
    E = sp.Matrix(k, N, lambda s, j: sp.Symbol(f"e{s}_{j}", real=True))

    D = sp.diag(*R)
    Kx = sp.zeros(N, N)
    for s in range(k):
        Kx += sp.Matrix(N, N, lambda i, j: E[s, i] * w[j] * A[s, j])
    T = D + Kx

    p = (T - z * sp.eye(N)).det()

    prod = sp.prod([R[j] - z for j in range(N)])
    M = sp.Matrix(k, k, lambda s, t: sum(
        w[j] * A[s, j] * E[t, j] / (R[j] - z) for j in range(N)))
    target = prod * (sp.eye(k) + M).det()

    resid = sp.simplify(p - target)
    ok = resid == 0
    return StructResult(
        "BC6", "sympy: det(T_k - zI) == prod(R_j - z) * det(I_k + M(z))",
        0.0 if ok else 1.0, 0.0, 0.0 if ok else 1.0, 0.0, ok,
        f"k=2, N=3 symbolic; simplify(det - prod*det(I_k + M)) = {resid}",
    )


# ---------------------------------------------------------------------------
# BC7  -- det(I_k + M(z)) = 0 at the eigenvalues of a concrete discrete T_k
# ---------------------------------------------------------------------------

def _rand_scene_k(N: int, k: int, seed: int, r_hi: float = 0.85,
                  centers: list[float] | None = None,
                  floor: float = 0.05, width: float = 0.10):
    g = _rng(seed)
    R = torch.sort(torch.rand(N, generator=g) * r_hi + 0.02).values
    idx = torch.linspace(0, 1, N)
    if centers is None:
        centers = [0.5 + 0.9 * (s - (k - 1) / 2) / max(k, 1) for s in range(k)]
    # floor > 0 keeps genuine inter-species coupling (Gamma dense); floor = 0
    # (pure Gaussians) lets well-separated species decouple -- see BC10.
    A = torch.stack([
        floor + torch.exp(-0.5 * ((idx - c) / width) ** 2) for c in centers])
    E = torch.stack([
        floor + torch.exp(-0.5 * ((idx - (c + 0.06)) / width) ** 2)
        for c in centers])
    w = torch.rand(N, generator=g) * 0.5 + 0.5
    return R, A, E, w


def test_BC7() -> list[StructResult]:
    # Narrow, asymmetrically-placed species: M(z) is non-symmetric enough that
    # the k-species secular factor of det(T_k - zI) has a complex-conjugate
    # root pair -- this is why the condition is stated over |z| >= 1 in the
    # complex plane, not just at z = 1.
    R, A, E, w = _rand_scene_k(N=48, k=3, seed=0,
                               centers=[0.3, 0.5, 0.7], width=0.05)
    T = build_T_k(R, A, E, w)
    ev = torch.linalg.eigvals(T)
    k = A.shape[0]
    Ik = torch.eye(k)
    Ik_c = Ik.to(torch.complex128)

    # T_k = diag(R) + E (w*A)^T is entrywise nonnegative, so by Perron-Frobenius
    # rho(T_k) is itself an eigenvalue (real, nonnegative) -- even though the
    # other k-species eigenvalues are genuinely complex.
    n_complex = int((ev.imag.abs() > 1e-9).sum().item())
    perron = ev[ev.abs().argmax()]
    perron_im = perron.imag.abs().item()

    worst = 0.0
    for zc in ev.tolist():
        Mz = M_of_z(zc, R, A, E, w)
        base = Ik_c if isinstance(zc, complex) and zc.imag != 0.0 else Ik
        worst = max(worst, torch.linalg.svdvals(base + Mz).min().item())

    return [
        StructResult(
            "BC7", "T_k has genuinely complex eigenvalues for k>=2",
            float(n_complex), None, None, None, n_complex > 0,
            f"{n_complex}/{2 * k + 1} eigenvalues off the real axis -- why the "
            "condition is stated for |z| >= 1 in C, not just z = 1",
        ),
        StructResult(
            "BC7", "Perron root rho(T_k) is real (nonnegative matrix)",
            perron_im, 0.0, perron_im, 1e-9, perron_im < 1e-9,
            f"rho(T_k) = {perron.real.item():.4f}",
        ),
        StructResult(
            "BC7", "max sigma_min(I_k + M(z)) over ALL eigenvalues z of T_k (z in C)",
            worst, 0.0, worst, 1e-8, worst < 1e-8,
            "det(I_k + M(z)) = 0 at every eigenvalue, complex ones included",
        ),
    ]


# ---------------------------------------------------------------------------
# BC8  -- M(1) = -Gamma, and the Woodbury inverse formula
# ---------------------------------------------------------------------------

def test_BC8() -> list[StructResult]:
    out = []
    R, A, E, w = _rand_scene_k(N=48, k=3, seed=21)
    k = A.shape[0]
    Ik = torch.eye(k)

    M1 = M_of_z(1.0, R, A, E, w)
    Gam = gamma_matrix(R, A, E, w)
    diff = (M1 + Gam).abs().max().item()
    out.append(StructResult(
        "BC8", "M(1) == -Gamma",
        diff, 0.0, diff, 1e-13, diff < 1e-13,
        "R - 1 = -(1 - R) folds the free-z condition into the Woodbury bracket",
    ))

    det_M1 = torch.linalg.det(Ik + M1).item()
    det_Gam = torch.linalg.det(Ik - Gam).item()
    rd = abs(det_M1 - det_Gam) / max(abs(det_Gam), 1e-30)
    out.append(StructResult(
        "BC8", "det(I_k + M(1)) == det(I_k - Gamma)",
        rd, 0.0, rd, 1e-12, rd < 1e-12,
        f"both = {det_Gam:.6f}",
    ))

    # Woodbury inverse on a well-posed scene:
    #   (I - T)^-1 = S + S E (I_k - Gamma)^-1 (w*A)^T S,   S = (I - M_R)^-1 = diag(1/(1-R))
    N = R.shape[0]
    T = build_T_k(R, A, E, w)
    rho = spectral_radius(T)
    S = torch.diag(1.0 / (1.0 - R))
    U = (w * A).T                      # (N, k)
    Emat = E.T                          # (N, k)
    wood = S + S @ Emat @ torch.linalg.solve(Ik - Gam, U.T @ S)
    direct = torch.linalg.inv(torch.eye(N) - T)
    rel = torch.linalg.matrix_norm(wood - direct, 2).item() / \
        torch.linalg.matrix_norm(direct, 2).item()
    out.append(StructResult(
        "BC8", "Woodbury (I-T)^-1 formula vs direct inverse (well-posed scene)",
        rel, 0.0, rel, 1e-10, rel < 1e-10,
        f"rho(T) = {rho:.4f}; relative 2-norm error {rel:.2e}",
    ))
    return out


# ---------------------------------------------------------------------------
# BC9  -- k-species necessity: rho(Gamma) < 1  IFF  rho(T_k) < 1
# ---------------------------------------------------------------------------

def _rescale_to_rho_gamma(A: torch.Tensor, R: torch.Tensor, E: torch.Tensor,
                          w: torch.Tensor, target: float) -> torch.Tensor:
    Gam = gamma_matrix(R, A, E, w)
    return A * (target / spectral_radius(Gam))


def test_BC9() -> list[StructResult]:
    out = []
    R, A, E, w = _rand_scene_k(N=48, k=3, seed=22, r_hi=0.80)

    Ik = torch.eye(3)
    for tag, target in (("well posed", 0.55), ("ill posed", 1.60)):
        A_s = _rescale_to_rho_gamma(A, R, E, w, target)
        Gam = gamma_matrix(R, A_s, E, w)
        T = build_T_k(R, A_s, E, w)
        rho_g = spectral_radius(Gam)
        rho_t = spectral_radius(T)                       # Perron root (real, P-F)
        both_side = (rho_g < 1.0) == (rho_t < 1.0)
        # identity at the Perron root: det(I_k + M(rho(T_k))) = 0
        det_at_perron = abs(torch.linalg.det(Ik + M_of_z(rho_t, R, A_s, E, w)).item())
        out.append(StructResult(
            "BC9", f"{tag}: sign(rho(Gamma) - 1) == sign(rho(T_k) - 1)",
            0.0 if both_side else 1.0, 0.0, 0.0 if both_side else 1.0, 0.0, both_side,
            f"rho(Gamma) = {rho_g:.4f}, rho(T_k) = {rho_t:.4f}, sup R = "
            f"{R.max().item():.3f} (well-posedness IFF rho(Gamma) < 1)",
        ))
        out.append(StructResult(
            "BC9", f"{tag}: |det(I_k + M(z))| = 0 at z = rho(T_k)",
            det_at_perron, 0.0, det_at_perron, 1e-8, det_at_perron < 1e-8,
            f"z = {rho_t:.6f}",
        ))
    return out


# ---------------------------------------------------------------------------
# BC10  -- decoupling: species pulled apart -> det(I_k - Gamma) = prod(1 - Gamma_ss)
# ---------------------------------------------------------------------------

def test_BC10() -> list[StructResult]:
    # k=3 species with far-apart, non-overlapping bands and NO baseline floor,
    # so Gamma collapses to diagonal and the biconditional decouples.
    R, A, E, w = _rand_scene_k(N=90, k=3, seed=30, r_hi=0.75,
                               centers=[0.12, 0.5, 0.88], floor=0.0, width=0.06)
    Gam = gamma_matrix(R, A, E, w)
    k = Gam.shape[0]

    offdiag = (Gam - torch.diag(torch.diagonal(Gam))).abs().max().item()
    diag_min = torch.diagonal(Gam).abs().min().item()
    ratio = offdiag / diag_min

    det_full = torch.linalg.det(torch.eye(k) - Gam).item()
    det_diag = torch.prod(1.0 - torch.diagonal(Gam)).item()
    rel = abs(det_full - det_diag) / abs(det_diag)

    return [
        StructResult(
            "BC10", "well-separated species: Gamma off-diagonal / diagonal ratio",
            ratio, 0.0, ratio, 5e-3, ratio < 5e-3,
            f"max |off-diag| = {offdiag:.2e}, min |diag| = {diag_min:.2e}",
        ),
        StructResult(
            "BC10", "det(I_k - Gamma) approx prod_s (1 - Gamma_ss)",
            rel, 0.0, rel, 5e-3, rel < 5e-3,
            "biconditional decouples into k independent rank-1 statements "
            "1 - B_fl,s < 1 (V13 negative-control regime)",
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL = [
    test_BC1_BC2, test_BC3, test_BC4, test_BC5,
    test_BC6, test_BC7, test_BC8, test_BC9, test_BC10,
]


def main() -> None:
    rep = Reporter()
    for fn in ALL:
        try:
            result = fn()
            if isinstance(result, list):
                for r in result:
                    rep.add(r)
            else:
                rep.add(result)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {e}", file=sys.stderr)
            raise
    rep.print_struct_table()
    sys.exit(0 if rep.all_passed() else 1)


if __name__ == "__main__":
    main()
