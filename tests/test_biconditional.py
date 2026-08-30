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

F4 remainder (2026-08-30):

  BC11 reabsorption loop A->B->A stays rank-k; Woodbury solve is EXACT vs a
       brute-force dense solve of (I - T_k) L = L_e on the propagating set
  BC12 the 2-cycle is the Gamma_AB Gamma_BA term of (I_k - Gamma)^-1: it sits
       in (Gamma^2)_AA and in det(I_2 - Gamma)
  BC13 cycle-sourced ill-posedness: a PURE off-diagonal cycle (Gamma_AA =
       Gamma_BB = 0) has rho(Gamma) = sqrt(Gamma_AB Gamma_BA), which crosses 1
       and drives rho(T_k) past 1 (biconditional, BC9, sourced by the loop);
       breaking one leg kills it
  BC14 Fredholm "only if" shadow: sup R -> 1 (K_x = 0) makes a CLUSTER of
       (I - T) singular values collapse (count ~ f*N, grows with the grid) --
       1 entering the essential spectrum; a fluorescence-tuned B_fl = 1 with
       R bounded away from 1 collapses EXACTLY ONE (discrete eigenvalue, still
       Fredholm index 0)
  BC15 Vitali "only if" by counterexample: for a moving-boundary integral with
       a non-L1 endpoint singularity, the substituted (uniformly integrable)
       fixed-domain form has autograd == FD, while the raw naive interior-only
       term diverges -- AD = I' exactly where uniform integrability holds

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


# ===========================================================================
# F4 remainder: reabsorption loop (BC11-BC13), Fredholm "only if" (BC14),
# Vitali "only if" (BC15).
# ===========================================================================

def _cycle_scene_k2(N=140, seed=40, r_hi=0.70, qy=(0.9, 0.9),
                    a_ctr=(0.35, 0.55), e_ctr=(0.55, 0.35), width=0.05,
                    L0=1.0):
    """k=2 scene with a genuine A->B->A coupling cycle.

    a_A overlaps e_B  (species A reabsorbs B's emission -> Gamma_AB != 0) and
    a_B overlaps e_A  (species B reabsorbs A's emission -> Gamma_BA != 0).
    All R < r_hi < 1, so every wavelength is propagating and (1 - R) G0 = 1
    holds exactly: the rank-k Woodbury reduction is then EXACT, not an
    approximation. (An anti-Stokes leg is unphysical for real dyes; this is an
    operator-algebra test, profiles placed to realize the 2-cycle.)
    """
    g = _rng(seed)
    R = torch.sort(torch.rand(N, generator=g) * r_hi + 0.02).values
    w = torch.rand(N, generator=g) * 0.5 + 0.5
    idx = torch.linspace(0.0, 1.0, N)
    A = torch.stack([torch.exp(-0.5 * ((idx - c) / width) ** 2) for c in a_ctr])
    E = torch.stack([torch.exp(-0.5 * ((idx - c) / width) ** 2) for c in e_ctr])
    L_e = torch.full((N,), float(L0))
    return R, A, E, w, torch.tensor(qy, dtype=torch.float64), L_e


def _gamma_b_cycle(R, A, E, w, qy, L_e=None):
    D = w / (1.0 - R)
    Gam = qy[:, None] * torch.einsum("j,sj,mj->sm", D, A, E)
    if L_e is None:
        return Gam, None
    b = qy * torch.einsum("j,sj,j->s", D, A, L_e)
    return Gam, b


def _rescale_qy_to_rho(R, A, E, w, qy, target):
    """Rescale qy so spectral_radius(Gamma) == target (Gamma is linear in qy
    only when qy is uniform, which it is in these cycle scenes)."""
    Gam, _ = _gamma_b_cycle(R, A, E, w, qy)
    return qy * (target / spectral_radius(Gam))


def _build_Tk_cycle(R, A, E, w, qy):
    return torch.diag(R) + torch.einsum("si,sj->ij", E, (qy[:, None] * w * A))


def test_BC11() -> list[StructResult]:
    R, A, E, w, qy0, L_e = _cycle_scene_k2()
    qy = _rescale_qy_to_rho(R, A, E, w, qy0, 0.7)   # well-posed cycle
    k, N = A.shape
    Gam, b = _gamma_b_cycle(R, A, E, w, qy, L_e)
    Tk = _build_Tk_cycle(R, A, E, w, qy)

    # rank of K_x = T_k - diag(R): must be exactly k despite the cycle
    kx_rank = torch.linalg.matrix_rank(Tk - torch.diag(R), tol=1e-9).item()

    # Woodbury reduced solve vs brute-force dense solve on the full grid
    s_wood = torch.linalg.solve(torch.eye(k) - Gam, b)
    L_brute = torch.linalg.solve(torch.eye(N) - Tk, L_e)
    s_brute = qy * torch.einsum("sj,j,j->s", A, w, L_brute)
    rel = (s_wood - s_brute).abs().max().item() / s_brute.abs().max().item()

    cycle_live = min(Gam[0, 1].item(), Gam[1, 0].item()) > 1e-6   # both legs

    return [
        StructResult(
            "BC11", "reabsorption loop: rank(K_x) == k exactly (no new rank)",
            float(kx_rank), float(k), abs(kx_rank - k), 0.0, kx_rank == k,
            "every re-emission is into a FIXED profile e_j, so K_x stays "
            "sum_j e_j (x) (qy_j a_j) whatever the coupling-graph topology",
        ),
        StructResult(
            "BC11", "A->B->A cycle is live (both Gamma_AB and Gamma_BA nonzero)",
            min(Gam[0, 1].item(), Gam[1, 0].item()), None, None, None, cycle_live,
            f"Gamma = {[[round(x, 4) for x in row] for row in Gam.tolist()]}",
        ),
        StructResult(
            "BC11", "Woodbury s = (I_k - Gamma)^-1 b matches brute-force dense solve",
            rel, 0.0, rel, 1e-12, rel < 1e-12,
            "(1 - R) G0 = 1 on the all-propagating grid makes the reduction exact",
        ),
    ]


def test_BC12() -> list[StructResult]:
    R, A, E, w, qy0, L_e = _cycle_scene_k2()
    qy = _rescale_qy_to_rho(R, A, E, w, qy0, 0.7)
    Gam, _ = _gamma_b_cycle(R, A, E, w, qy)
    gAA, gAB = Gam[0, 0].item(), Gam[0, 1].item()
    gBA, gBB = Gam[1, 0].item(), Gam[1, 1].item()

    # the 2-cycle term lives in (Gamma^2)_AA
    g2_AA = (Gam @ Gam)[0, 0].item()
    pred = gAA ** 2 + gAB * gBA
    rel_g2 = abs(g2_AA - pred) / abs(pred)

    # ... and in det(I_2 - Gamma)
    det_full = torch.linalg.det(torch.eye(2) - Gam).item()
    det_nocycle = (1.0 - gAA) * (1.0 - gBB)          # drop Gamma_AB Gamma_BA
    cycle_contrib = det_full - det_nocycle
    rel_det = abs(cycle_contrib - (-gAB * gBA)) / abs(gAB * gBA)

    # Neumann sum of Gamma^n reproduces (I_2 - Gamma)^-1 (rho(Gamma) < 1)
    rho_g = spectral_radius(Gam)
    P = torch.eye(2)
    S = torch.zeros(2, 2)
    for _ in range(400):
        S = S + P
        P = P @ Gam
    rel_neu = (S - torch.linalg.inv(torch.eye(2) - Gam)).abs().max().item()

    return [
        StructResult(
            "BC12", "(Gamma^2)_AA == Gamma_AA^2 + Gamma_AB Gamma_BA (the 2-cycle path)",
            rel_g2, 0.0, rel_g2, 1e-12, rel_g2 < 1e-12,
            f"A absorbs B absorbs A: Gamma_AB Gamma_BA = {gAB * gBA:.4e}",
        ),
        StructResult(
            "BC12", "det(I_2 - Gamma) cycle contribution == -Gamma_AB Gamma_BA",
            rel_det, 0.0, rel_det, 1e-10, rel_det < 1e-10,
            f"det with cycle {det_full:.5f} vs without {det_nocycle:.5f}",
        ),
        StructResult(
            "BC12", "Neumann sum_n Gamma^n == (I_2 - Gamma)^-1 (rho(Gamma) < 1)",
            rel_neu, 0.0, rel_neu, 1e-10, rel_neu < 1e-10,
            f"rho(Gamma) = {rho_g:.4f}; the loop is all walks through the 2-cycle",
        ),
    ]


def test_BC13() -> list[StructResult]:
    out = []
    # PURE off-diagonal cycle: species do not self-absorb (a_A misses e_A,
    # a_B misses e_B), so Gamma_AA = Gamma_BB ~ 0 and rho(Gamma) is entirely
    # the cycle:  eig([[0, g_AB], [g_BA, 0]]) = +- sqrt(g_AB g_BA).
    base = dict(N=160, seed=41, r_hi=0.65, width=0.04,
                a_ctr=(0.30, 0.62), e_ctr=(0.62, 0.30))

    def scene(qy_scale):
        return _cycle_scene_k2(qy=(0.95 * qy_scale, 0.95 * qy_scale), **base)

    # calibrate qy_scale so rho(Gamma) hits a target (rho ~ qy_scale here)
    R, A, E, w, qy, L_e = scene(1.0)
    Gam1, _ = _gamma_b_cycle(R, A, E, w, qy)
    rho1 = spectral_radius(Gam1)
    diag_frac = torch.diagonal(Gam1).abs().max().item() / rho1
    out.append(StructResult(
        "BC13", "pure cycle: Gamma_AA, Gamma_BB negligible vs rho(Gamma)",
        diag_frac, 0.0, diag_frac, 5e-3, diag_frac < 5e-3,
        f"diag(Gamma) = {torch.diagonal(Gam1).tolist()}, rho(Gamma) = {rho1:.4f}",
    ))
    gAB, gBA = Gam1[0, 1].item(), Gam1[1, 0].item()
    rel_sqrt = abs(rho1 - (gAB * gBA) ** 0.5) / rho1
    out.append(StructResult(
        "BC13", "pure cycle: rho(Gamma) == sqrt(Gamma_AB Gamma_BA)",
        rel_sqrt, 0.0, rel_sqrt, 1e-6, rel_sqrt < 1e-6,
        f"sqrt({gAB:.4e} * {gBA:.4e}) = {(gAB * gBA) ** 0.5:.4f}",
    ))

    for tag, target in (("well posed", 0.6), ("ill posed", 1.5)):
        sc = target / rho1
        R, A, E, w, qy, L_e = scene(sc)
        Gam, b = _gamma_b_cycle(R, A, E, w, qy, L_e)
        Tk = _build_Tk_cycle(R, A, E, w, qy)
        rho_g = spectral_radius(Gam)
        rho_t = spectral_radius(Tk)
        both = (rho_g < 1.0) == (rho_t < 1.0)
        det_perron = abs(torch.linalg.det(
            torch.eye(2) + _cycle_M_of_z(rho_t, R, A, E, w, qy)).item())
        out.append(StructResult(
            "BC13", f"{tag}: sign(rho(Gamma) - 1) == sign(rho(T_k) - 1), cycle-sourced",
            0.0 if both else 1.0, 0.0, 0.0 if both else 1.0, 0.0, both,
            f"rho(Gamma) = {rho_g:.4f}, rho(T_k) = {rho_t:.4f}, sup R = "
            f"{R.max().item():.3f} -- ill-posedness here comes ONLY from the loop",
        ))
        out.append(StructResult(
            "BC13", f"{tag}: |det(I_2 + M(z))| = 0 at z = rho(T_k)",
            det_perron, 0.0, det_perron, 1e-7, det_perron < 1e-7,
            f"z = {rho_t:.6f}",
        ))

    # break one leg: move a_B off e_A -> Gamma_BA ~ 0 -> rho(Gamma) collapses
    broke = dict(base)
    broke["a_ctr"] = (0.30, 0.88)                 # a_B now far from e_A @ 0.62
    R, A, E, w, qy, L_e = _cycle_scene_k2(qy=(0.95 * 1.5 / rho1, 0.95 * 1.5 / rho1),
                                          **broke)
    Gam, _ = _gamma_b_cycle(R, A, E, w, qy)
    rho_broken = spectral_radius(Gam)
    still_has_AB = Gam[0, 1].item() > 1e-3
    out.append(StructResult(
        "BC13", "break one leg (a_B off e_A): rho(Gamma) collapses below 1",
        rho_broken, None, None, None, rho_broken < 1.0 and still_has_AB,
        f"rho(Gamma) {rho1 * (1.5 / rho1):.2f} -> {rho_broken:.4f} at the SAME qy "
        f"scaling; Gamma_AB still {Gam[0, 1].item():.3e} -- it is the CYCLE, not "
        "one leg, that breaks well-posedness",
    ))
    return out


def _cycle_M_of_z(z, R, A, E, w, qy):
    # M(z)_{sm} for the qy-folded k-species T_k = diag(R) + sum_s e_s (qy_s w a_s)^T
    D = (qy[:, None] * w) / (R - z)
    return torch.einsum("sj,mj->sm", D * A, E)


def test_BC14() -> list[StructResult]:
    out = []
    tol = 5e-2

    # Case A -- 1 entering the ESSENTIAL spectrum: K_x = 0, sup R -> 1.
    # I - T = diag(1 - R); a whole BAND of eigenvalues collapses toward 0, and
    # the near-null count scales ~linearly with N (a continuum, not a point).
    delta = 2e-2
    counts_A = []
    for N in (100, 200, 400):
        R = torch.linspace(0.10, 1.0 - delta, N)
        sv = torch.linalg.svdvals(torch.eye(N) - torch.diag(R))
        counts_A.append(int((sv < 5.0 * delta).sum().item()))
    ratios_A = [counts_A[i + 1] / counts_A[i] for i in range(len(counts_A) - 1)]
    scales_with_N = all(1.7 < r < 2.3 for r in ratios_A)
    out.append(StructResult(
        "BC14", "essential-spectrum shadow: near-null count scales ~linearly with N",
        sum(ratios_A) / len(ratios_A), 2.0,
        abs(sum(ratios_A) / len(ratios_A) - 2.0) / 2.0, 0.15, scales_with_N,
        f"count at N=100,200,400 = {counts_A} (per-doubling ratios "
        f"{[f'{r:.2f}' for r in ratios_A]} ~ 2) -- a CONTINUUM of near-eigenvalues "
        "at 1 (I - T not Fredholm)",
    ))

    # Case B -- 1 as a DISCRETE eigenvalue: R bounded away from 1, fluorescence
    # tuned to B_fl = 1. Exactly one singular value collapses, at every N.
    counts_B = []
    for i, N in enumerate((100, 200, 400)):
        R, a, e, w = _scene_with_Bfl(N, 50 + i, sup_R=0.85, target_Bfl=1.0)
        sv = torch.linalg.svdvals(torch.eye(N) - build_T(R, a, e, w))
        counts_B.append(int((sv < 1e-6).sum().item()))
    out.append(StructResult(
        "BC14", "discrete eigenvalue at 1: EXACTLY ONE near-null s.v. at every N",
        float(max(counts_B)), 1.0, float(max(counts_B) - 1), 0.0,
        all(c == 1 for c in counts_B),
        f"count at N=100,200,400 = {counts_B} -- 1-dim kernel, I - T still "
        "Fredholm index 0 (contrast Case A)",
    ))
    return out


def test_BC15() -> list[StructResult]:
    from src.gradient import gauss_legendre_01

    g_fn = lambda x: 1.0 + x ** 2 + 0.3 * x ** 3          # smooth, nontrivial
    theta = 0.30

    # Moving-boundary integral  I(theta) = int_theta^1 g(x) / sqrt(x - theta) dx.
    # Substitute w = sqrt(x - theta):  I(theta) = 2 int_0^sqrt(1-theta) g(theta + w^2) dw.
    nodes, wts = gauss_legendre_01(96)

    def I_sub(th):
        W = torch.sqrt(1.0 - th)
        wq = W * nodes
        return 2.0 * (wts * W * g_fn(th + wq ** 2)).sum()

    th_t = torch.tensor(theta, requires_grad=True)
    I_sub(th_t).backward()
    ad_sub = th_t.grad.item()

    h = 1e-6
    with torch.no_grad():
        fd = ((I_sub(torch.tensor(theta + h)) - I_sub(torch.tensor(theta - h)))
              / (2.0 * h)).item()
    rel_sub = abs(ad_sub - fd) / abs(fd)

    # Raw naive moving-domain reverse-mode = interior term only:
    #   int_theta^1 d/dtheta [ g(x)/sqrt(x - theta) ] dx
    #     = int_theta^1 g(x) * 0.5 (x - theta)^{-3/2} dx      -- NOT in L^1.
    def naive_interior(eps):
        xs = torch.linspace(theta + eps, 1.0, 200000)
        integ = g_fn(xs) * 0.5 * (xs - theta) ** (-1.5)
        return torch.trapz(integ, xs).item()

    eps_seq = (1e-2, 1e-3, 1e-4, 1e-5)
    naive_seq = [naive_interior(e) for e in eps_seq]
    naive_ratios = [naive_seq[i + 1] / naive_seq[i] for i in range(len(naive_seq) - 1)]
    naive_diverges = all(r > 2.0 for r in naive_ratios)

    return [
        StructResult(
            "BC15", "uniformly integrable (substituted): autograd == true I'(theta)",
            rel_sub, 0.0, rel_sub, 1e-7, rel_sub < 1e-7,
            f"AD {ad_sub:.8f} vs FD {fd:.8f} -- differentiation under the "
            "integral is valid, DCT holds",
        ),
        StructResult(
            "BC15", "NOT uniformly integrable (raw): naive interior term diverges",
            naive_ratios[-1] if naive_ratios else 0.0, None, None, None,
            naive_diverges,
            f"int_{{theta+eps}}^1 dh/dtheta dx at eps=1e-2..1e-5: "
            f"{[f'{x:.1f}' for x in naive_seq]} (ratios {[f'{r:.2f}' for r in naive_ratios]}"
            f", ~sqrt(10)) -- dh/dtheta not in L^1, so naive moving-domain "
            "reverse-mode != I'(theta): the Vitali 'only if' by counterexample",
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL = [
    test_BC1_BC2, test_BC3, test_BC4, test_BC5,
    test_BC6, test_BC7, test_BC8, test_BC9, test_BC10,
    test_BC11, test_BC12, test_BC13, test_BC14, test_BC15,
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
