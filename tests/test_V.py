"""V-series: verification against the Python forward oracle.

V1-V9 run on the Python oracle (Phase 1).
V10-V12 require the C++ path tracer (Phase 2) -- stubbed here for completeness.

Run:
    conda activate Spectral
    python -m pytest tests/test_V.py -v
  or
    python -m tests.test_V

Tests
-----
V1   TIR energy conservation     -- furnace test: closed cavity including TIR interface
V2   lambda_ex invariance        -- flat-illuminant fluorescent scene, two lambda_ex, same seed
V3   Correct vs wrong adjoint    -- single-bounce scene, FD vs both adjoint forms, all 5 params
V4   TIR + moving boundary full  -- two-bounce scene combining TIR-adjacent and lambda* crossing
V5   Substrate confound, grad.   -- gradient-descent recovery variance matches SVD prediction
V6   Inverse-crime check         -- ground truth from deliberately mismatched forward model
V7   Three-estimator bias test   -- Zeltner taxonomy, scene with K_x and TIR-adjacent sampling
V8   Near-rational FSR aliasing  -- d swept so FSR/delta_lambda crosses near-integer ratios
V9   C10 at full scene           -- two-bounce fluor-behind-glass with lambda* in emission band

--- Phase 2 (require C++ path tracer) ---
V10  C++ vs Python oracle        -- identical scene: C++ must match oracle to FD precision
V11  MIS balance correctness     -- multi-strategy scene: weights sum correctly, estimator unbiased
V12  Suite sensitivity check     -- deliberately revert to wrong form, suite must flag it
"""
from __future__ import annotations

import sys
import torch
torch.set_default_dtype(torch.float64)

from tests.harness import Reporter, StructResult, GradResult


# ---------------------------------------------------------------------------
# V1: TIR energy conservation (furnace test)
# Setup: closed uniform-temperature cavity with a TIR-adjacent interface.
# Claim: L_out = L_in everywhere at equilibrium.
#        Leak/pooling *near* the critical angle isolates a J_TIR-specific bug.
#
# No path tracer needed -- "closed cavity" reduces cleanly to a per-channel
# Kirchhoff argument on the deterministic Fredholm solver already in this
# repo: an idealized wall segment at each incidence angle cos_i reflects a
# fraction R(cos_i) of incoming radiance and re-emits the rest at the SAME
# wall temperature (emissivity = 1-R). At equilibrium every channel must read
# L_wall = R*L_wall + (1-R)*L_wall = L_wall identically -- true for ANY R in
# [0,1], including R=1 (TIR) in the idealized continuum equation.
#
# The truncated Neumann series (what neumann_forward actually computes) sums
# a finite geometric series in R: L = L_wall*(1 - R^(max_depth+1)), an exact
# closed form (verified against the real code below). For R<1 this converges
# to L_wall but SLOWS as R->1 (near-critical channels look like they're
# "leaking" energy at any finite bounce depth -- real, not a bug). At R=1
# EXACTLY, L=L_wall*(1-1)=0 identically, for ANY max_depth: the fixed-point
# equation (I-T)L=L_e degenerates to 0=0 there (any L solves it), and the
# iteration -- starting from L_e=(1-R)*L_wall=0 -- has no way to find the
# correct root L_wall. This is the concrete "J_TIR-specific bug" the table
# entry points at: it's the reason Theorem 3's v-substituted, EXPLICITLY
# defined-at-the-limit J_TIR machinery (A7/T5/G1) is load-bearing rather than
# a convenience -- a naive per-channel transport model has no other way to
# inject the correct equilibrium value exactly at the degenerate point.
# ---------------------------------------------------------------------------

def test_V1() -> list[StructResult]:
    from src.fresnel import fresnel_R
    from src.cauchy_ior import is_tir
    from src.forward import neumann_forward, fredholm_solve_exact

    n_i, n_t = 1.6, 1.0
    crit_cos = (1.0 - (n_t / n_i) ** 2) ** 0.5
    L_wall = 3.7
    max_depth = 32

    cos_i = torch.linspace(1.0, 0.02, 80, dtype=torch.float64)
    R = fresnel_R(n_i, n_t, cos_i, "unpolarized")
    tir = is_tir(cos_i, n_i, n_t)
    T = torch.diag(R)
    L_e = (1.0 - R) * L_wall

    L_neu = neumann_forward(T, L_e, max_depth)
    L_pred = L_wall * (1.0 - R ** (max_depth + 1))

    # Check 1: closed form matches the actual code exactly.
    closed_form_err = (L_neu - L_pred).abs().max().item()

    # Check 2: propagating subset converges to L_wall (equilibrium reached).
    prop_mask = ~tir
    prop_rel_err = ((L_neu[prop_mask] - L_wall).abs() / L_wall).max().item()

    # Check 3: TIR subset gives EXACTLY 0, not L_wall -- the hard "leak".
    tir_leak = L_neu[tir].abs().max().item()

    # Check 4: near-critical (propagating side), truncation error grows
    # monotonically toward the critical angle, tracking R^(max_depth+1)
    # exactly -- ties the "leak" directly to rho(T)->1 slow convergence,
    # not an unrelated numerical artifact.
    eps = torch.tensor([0.2, 0.05, 0.01, 0.002, 0.0005], dtype=torch.float64)
    cos_i_near = crit_cos + eps
    R_near = fresnel_R(n_i, n_t, cos_i_near, "unpolarized")
    L_near = neumann_forward(torch.diag(R_near), (1.0 - R_near) * L_wall, max_depth)
    rel_err_near = (L_wall - L_near).abs() / L_wall
    monotone_near = bool((rel_err_near[1:] >= rel_err_near[:-1] - 1e-14).all())
    pred_err_near = R_near ** (max_depth + 1)
    max_pred_err_diff = (rel_err_near - pred_err_near).abs().max().item()

    # Check 5: exact solve (infinite-bounce, no truncation) on the
    # propagating-only subset matches L_wall regardless of how close R gets
    # to 1 -- confirms Check 4's slowdown is a TRUNCATION artifact of finite
    # max_depth, not a problem with the underlying equation.
    T_prop = torch.diag(R[prop_mask])
    L_e_prop = (1.0 - R[prop_mask]) * L_wall
    L_exact_prop = fredholm_solve_exact(T_prop, L_e_prop)
    exact_prop_rel_err = ((L_exact_prop - L_wall).abs() / L_wall).max().item()

    # Check 6: exact solve on the FULL mixed set (including TIR channels)
    # is genuinely singular -- (I-T) has an exact zero row/col wherever R=1
    # -- confirming the degeneracy is structural, not a slow-convergence
    # illusion that infinite bounces would fix.
    exact_full_raised = False
    try:
        fredholm_solve_exact(T, L_e)
    except torch._C._LinAlgError:
        exact_full_raised = True

    tol = 1e-10
    return [
        StructResult("V1", "closed form L_wall*(1-R^(depth+1)) matches actual neumann_forward exactly",
                     closed_form_err, 0.0, closed_form_err, tol, closed_form_err < tol,
                     f"max|L_neu - L_pred| over {len(cos_i)} channels ({int(tir.sum())} TIR, "
                     f"{int((~tir).sum())} propagating)"),
        StructResult("V1", "propagating subset reaches equilibrium L_wall",
                     prop_rel_err, 0.0, prop_rel_err, tol, prop_rel_err < tol,
                     f"max rel err vs L_wall={L_wall} over {int(prop_mask.sum())} propagating channels"),
        StructResult("V1", "TIR subset gives EXACTLY 0, not L_wall -- the hard energy leak",
                     tir_leak, 0.0, tir_leak, 0.0, tir_leak == 0.0,
                     f"max|L| over {int(tir.sum())} TIR channels (should be L_wall={L_wall} physically, "
                     "is exactly 0 -- degenerate fixed-point equation picks the wrong root)"),
        StructResult("V1", "near-critical truncation error grows monotonically, tracks R^(depth+1) exactly",
                     max_pred_err_diff, 0.0, max_pred_err_diff, tol,
                     monotone_near and max_pred_err_diff < tol,
                     f"rel err at eps={eps.tolist()} from critical: {rel_err_near.tolist()} "
                     "-- ties 'leak near critical angle' to rho(T)->1, not unrelated noise"),
        StructResult("V1", "exact (infinite-bounce) solve on propagating subset hits L_wall regardless of R",
                     exact_prop_rel_err, 0.0, exact_prop_rel_err, tol, exact_prop_rel_err < tol,
                     "confirms Check 4's slowdown is a max_depth truncation artifact, not a real problem "
                     "with the underlying equation"),
        StructResult("V1", "exact solve on the FULL mixed (incl. TIR) set is genuinely singular",
                     float(not exact_full_raised), 0.0, float(not exact_full_raised), 0.0, exact_full_raised,
                     "torch.linalg.solve raises _LinAlgError -- (I-T) has an exact zero row/col at R=1, "
                     "a structural degeneracy, not just slow convergence"),
    ]


# ---------------------------------------------------------------------------
# V2: lambda_ex exact invariance (cheapest, strongest falsifier -- run first)
# Setup: flat-illuminant fluorescent scene; render at two different lambda_ex
#        (or vectors), same RNG seed.
# Claim: two renders agree within pure MC noise, zero systematic residual.
#
# Fully deterministic here (no RNG/seed needed at all -- this repo has no
# stochastic estimator yet, same reason G12 was deferred) -- run through the
# ACTUAL kernel_fluorescence + neumann_forward + Sensor.measure() pipeline,
# not a hand-rolled closed form like A9/A11. A deterministic check is a
# STRICTLY STRONGER falsifier than "agrees within MC noise": it must show
# EXACTLY 0 systematic residual, not just something small relative to noise.
#
# Theorem 7's own scope is explicitly "flat-illuminant + SINGLE-BOUNCE"
# (CLAUDE.md's identifiability table, row 4) -- verified here by contrast,
# same discipline as Sec9's own "scope condition shown load-bearing":
# max_depth=1 (single bounce) gives exact invariance; max_depth>=2 lets the
# fluorescent emission re-absorb itself, and the a-e overlap integral
# GENUINELY depends on lambda_ex -- the single-bounce qualifier is real,
# not decorative.
# ---------------------------------------------------------------------------

def test_V2() -> list[StructResult]:
    from src.kernels import kernel_fluorescence
    from src.forward import neumann_forward
    from src.sensor import hyperspectral_fx10

    lam = torch.linspace(200.0, 900.0, 8001, dtype=torch.float64)
    dlam = (lam[1] - lam[0]).item()
    weights = torch.full_like(lam, dlam)
    sensor = hyperspectral_fx10(lam, M=20)
    L_e = torch.full_like(lam, 4.2)   # flat illuminant, matches Theorem 7's hypothesis exactly

    def render_single(lam_ex: float, depth: int) -> torch.Tensor:
        K = kernel_fluorescence(lam, lam_ex, 620.0, 15.0, weights, quantum_yield=0.8)
        L = neumann_forward(K, L_e, depth)
        return sensor.measure(L)

    # Check 1: single species, single bounce -- EXACTLY invariant, no tolerance needed.
    img1 = render_single(500.0, depth=1)
    img2 = render_single(600.0, depth=1)
    single_species_diff = (img1 - img2).abs().max().item()

    torch.manual_seed(2)
    lam_ex_base = [480.0, 500.0, 520.0]
    lam_ex_pert = [440.0, 560.0, 505.0]
    lam_em_list, sf_list, qy_list = [560.0, 600.0, 650.0], [12.0, 18.0, 10.0], [0.5, 0.3, 0.2]

    def render_multi(lam_ex_list, depth: int) -> torch.Tensor:
        K = torch.zeros(len(lam), len(lam), dtype=torch.float64)
        for lex, lem, sf, qy in zip(lam_ex_list, lam_em_list, sf_list, qy_list):
            K = K + kernel_fluorescence(lam, lex, lem, sf, weights, quantum_yield=qy)
        L = neumann_forward(K, L_e, depth)
        return sensor.measure(L)

    # Check 2: k=3 species, single bounce -- simultaneous lambda_ex
    # perturbation, extends A11's formula-level proof through the full
    # kernel+solve+sensor pipeline.
    img_a = render_multi(lam_ex_base, depth=1)
    img_b = render_multi(lam_ex_pert, depth=1)
    multi_species_diff = (img_a - img_b).abs().max().item()
    scale = img_a.abs().max().item()

    # Check 3: NOT vacuous -- multi-bounce genuinely breaks the invariance,
    # confirming "single-bounce" is a real, load-bearing scope condition
    # rather than an unnecessary hedge.
    depths = [1, 2, 3, 8]
    diffs_by_depth = []
    for d in depths:
        i1 = render_single(500.0, depth=d)
        i2 = render_single(600.0, depth=d)
        diffs_by_depth.append((i1 - i2).abs().max().item())
    multi_bounce_grows = all(diffs_by_depth[i] <= diffs_by_depth[i + 1] + 1e-9
                              for i in range(len(diffs_by_depth) - 1))
    multi_bounce_breaks = diffs_by_depth[-1] > 1.0   # depth=8 genuinely nonzero, not noise

    tol = 1e-9
    return [
        StructResult("V2", "single species, single-bounce: image EXACTLY invariant to lambda_ex (deterministic, not just within noise)",
                     single_species_diff, 0.0, single_species_diff, 0.0, single_species_diff == 0.0,
                     f"max|image(lam_ex=500)-image(lam_ex=600)| over M={sensor.M} channels, depth=1"),
        StructResult("V2", "k=3 species, single-bounce: image invariant to simultaneous lambda_ex perturbation",
                     multi_species_diff / scale, 0.0, multi_species_diff / scale, tol,
                     multi_species_diff / scale < tol,
                     f"max|image_a-image_b|/scale = {multi_species_diff:.2e}/{scale:.2e} over 3 species, depth=1"),
        StructResult("V2", "NOT vacuous: multi-bounce genuinely breaks the invariance (single-bounce scope is load-bearing)",
                     diffs_by_depth[-1], None, None, 1.0, multi_bounce_grows and multi_bounce_breaks,
                     f"max|diff| by depth {dict(zip(depths, diffs_by_depth))} -- grows with depth, "
                     "confirming Theorem 7's single-bounce qualifier is real, not decorative"),
    ]


# ---------------------------------------------------------------------------
# V3: Correct vs wrong adjoint gradient (single-bounce)
# Setup: single-bounce fluorescent scene; gradient w.r.t. each of the 5
#        non-degenerate parameters.
# Claim: FD on re-rendered image agrees with correct adjoint; wrong adjoint
#        must fail on *every* parameter, including elastic ones.
#
# Uses the ACTUAL two_bounce() scene (thin film + fluorophore combined,
# K=K_tf+K_fl) -- G11's toy single-formula wrong-vs-correct finding, now
# validated at full-scene scale: does it survive real scene-building,
# H_wp/rho(T) checks, and an actual Sensor, not just a bare rank-1 matrix?
# Deterministic multi-bounce (max_depth=32, matching the rest of the repo)
# is the right regime here -- V3 is explicitly testing the ADJOINT machinery
# itself, unlike V2 which specifically needed single-bounce to match
# Theorem 7's exact scope.
# ---------------------------------------------------------------------------

def test_V3() -> list[StructResult]:
    from src.spectral_grid import make_grid
    from src.scenes import two_bounce, d65_on_grid
    from src.sensor import hyperspectral_fx10
    from src.gradient import (
        fd_gradient, kernel_gradient, kernel_gradient_wrong_adjoint,
        fluorescence_dK_dlam_em, fluorescence_dK_dsigma_f,
    )
    from src.kernels import kernel_thinfilm, kernel_fluorescence, fabry_airy_dR_dd, fabry_airy_dR_dA, fabry_airy_dR_dB

    grid = make_grid()
    sensor = hyperspectral_fx10(grid.lam, M=20)
    max_depth = 32

    base = dict(d=5.0, A=1.5, B=5000.0, cos_i=0.9, lam_ex=450.0, lam_em=550.0, sigma_f=20.0, quantum_yield=0.99)
    L_e = d65_on_grid(grid)
    g = sensor.S.T @ torch.ones(sensor.M, dtype=torch.float64)

    K_tf = kernel_thinfilm(grid.lam, base["cos_i"], base["d"], base["A"], base["B"])
    K_fl = kernel_fluorescence(grid.lam, base["lam_ex"], base["lam_em"], base["sigma_f"], grid.weights, base["quantum_yield"])
    K = K_tf + K_fl

    def loss_fn(**override) -> torch.Tensor:
        params = dict(base)
        params.update(override)
        res = two_bounce(grid, sensor, **params)
        return res.image.sum()

    dT_dtheta = {
        "d":       fabry_airy_dR_dd(grid.lam, base["cos_i"], base["d"], base["A"], base["B"]),
        "A":       fabry_airy_dR_dA(grid.lam, base["cos_i"], base["d"], base["A"], base["B"]),
        "B":       fabry_airy_dR_dB(grid.lam, base["cos_i"], base["d"], base["A"], base["B"]),
        "sigma_f": fluorescence_dK_dsigma_f(grid.lam, base["lam_ex"], base["lam_em"], base["sigma_f"], grid.weights, base["quantum_yield"]),
        "lam_em":  fluorescence_dK_dlam_em(grid.lam, base["lam_ex"], base["lam_em"], base["sigma_f"], grid.weights, base["quantum_yield"]),
    }
    elastic_params = {"d", "A", "B"}   # live entirely in the diagonal, self-adjoint K_tf channel

    tol = 1e-6
    results: list[StructResult] = []
    for name, dTdtheta in dT_dtheta.items():
        t = torch.tensor(float(base[name]), requires_grad=True)
        loss = loss_fn(**{name: t})
        loss.backward()
        ag = t.grad.item()

        fd = fd_gradient(lambda: loss_fn(**{name: t}), t).item()
        correct = kernel_gradient(K, dTdtheta, L_e, g, max_depth).item()
        wrong = kernel_gradient_wrong_adjoint(K, dTdtheta, L_e, g, max_depth).item()

        rel_ag_fd = abs(ag - fd) / max(abs(fd), 1e-30)
        rel_correct_fd = abs(correct - fd) / max(abs(fd), 1e-30)
        rel_wrong_correct = abs(wrong - correct) / max(abs(correct), 1e-30)

        tag = "elastic" if name in elastic_params else "fluorescence"
        results.append(StructResult(
            "V3", f"{name} ({tag}): autograd, correct adjoint, and FD agree on the full scene",
            max(rel_ag_fd, rel_correct_fd), 0.0, max(rel_ag_fd, rel_correct_fd), tol,
            max(rel_ag_fd, rel_correct_fd) < tol,
            f"ag={ag:.6f} correct={correct:.6f} fd={fd:.6f}"))
        results.append(StructResult(
            "V3", f"{name} ({tag}): wrong adjoint measurably fails",
            rel_wrong_correct, None, None, 5e-3, rel_wrong_correct > 5e-3,
            f"wrong={wrong:.6f} vs correct={correct:.6f}, rel err {rel_wrong_correct:.4f} -- "
            + ("wrong adjoint contaminates even this elastic param via the full non-symmetric T, "
               "not just the fluorescence channel where the Stokes shift lives" if name in elastic_params
               else "expected: this param lives directly in the asymmetric fluorescence channel")))

    return results


# ---------------------------------------------------------------------------
# V4: Naive vs corrected gradient near criticality + moving lambda* combined
# Setup: two-bounce scene combining TIR-adjacent sampling with emission band
#        straddling lambda*.
# Claim: FD oracle on full re-render -- tests variance behavior under MC
#        sampling, which T12 cannot test without a real estimator.
# ---------------------------------------------------------------------------

def test_V4() -> GradResult:
    raise NotImplementedError("V4")


# ---------------------------------------------------------------------------
# V5: Substrate confound and B-bottleneck in real gradient-based recovery
# Setup: render ground truth, recover via gradient descent, multiple random inits.
# Claim: recovery variance across seeds rank-orders parameters the same way
#        the Jacobian SVD predicts.
# ---------------------------------------------------------------------------

def test_V5() -> StructResult:
    raise NotImplementedError("V5")


# ---------------------------------------------------------------------------
# V6: Inverse-crime check (cross-cutting)
# Setup: any recovery test (V5), but generate ground truth with a deliberately
#        mismatched forward model.
# Claim: recovery quality must degrade under mismatch -- else the "successful"
#        recovery was partly circular.
# ---------------------------------------------------------------------------

def test_V6() -> StructResult:
    raise NotImplementedError("V6")


# ---------------------------------------------------------------------------
# V7: Three-estimator bias test (Zeltner taxonomy)
# Setup: scene exercising both K_x and TIR-adjacent sampling.
# Claim: all three estimators vs V3's FD oracle -- catch an estimator that is
#        unbiased in expectation but with variance blowing up near the singular
#        manifold.
# ---------------------------------------------------------------------------

def test_V7() -> list[StructResult]:
    raise NotImplementedError("V7")


# ---------------------------------------------------------------------------
# V8: Near-rational FSR aliasing (rendered)
# Setup: thin-film scene, d swept so FSR/delta_lambda_sample crosses near-integer
#        ratios, actually rendered.
# Claim: oversampled-lambda reference render vs production sampling rate;
#        any spurious beat pattern in the rendered image is a failure.
# ---------------------------------------------------------------------------

def test_V8() -> StructResult:
    raise NotImplementedError("V8")


# ---------------------------------------------------------------------------
# V9: C10 at full scene complexity
# Setup: full two-bounce scene: fluorescent emission inside dispersive medium,
#        escaping past TIR-bounded interface, lambda* swept through emission band.
# Claim: FD on full re-render at A +/- eps -- the explicit named falsifier
#        from the project's own list.
# ---------------------------------------------------------------------------

def test_V9() -> GradResult:
    raise NotImplementedError("V9")


# ---------------------------------------------------------------------------
# V10-V12: Phase 2 only (require C++ path tracer)
# ---------------------------------------------------------------------------

def test_V10() -> StructResult:
    raise NotImplementedError("V10 -- Phase 2: requires C++ path tracer")


def test_V11() -> StructResult:
    raise NotImplementedError("V11 -- Phase 2: requires C++ path tracer")


def test_V12() -> StructResult:
    raise NotImplementedError("V12 -- Phase 2: requires C++ path tracer")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

PHASE1 = [test_V1, test_V2, test_V3, test_V4, test_V5, test_V6, test_V7, test_V8, test_V9]
PHASE2 = [test_V10, test_V11, test_V12]
ALL    = PHASE1 + PHASE2


def main(phase2: bool = False) -> None:
    tests = ALL if phase2 else PHASE1
    rep = Reporter()
    for fn in tests:
        try:
            result = fn()
            if isinstance(result, list):
                for r in result:
                    rep.add(r)
            else:
                rep.add(result)
        except NotImplementedError as e:
            print(f"SKIP  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e}", file=sys.stderr)
    rep.print_all()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", action="store_true",
                        help="Also run Phase 2 tests (require C++ tracer)")
    args = parser.parse_args()
    main(phase2=args.phase2)
