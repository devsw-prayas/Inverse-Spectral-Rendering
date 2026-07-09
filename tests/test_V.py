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
#
# Scope note: the docstring names two distinct Table-1 findings. T6's
# substrate-confound result used a one-off inline 3-layer stack with no
# reusable scene machinery; kernel_thinfilm/fabry_airy_R (used by both
# existing scenes.py archetypes) is free-standing only. Built a real
# fabry_airy_R_substrate + film_on_substrate scene (kernels.py/scenes.py) so
# this can be tested as an actual recovery, not just a Jacobian-norm check
# (verified fabry_airy_R_substrate matches T6's inline R3_s exactly, 0.0
# max diff, before using it here).
#
# "Recovery via gradient descent, multiple random inits" turns out to need
# care: with a NOISELESS residual, a real Newton-type optimizer (LM here,
# since this genuinely is nonlinear least squares -- Adam/plain-GD would
# either mask or fake the conditioning effect, see below) converges to the
# exact global optimum from every random init regardless of conditioning
# (verified: std=0 in both high- and low-contrast regimes after LM, ~4
# iterations either way) -- there is no "recovery variance" to speak of
# without noise. The actual falsifiable claim needs measurement noise: add
# small IID noise to the rendered image per seed, recover via LM, and check
# the resulting empirical parameter scatter against the classical
# Gauss-Newton covariance formula Cov(theta_hat) ~ sigma^2 (J^T J)^-1 at the
# true point -- the same Jacobian object already used for SVD-based ranking
# in T9/G10, so "recovery variance matches the Jacobian SVD prediction" is
# tested directly rather than by analogy. (Plain gradient descent with a
# fixed step budget was tried first and gives a similar-looking result from
# *different random inits* with no noise at all -- but that is an artifact
# of an under-powered optimizer running out of budget, not a real
# identifiability effect; LM with noise is the honest version.)
# ---------------------------------------------------------------------------

def test_V5() -> list[StructResult]:
    from src.spectral_grid import SpectralGrid
    from src.scenes import film_on_substrate, two_bounce
    from src.sensor import hyperspectral_fx10
    from src.gradient import levenberg_marquardt

    results: list[StructResult] = []

    # --- Panel A: substrate confound (T6's own contrast endpoints) ---
    N = 200
    lam = torch.linspace(400.0, 700.0, N, dtype=torch.float64)
    dlam = (lam[1] - lam[0]).item()
    weights = torch.full_like(lam, dlam)
    grid = SpectralGrid(nu_tilde=torch.zeros(N, dtype=torch.float64), lam=lam,
                         weights=weights, N=N, d_nu=dlam)
    sensor = hyperspectral_fx10(lam, M=20)
    L_e = torch.full_like(lam, 4.2)

    d_true, A_film, B_film, D_sub = 120.0, 1.5, 5000.0, 5000.0
    noise_sigma_a = 1e-4
    n_seeds_a = 40

    def render_a(d, C_sub):
        return film_on_substrate(grid, sensor, d=d, A=A_film, B=B_film,
                                  C_sub=C_sub, D_sub=D_sub, cos_i=1.0, L_e=L_e).image

    stds_a: dict[float, float] = {}
    preds_a: dict[float, float] = {}
    for C_sub in (1.80, 1.505):
        img_true = render_a(torch.tensor(d_true), C_sub).detach()

        d_req = torch.tensor(d_true, requires_grad=True)
        J = torch.autograd.functional.jacobian(lambda d: render_a(d, C_sub), d_req).detach()
        preds_a[C_sub] = noise_sigma_a / (J @ J).sqrt().item()

        torch.manual_seed(42)
        dhats = []
        for _ in range(n_seeds_a):
            img_noisy = img_true + torch.randn(img_true.shape) * noise_sigma_a
            def residual_fn(theta, img_noisy=img_noisy, C_sub=C_sub):
                return render_a(theta[0], C_sub) - img_noisy
            theta_hat, _, _ = levenberg_marquardt(residual_fn, torch.tensor([d_true]))
            dhats.append(theta_hat[0].item())
        stds_a[C_sub] = torch.tensor(dhats).std().item()

    for C_sub in (1.80, 1.505):
        ratio = stds_a[C_sub] / preds_a[C_sub]
        results.append(StructResult(
            "V5", f"Panel A (substrate confound): empirical/predicted recovery std at C_sub={C_sub}",
            ratio, 1.0, abs(ratio - 1.0), 0.5, abs(ratio - 1.0) < 0.5,
            f"empirical_std={stds_a[C_sub]:.3e}  predicted_std={preds_a[C_sub]:.3e} (Gauss-Newton, sigma={noise_sigma_a:.0e})"))

    regime_ratio_emp = stds_a[1.505] / stds_a[1.80]
    regime_ratio_pred = preds_a[1.505] / preds_a[1.80]
    rel = abs(regime_ratio_emp - regime_ratio_pred) / regime_ratio_pred
    results.append(StructResult(
        "V5", "Panel A: low/high-contrast recovery-std ratio matches Jacobian-predicted ratio",
        regime_ratio_emp, regime_ratio_pred, rel, 0.3, rel < 0.3,
        f"empirical ratio={regime_ratio_emp:.1f}x  predicted ratio={regime_ratio_pred:.1f}x -- "
        "same confound T6 found via a bare Jacobian norm, now via real noisy recovery"))
    results.append(StructResult(
        "V5", "Panel A NOT vacuous: low-contrast recovery is genuinely much noisier",
        regime_ratio_emp, None, None, 10.0, regime_ratio_emp > 10.0,
        f"{regime_ratio_emp:.1f}x noisier recovery near C_sub=A_film=1.5 than at C_sub=1.80"))

    # --- Panel B: B-bottleneck (T9/G10's exact 5-param operating point) ---
    Nb = 300
    lam_b = torch.linspace(400.0, 700.0, Nb, dtype=torch.float64)
    dlam_b = (lam_b[1] - lam_b[0]).item()
    grid_b = SpectralGrid(nu_tilde=torch.zeros(Nb, dtype=torch.float64), lam=lam_b,
                           weights=torch.full_like(lam_b, dlam_b), N=Nb, d_nu=dlam_b)
    sensor_b = hyperspectral_fx10(lam_b, M=20)
    L_e_b = torch.full_like(lam_b, 4.2)

    cos_i_b, lam_ex_b, qy_b = 1.0, 450.0, 0.8
    theta_true = torch.tensor([120.0, 1.5, 5000.0, 30.0, 550.0])   # d, A, B, sigma_f, lam_em
    names = ["d", "A", "B", "sigma_f", "lam_em"]
    noise_sigma_b = 3e-4
    n_seeds_b = 40

    def render_b(theta):
        d, A, B, sf, lem = theta[0], theta[1], theta[2], theta[3], theta[4]
        return two_bounce(grid_b, sensor_b, d=d, A=A, B=B, cos_i=cos_i_b, lam_ex=lam_ex_b,
                           lam_em=lem, sigma_f=sf, quantum_yield=qy_b, L_e=L_e_b).image

    img_true_b = render_b(theta_true).detach()
    theta_req = theta_true.clone().requires_grad_(True)
    J_b = torch.autograd.functional.jacobian(render_b, theta_req).detach()
    predicted_std_b = noise_sigma_b * torch.diag(torch.linalg.pinv(J_b.T @ J_b)).clamp(min=0).sqrt()

    torch.manual_seed(7)
    thetas_hat = []
    for _ in range(n_seeds_b):
        img_noisy = img_true_b + torch.randn(img_true_b.shape) * noise_sigma_b
        def residual_fn(theta, img_noisy=img_noisy):
            return render_b(theta) - img_noisy
        theta_hat, _, _ = levenberg_marquardt(residual_fn, theta_true.clone())
        thetas_hat.append(theta_hat)
    empirical_std_b = torch.stack(thetas_hat).std(dim=0)

    ratios_b = (empirical_std_b / predicted_std_b)
    max_dev = (ratios_b - 1.0).abs().max().item()
    results.append(StructResult(
        "V5", "Panel B: empirical/predicted recovery std per-parameter (all 5, worst-case shown)",
        max_dev, 0.0, max_dev, 0.5, max_dev < 0.5,
        "ratios: " + " ".join(f"{n}={r:.3f}" for n, r in zip(names, ratios_b.tolist()))))

    pred_rank = torch.argsort(predicted_std_b, descending=True).tolist()
    emp_rank = torch.argsort(empirical_std_b, descending=True).tolist()
    rank_match = pred_rank == emp_rank
    results.append(StructResult(
        "V5", "Panel B: empirical recovery-std ranking across params exactly matches Jacobian-predicted ranking",
        float(rank_match), 1.0, float(not rank_match), 0.0, rank_match,
        f"predicted (worst->best): {[names[i] for i in pred_rank]}  "
        f"empirical: {[names[i] for i in emp_rank]}"))
    results.append(StructResult(
        "V5", "Panel B NOT vacuous: B is confirmed the worst-recovered parameter (bottleneck)",
        float(names[emp_rank[0]] == "B"), 1.0, float(names[emp_rank[0]] != "B"), 0.0,
        names[emp_rank[0]] == "B",
        f"largest empirical recovery std belongs to '{names[emp_rank[0]]}' -- matches T9/G10's finding"))

    return results


# ---------------------------------------------------------------------------
# V6: Inverse-crime check (cross-cutting)
# Setup: any recovery test (V5), but generate ground truth with a deliberately
#        mismatched forward model.
# Claim: recovery quality must degrade under mismatch -- else the "successful"
#        recovery was partly circular.
#
# Mismatch axis: bounce depth D. scenes.py's docstrings already treat
# max_depth as a first-class, deliberate model axis ("bounce depth D --
# configurable axis per spec Sec6.1"), and V2 already showed depth genuinely
# changes the physics (multi-bounce fluorescent re-absorption). So: generate
# "true" data with the correct, converged multi-bounce two_bounce() (depth
# high enough to be at the fixed point), then recover with an assumed model
# that WRONGLY truncates bounce depth -- a real, physically motivated
# inverse crime (fitting a single/few-scatter model to multi-scatter data),
# not an arbitrary synthetic perturbation.
#
# Falsifier: reduced chi-square (post-fit loss / ((M-P) sigma^2)) is the
# standard diagnostic -- ~1 under a correctly matched model plus noise (this
# is V5 Panel B's own regime), and >>1 under structural misspecification
# that noise-rescaling cannot explain. Checked to be monotonic in mismatch
# severity (assumed depth closer to the true depth -> better fit), not just
# a single before/after number.
# ---------------------------------------------------------------------------

def test_V6() -> list[StructResult]:
    from src.spectral_grid import SpectralGrid
    from src.scenes import two_bounce
    from src.sensor import hyperspectral_fx10
    from src.gradient import levenberg_marquardt

    N = 300
    lam = torch.linspace(400.0, 700.0, N, dtype=torch.float64)
    dlam = (lam[1] - lam[0]).item()
    grid = SpectralGrid(nu_tilde=torch.zeros(N, dtype=torch.float64), lam=lam,
                         weights=torch.full_like(lam, dlam), N=N, d_nu=dlam)
    sensor = hyperspectral_fx10(lam, M=20)
    L_e = torch.full_like(lam, 4.2)

    cos_i, lam_ex, qy = 1.0, 450.0, 0.8
    theta_true = torch.tensor([120.0, 1.5, 5000.0, 30.0, 550.0])   # d, A, B, sigma_f, lam_em
    P = theta_true.numel()
    M = sensor.M
    dof = M - P
    noise_sigma = 3e-4
    n_seeds = 15
    true_depth = 32   # converged multi-bounce -- the actual physical process

    def render(theta, depth):
        d, A, B, sf, lem = theta[0], theta[1], theta[2], theta[3], theta[4]
        return two_bounce(grid, sensor, d=d, A=A, B=B, cos_i=cos_i, lam_ex=lam_ex,
                           lam_em=lem, sigma_f=sf, quantum_yield=qy, L_e=L_e,
                           max_depth=depth).image

    img_true = render(theta_true, true_depth).detach()

    assumed_depths = [1, 4, 8, true_depth]
    reduced_chi2: dict[int, float] = {}
    mean_bias: dict[int, torch.Tensor] = {}
    for assumed_depth in assumed_depths:
        torch.manual_seed(99)
        chi2s, biases = [], []
        for _ in range(n_seeds):
            img_noisy = img_true + torch.randn(M) * noise_sigma
            def residual_fn(theta, img_noisy=img_noisy, depth=assumed_depth):
                return render(theta, depth) - img_noisy
            theta_hat, loss, _ = levenberg_marquardt(residual_fn, theta_true.clone())
            chi2s.append(loss / (dof * noise_sigma ** 2))
            biases.append((theta_hat - theta_true).abs())
        reduced_chi2[assumed_depth] = torch.tensor(chi2s).mean().item()
        mean_bias[assumed_depth] = torch.stack(biases).mean(dim=0)

    tol = 0.5
    results: list[StructResult] = []
    results.append(StructResult(
        "V6", f"matched model (depth={true_depth}, the true generating depth): reduced chi^2 ~ 1",
        reduced_chi2[true_depth], 1.0, abs(reduced_chi2[true_depth] - 1.0), 3.0,
        abs(reduced_chi2[true_depth] - 1.0) < 3.0,
        "consistent with pure measurement noise, no structural misfit -- this is V5 Panel B's own regime"))
    results.append(StructResult(
        "V6", "maximally mismatched model (depth=1 vs true depth=32): reduced chi^2 >> 1",
        reduced_chi2[1], None, None, 100.0, reduced_chi2[1] > 100.0,
        f"reduced chi^2 = {reduced_chi2[1]:.3e} -- structural misfit that noise rescaling cannot explain, "
        "confirms V5's recovery success was not an inverse crime"))
    chi2_seq = [reduced_chi2[d] for d in assumed_depths]
    # 20% relative slack: the true model converges by depth~8 (see V2), so the
    # depth=8 vs depth=32 pair is a physically-identical plateau and only
    # differs by LM/seed noise -- a strict decrease there would be a false
    # positive, not a real monotonicity violation.
    monotone = all(chi2_seq[i] > chi2_seq[i + 1] * 0.8 for i in range(len(chi2_seq) - 1))
    results.append(StructResult(
        "V6", "NOT vacuous: reduced chi^2 decreases monotonically as assumed depth -> true depth",
        float(monotone), 1.0, float(not monotone), 0.0, monotone,
        "reduced chi^2 by assumed depth: " + " ".join(f"{d}:{reduced_chi2[d]:.3e}" for d in assumed_depths)))

    worst_param = int(mean_bias[1].argmax().item())
    names = ["d", "A", "B", "sigma_f", "lam_em"]
    bias_ratio = (mean_bias[1][worst_param] / mean_bias[true_depth][worst_param].clamp(min=1e-30)).item()
    results.append(StructResult(
        "V6", f"mismatched-model parameter bias ({names[worst_param]}) is much larger than matched-model bias",
        bias_ratio, None, None, 10.0, bias_ratio > 10.0,
        f"mean|bias| at depth=1: {mean_bias[1][worst_param]:.4g}  at depth={true_depth}: "
        f"{mean_bias[true_depth][worst_param]:.4g} -- {bias_ratio:.1f}x worse under mismatch"))

    return results


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
#
# G8 already checked this at the Jacobian-conditioning level (negative
# result: no dominant FSR-locked periodicity). This promotes it to the
# render level per the table's own distinction: does an actual rendered
# sensor image at make_grid()'s production oversampling diverge from a much
# finer reference, in a way that beats against the fringe period as d sweeps?
#
# Landmine hit and fixed: Sensor.measure() (raw S@L, no dlam factor) is NOT
# resolution-invariant -- comparing images built on two different-density
# grids directly is comparing unlike quantities (the raw sum scales with
# how many grid points fall in a sensor band). Same footgun as the V9
# exploration. Fixed by weighting with the grid's own quadrature weights
# before measuring (S @ (R * grid.weights)), approximating the same
# continuum integral regardless of grid density.
# ---------------------------------------------------------------------------

def test_V8() -> list[StructResult]:
    from src.spectral_grid import make_grid
    from src.kernels import fabry_airy_R
    from src.sensor import hyperspectral_fx10

    cos_i = torch.tensor(1.0, dtype=torch.float64)
    A0, B0 = 1.5, 5000.0
    lam_c = 550.0
    n_ref = A0 + B0 / lam_c ** 2
    d_fringe = lam_c / (2.0 * n_ref)   # G8's own d-domain fringe period at lam_c

    n_periods = 16
    d0 = 300.0
    span = n_periods * d_fringe
    d_max_sweep = d0 + span
    lam_min, lam_max = 450.0, 650.0

    def image(d, grid, sensor):
        R = fabry_airy_R(grid.lam, cos_i, d, A0, B0, "unpolarized")
        return sensor.measure(R * grid.weights)

    grid_fine = make_grid(lam_min, lam_max, A0, B0, d_max=d_max_sweep, oversampling=32.0)
    sensor_fine = hyperspectral_fx10(grid_fine.lam, M=6)

    N = 300
    d_sweep = torch.linspace(d0, d_max_sweep, N, dtype=torch.float64)
    img_fine = [image(d_sweep[i].item(), grid_fine, sensor_fine) for i in range(N)]

    def err_curve(oversampling: float) -> torch.Tensor:
        grid_p = make_grid(lam_min, lam_max, A0, B0, d_max=d_max_sweep, oversampling=oversampling)
        sensor_p = hyperspectral_fx10(grid_p.lam, M=6)
        errs = torch.zeros(N, dtype=torch.float64)
        for i in range(N):
            img_p = image(d_sweep[i].item(), grid_p, sensor_p)
            errs[i] = (img_fine[i] - img_p).norm() / img_fine[i].norm().clamp(min=1e-30)
        return errs

    err_prod = err_curve(4.0)    # make_grid's own default oversampling
    err_bad = err_curve(0.25)    # deliberately well below Nyquist

    results: list[StructResult] = []
    results.append(StructResult(
        "V8", "production oversampling (4.0, make_grid's own default): max relative image error across 16 FSR periods",
        err_prod.max().item(), 0.0, err_prod.max().item(), 0.01, err_prod.max().item() < 0.01,
        f"mean={err_prod.mean().item():.4f}  max={err_prod.max().item():.4f} over N={N} points, "
        "no aliasing hot-spot anywhere in the sweep"))
    results.append(StructResult(
        "V8", "NOT vacuous: deliberately undersampled grid (oversampling=0.25) shows a large, clearly detectable error",
        err_bad.mean().item(), None, None, 0.3, err_bad.mean().item() > 0.3,
        f"mean={err_bad.mean().item():.3f} -- confirms the comparison has teeth (violating the Nyquist "
        "safety margin is actually detected), not just insensitive to everything"))

    means = []
    for ov in (4.0, 2.0, 1.0, 0.5, 0.25):
        means.append(err_curve(ov).mean().item())
    monotone = all(means[i] < means[i + 1] for i in range(len(means) - 1))
    results.append(StructResult(
        "V8", "mean image error increases monotonically as oversampling decreases (genuine sampling-density effect, not noise)",
        float(monotone), 1.0, float(not monotone), 0.0, monotone,
        "mean err by oversampling [4,2,1,0.5,0.25]: " + " ".join(f"{m:.4f}" for m in means)))

    def detrend(y: torch.Tensor) -> torch.Tensor:
        x = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
        X = torch.stack([torch.ones(N, dtype=torch.float64), x], dim=1)
        beta = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze()
        return y - X @ beta

    def r2_at_period(detr: torch.Tensor, period: float) -> float:
        theta = 2.0 * torch.pi * d_sweep / period
        Xs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        b = torch.linalg.lstsq(Xs, detr.unsqueeze(1)).solution.squeeze()
        ss_res = ((detr - Xs @ b) ** 2).sum().item()
        total_var = (detr ** 2).sum().item()
        return 1.0 - ss_res / total_var

    r2_prod = r2_at_period(detrend(err_prod), d_fringe)
    r2_bad = r2_at_period(detrend(err_bad), d_fringe)
    tol_periodic = 0.3
    results.append(StructResult(
        "V8", "even at maximal mismatch, image error is not dominated by a single tone at the FSR fundamental "
        "(same negative result as G8, now at the render level)",
        max(r2_prod, r2_bad), 0.0, max(r2_prod, r2_bad), tol_periodic, max(r2_prod, r2_bad) < tol_periodic,
        f"R^2 at d_fringe={d_fringe:.2f}nm: production={r2_prod:.3f}, undersampled={r2_bad:.3f} -- "
        "the large undersampled error is a broad-spectrum discrepancy, not a clean single-frequency beat"))

    return results


# ---------------------------------------------------------------------------
# V9: moving-boundary Leibniz term at full multi-bounce scene complexity
# Setup: fluorescent emission inside a dispersive medium, escaping past a
#        TIR-bounded interface, lambda* swept through the emission band.
# Claim: FD on the (closed-form) full scene at A/B +/- eps -- the explicit
#        named falsifier from the project's own list.
#
# RESOLVED 2026-07-08 (theory) / formalized here. Sec13's locked
# moving-boundary lemma (-e(lambda*)*dlambda*/dtheta) is derived for a bare
# single integral with a BOUNDED integrand at the moving endpoint (T12/G3's
# own setup). Promoting it to full scene complexity hit three compounding
# issues:
#   1. TIR channels don't converge under truncated Neumann bounces -- fixed
#      by restricting to the propagating-only subset and solving exactly
#      (Sherman-Morrison, rank-1 K_fl), matching V1 Check 5/6's precedent.
#   2. A raw/unweighted sum over L is not grid-resolution-invariant -- fixed
#      by using a genuine continuum quadrature (Gauss-Legendre), not a
#      discrete SpectralGrid sum, throughout.
#   3. The genuinely hard part: near the moving critical wavelength, the
#      integrand feeding the Sherman-Morrison feedback amplitude c(theta)
#      diverges as 1/sqrt(lambda-lambda*) -- an INTEGRABLE singularity sitting
#      exactly at the boundary the Leibniz term describes. Fixed via a
#      Theorem-3-style substitution w=sqrt(lambda-lambda*(theta)), mapped to a
#      fixed t in [0,1] -- see src/gradient.py's v9_escaping_flux and the long
#      comment above it for the full derivation.
#
# Validation here deliberately does NOT use a raw discrete-grid FD (confirmed
# unreliable at any tested resolution during derivation -- it fights the
# lambda* escape crossing and the excitation-tail crossing at once). Instead:
#   - Primary: autograd vs FD, both against the SAME closed form
#     v9_escaping_flux (this is the "renderer" here; FD on it is a legitimate
#     three-way-oracle-style check, just not grid-based).
#   - Not vacuous: the classical Sec13 lemma applied naively to the whole
#     escaping-flux integral (ignoring that c(theta) has its own interior
#     theta-dependence) is shown to measurably disagree near the absorption
#     peak and to correctly recover the right answer once lambda* is several
#     sigma_f away from lam_ex -- same "edges of validity" pattern as G3/T14.
#   - Physical sanity: the closed form's VALUE is cross-checked against a
#     real discrete exact linear solve on the propagating subset, confirming
#     convergence (not just internal self-consistency) as the discrete grid
#     is refined -- the discrete solve converges slowly (the same 1/sqrt
#     singularity, unresolved on any finite grid), so this is a
#     convergence-trend check, not a tight bound (same discipline as A5/A6).
# ---------------------------------------------------------------------------

def _v9_discrete_reference(A, B, cos_i, lam_max, lam_ex, lam_em, sigma_f, quantum_yield, L0, N):
    from src.gradient import lambda_star

    lam_star = lambda_star(A, B, cos_i).item()
    lam = torch.linspace(lam_star + 1e-6, lam_max, N, dtype=torch.float64)
    dlam = (lam[1] - lam[0]).item()

    n = A + B / lam ** 2
    sin2_t = n ** 2 * (1.0 - cos_i ** 2)
    v = torch.sqrt((1.0 - sin2_t).clamp(min=0.0))
    rs = (n * cos_i - v) / (n * cos_i + v)
    rp = (cos_i - n * v) / (cos_i + n * v)
    R = 0.5 * (rs ** 2 + rp ** 2)

    a     = torch.exp(-0.5 * ((lam - lam_ex) / sigma_f) ** 2)
    e_raw = torch.exp(-0.5 * ((lam - lam_em) / sigma_f) ** 2)
    e     = e_raw / ((2.0 * torch.pi) ** 0.5 * sigma_f)
    L_e   = torch.full_like(lam, L0)

    G0 = 1.0 / (1.0 - R)
    v_vec = quantum_yield * a * dlam
    A_int = (v_vec * G0 * L_e).sum()
    B_int = (v_vec * G0 * e).sum()
    c_val = A_int / (1.0 - B_int)
    L = G0 * (L_e + c_val * e)
    phi = (1.0 - R) * L
    return (phi * dlam).sum().item()


def test_V9() -> list:
    from src.gradient import (
        gauss_legendre_01, v9_escaping_flux, lambda_star,
        dlambda_star_dA, moving_boundary_grad, fd_gradient,
    )

    cos_i = torch.tensor(0.5, dtype=torch.float64)
    lam_max = 700.0
    lam_ex, lam_em, sigma_f = 480.0, 550.0, 20.0
    quantum_yield, L0 = 0.8, 4.2
    B = torch.tensor(9000.0, dtype=torch.float64)
    kappa = 1.0 / (1.0 - cos_i ** 2) ** 0.5
    t_nodes, t_weights = gauss_legendre_01(64)

    lam_star_targets = [420.0, 480.0, 520.0, 550.0, 580.0, 620.0, 680.0]
    tol = 1e-5

    results: list = []
    naive_rel: dict[float, float] = {}

    for target in lam_star_targets:
        A0 = (kappa - B / target ** 2).item()

        for name, param0 in (("A", A0), ("B", B.item())):
            theta = torch.tensor(param0, dtype=torch.float64, requires_grad=True)
            A_arg = theta if name == "A" else torch.tensor(A0, dtype=torch.float64)
            B_arg = theta if name == "B" else B

            I_val, info = v9_escaping_flux(A_arg, B_arg, cos_i, lam_max, lam_ex, lam_em,
                                            sigma_f, quantum_yield, L0, t_nodes, t_weights)
            ag = torch.autograd.grad(I_val, theta)[0].item()

            def fn():
                A_a = theta if name == "A" else torch.tensor(A0, dtype=torch.float64)
                B_a = theta if name == "B" else B
                val, _ = v9_escaping_flux(A_a, B_a, cos_i, lam_max, lam_ex, lam_em,
                                           sigma_f, quantum_yield, L0, t_nodes, t_weights)
                return val

            fd = fd_gradient(fn, theta).item()
            rel = abs(ag - fd) / max(abs(fd), 1e-30)
            results.append(GradResult(
                test_id="V9", label=f"lambda*={target:.0f}nm: autograd vs FD on the closed form",
                param=name, autograd=ag, analytic=None, fd=fd,
                rel_ag_fd=rel, rel_an_fd=None, tol=tol))

        # Not-vacuous: naive Sec13-boundary-only term vs the correct dI/dA.
        A_t = torch.tensor(A0, dtype=torch.float64, requires_grad=True)
        I_val, info = v9_escaping_flux(A_t, B, cos_i, lam_max, lam_ex, lam_em,
                                        sigma_f, quantum_yield, L0, t_nodes, t_weights)
        correct = torch.autograd.grad(I_val, A_t)[0].item()

        lam_s = info["lam_star"].detach()
        c_val = info["c_val"].detach()
        e_star = torch.exp(-0.5 * ((lam_s - lam_em) / sigma_f) ** 2) / ((2.0 * torch.pi) ** 0.5 * sigma_f)
        phi_star = L0 + e_star * c_val
        dlam_star_dA = dlambda_star_dA(lam_s, B)
        naive = moving_boundary_grad(phi_star, dlam_star_dA).item()
        naive_rel[target] = abs(naive - correct) / max(abs(correct), 1e-30)

    close_to_peak = naive_rel[480.0]
    far_from_peak = naive_rel[680.0]
    results.append(StructResult(
        "V9", "naive Sec13-boundary-only term breaks down near the absorption peak "
        "(lambda* within a few sigma_f of lam_ex)",
        close_to_peak, None, None, 0.3, close_to_peak > 0.3,
        f"rel diff vs correct at lambda*=480 (=lam_ex): {close_to_peak:.3f} -- classical single-integral "
        "Leibniz term is blind to c(theta)'s own singular-integrand interior dependence"))
    results.append(StructResult(
        "V9", "NOT vacuous the other way: naive term correctly recovers the right answer "
        "once lambda* is far (several sigma_f) from lam_ex",
        far_from_peak, 0.0, far_from_peak, 1e-3, far_from_peak < 1e-3,
        f"rel diff vs correct at lambda*=680 (5 sigma_f past emission band): {far_from_peak:.2e} -- "
        "confirms the breakdown is a real, localized effect, not a permanently-broken formula"))

    # Physical sanity: closed form's value converges toward a real discrete
    # exact-solve as the grid is refined (slowly -- unresolved 1/sqrt
    # singularity on any finite grid, same landmine as the derivation
    # session's own raw-grid-FD warning -- so this is a trend, not a bound).
    A_mid = torch.tensor((kappa - B / 520.0 ** 2).item(), dtype=torch.float64)
    I_closed_val, _ = v9_escaping_flux(A_mid, B, cos_i, lam_max, lam_ex, lam_em,
                                        sigma_f, quantum_yield, L0, t_nodes, t_weights)
    I_closed_val = I_closed_val.item()
    Ns = [400_000, 2_000_000, 8_000_000]
    discrete_vals = [_v9_discrete_reference(A_mid, B, cos_i, lam_max, lam_ex, lam_em,
                                             sigma_f, quantum_yield, L0, N) for N in Ns]
    rel_errs = [abs(v - I_closed_val) / abs(I_closed_val) for v in discrete_vals]
    converging = all(rel_errs[i] > rel_errs[i + 1] for i in range(len(rel_errs) - 1))
    results.append(StructResult(
        "V9", "closed form's VALUE converges toward a real discrete exact-solve as N increases",
        rel_errs[-1], 0.0, rel_errs[-1], 0.05, converging and rel_errs[-1] < 0.05,
        f"rel err by N {dict(zip(Ns, [f'{e:.2e}' for e in rel_errs]))} -- monotonically improving, "
        "confirms the closed form matches real physics, not just internal self-consistency"))

    return results


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
