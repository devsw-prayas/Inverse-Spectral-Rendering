# Differentiable Bispectral Rendering

Research code for a differentiable renderer that correctly handles materials
that shift light between wavelengths — fluorescence and thin-film interference
(soap bubbles, oil slicks, iridescent coatings) — instead of just reflecting
each wavelength back at itself.

## Why this matters

Most differentiable renderers assume light of wavelength λ only ever scatters
back out at wavelength λ. That's true for ordinary reflective materials, but
it breaks down for anything that shifts color: a fluorescent dye absorbs blue
light and re-emits it as green; a thin film interferes light across
wavelengths depending on its thickness. Modeling this properly requires a
scattering kernel `K(λ, λ')` that is *dense* — every input wavelength can
contribute to every output wavelength — rather than diagonal.

Once the kernel is dense, differentiating the render with respect to a scene
parameter (say, film thickness or dye concentration) gets subtle. A naive
automatic-differentiation pass differentiates the integrand but silently
ignores that the integration domain itself moves as the parameter changes
(e.g. the total-internal-reflection boundary shifts). That missing term
produces a gradient that is *wrong*, not just approximate. This project
derives and validates the correct gradient estimator, including that missing
domain-motion (velocity) term.

## What's here

This repository is a Python research prototype: closed-form derivations
implemented and cross-checked against automatic differentiation and numerical
finite differences, all in `float64` for precision. It includes a real
(deterministic — Neumann-series/quadrature, not Monte Carlo) multi-bounce
forward renderer and sensor model, used throughout the V-series to validate
gradients on full scenes, not just individual components. There is no
stochastic path tracer yet — that's the Phase 2 C++ implementation this
Python oracle is groundwork for.

## Repository layout

```
src/
  spectral_grid.py     spectral discretization (wavelength sampling)
  cauchy_ior.py         dispersive index of refraction n(λ) = A + B/λ²
  fresnel.py             Fresnel reflectance/transmittance
  snell_jacobian.py       per-vertex refraction Jacobian, incl. total-internal-reflection handling
  kernels.py               scattering kernels K(λ,λ') for reflectance / fluorescence / thin-film
  forward.py                 forward light-transport solve (Neumann series / exact Fredholm solve)
  gradient.py                 gradients (analytic + adjoint) and a finite-difference reference oracle
  scenes.py                     test scene construction
  sensor.py                     sensor / measurement model
  check_env.py                 environment sanity check (dtype, CUDA, package versions)

tests/
  harness.py     shared result types and a "three-way" test runner
  test_A.py       closed-form / symbolic proofs (no rendering)
  test_T.py       small numerical checks of individual components
  test_G.py       parameter sweeps that double as validation figures
  test_V.py       full-scene checks against the forward-rendering oracle
```

Each test module runs standalone and prints a pass/fail table. Where
applicable, a test validates a quantity three independent ways: closed-form
analytic result vs. `torch` automatic differentiation vs. central finite
differences — the three should agree to numerical precision.

## What each test checks

A few terms used below: **TIR** (total internal reflection) is the point
where light hitting an interface from the denser side stops transmitting and
reflects entirely, past a "critical angle." Reflectance/transmittance
formulas and their derivatives can behave badly right at that boundary if
they aren't written carefully — several tests exist specifically to check
that edge.

**A-series — closed-form proofs** (no numerical rendering; sympy or exact
tensor algebra):
- **A1** — the thin-film reflectance formula always stays within the
  physically valid range [0, 1].
- **A2** — a zero-thickness film has no interference effect, and its
  reflectance-vs-thickness derivative vanishes smoothly there (no kink).
- **A3** — the fluorescence kernel's operator norm bound is achieved exactly
  at the predicted worst-case input, not just approximately.
- **A4** — the naive (wrong) adjoint kernel only happens to be correct in the
  trivial case of zero wavelength shift; everywhere else it's provably wrong.
- **A5** — near the TIR critical angle, the transmitted angle vanishes as a
  square root of distance from the critical point, not linearly.
- **A6** — the refraction Jacobian's components collapse to the expected
  values at normal incidence, and diverge/vanish correctly near grazing angles.
- **A7** — the refraction Jacobian's two TIR-limit formulas (one per light
  polarization) agree with each other exactly when there's no index mismatch.
- **A8** — Brewster's angle (where reflectance drops to zero) is a distinct
  phenomenon from the TIR critical angle; they never coincide.
- **A9** — absorbed fluorescence power goes cleanly to zero as the
  fluorophore's spectral width shrinks to zero — no artificial floor value.
- **A10** — the critical wavelength's sensitivity to the dispersion
  parameters behaves consistently as the geometry approaches normal incidence.
- **A11** — the render is symbolically invariant to which excitation
  wavelength is used, generalized to multiple fluorescent species at once.
- **A12** — the well-posedness bound used elsewhere is sufficient but not
  necessary: a Cauchy-Schwarz upper bound, not the tight condition, so it can
  reject scenes that are actually fine.
- **A13** — Monte Carlo sampling-density gradients cancel exactly under the
  standard estimator design; not a live gap in this repo (no stochastic
  sampling here yet), carried forward as a correctness requirement for the
  future C++ tracer.

**T-series — small numerical checks** (single components, no full scene;
`T0` is the build gate, run first):
- **T0** — energy conservation: reflected + transmitted power sums to exactly
  1 at a lossless interface.
- **T1** — the reflectance-as-operator behaves correctly even for wavelengths
  right at the edge of the sampled spectral range.
- **T2** — demonstrates that "each column of a matrix sums to 1" alone is not
  enough to bound how much the matrix can amplify a signal (motivates the
  stability bound actually used elsewhere).
- **T4** — the refraction Jacobian's determinant formula holds on both sides
  of the TIR critical angle.
- **T5** — the refraction Jacobian stays finite exactly at the TIR onset, but
  must not be evaluated past it without an explicit guard.
- **T6** — film thickness becomes impossible to recover once the substrate's
  index of refraction matches the film's (no contrast, no signal).
- **T9** — the recoverability (Jacobian rank) of scene parameters actually
  drops, not just gets ill-conditioned, when fluorescence width and film
  thickness both shrink to zero together.
- **T11** — parameter recovery becomes exactly rank-deficient (not just
  poorly conditioned) when two fluorophores' emission peaks exactly coincide.
- **T12** — a scene with both a near-TIR interface and a moving integration
  boundary combines both correction terms correctly.
- **T13** — a naive numerical clamp in the code hides a real gradient
  discontinuity at a substrate-side critical angle.
- **T14** — excitation-wavelength invariance (A11's claim) degrades near the
  edge of the simulated spectral window — a genuine scope limit, not a bug.
- **T15** — the moving-boundary correction term goes smoothly to zero at
  normal incidence instead of producing an indeterminate `0 * inf`.

**G-series — parameter sweeps** (each one saves a CSV to
[`results/figures/`](results/figures) intended to become a paper figure):
- **G1** — full sweep confirming the TIR-limit Jacobian formula lands exactly
  on its predicted value for both polarizations, with no kink.
- **G2** — refraction Jacobian components swept over incidence angle for
  several index ratios.
- **G3** — the gradient stays smooth and continuous as the critical
  wavelength sweeps across the edges of the measurement window.
- **G4** — recovery conditioning worsens smoothly (no plateau) as two
  fluorophore emission peaks are moved closer together.
- **G5** — a heatmap of recovery conditioning over emission-peak separation
  and number of fluorescent species.
- **G6** — adding more measurement angles improves parameter recoverability,
  with diminishing returns.
- **G7** — how recovery conditioning behaves as substrate/film index
  contrast shrinks toward zero.
- **G8** — recovery conditioning vs. film thickness shows periodic structure
  tied to the interference fringe spacing.
- **G9** — sensitivity of absorbed power to excitation wavelength grows
  smoothly from exactly zero (not with a jump).
- **G10** — how many scene parameters are recoverable as a function of
  measurement spectral bandwidth.
- **G11** — using the wrong (transposed) adjoint kernel produces a gradient
  error that grows with the fluorescence wavelength shift, vanishing only
  when there's no shift at all.
- **G12** — *(deferred to Phase 2)* would confirm excitation-wavelength
  invariance in a rendered image within Monte Carlo noise, but there's no
  stochastic estimator in this repo yet to generate that noise — a toy
  wrapper around the existing closed form would just show a flat line,
  testing "noise around a known constant" rather than real estimator
  robustness. Covered in the meantime by G9 and T14.

**V-series — full-scene validation against the forward oracle** (V1, V2, V3,
V5, V6, V8, V9, V13 run today; V4 and V7 need real Monte Carlo sampling
variance and are Phase-2 stubs; V10–V12 need the future C++ renderer and are
stubbed):
- **V1** — a closed cavity containing a TIR interface reaches uniform
  thermal equilibrium (a "furnace test"); isolates a bug specific to the TIR
  limit.
- **V2** — a rendered image is exactly unchanged by excitation-wavelength
  shifts in a flat-illuminated, single-bounce fluorescent scene.
- **V3** — on a full scene, automatic differentiation, the correct adjoint
  gradient, and finite differences all agree, for every parameter; the wrong
  adjoint fails on all of them.
- **V4** — *(Phase 2 — needs real sampling variance)* naive vs. corrected
  gradient near criticality combined with a moving critical wavelength.
- **V5** — gradient-descent (Levenberg-Marquardt) parameter recovery variance
  under measurement noise matches what the Jacobian's conditioning predicts,
  for both the substrate-confound and the B-bottleneck cases.
- **V6** — an "inverse crime" check: fitting a bounce-depth-truncated model to
  data generated by the true (converged multi-bounce) model shows a
  structurally bad fit, confirming V5's recovery success wasn't circular.
- **V7** — *(Phase 2 — needs real sampling strategies)* compares three
  gradient-estimator designs for bias near the singular (TIR/rank-deficient)
  manifold.
- **V8** — checks for spurious beat-pattern aliasing in an actual rendered
  image when film thickness is swept near resonance with the wavelength
  sampling — none found, at any production sampling rate.
- **V9** — the full end-to-end falsifier: a fluorescent-behind-glass scene
  with the critical wavelength swept through the emission band. The naive
  boundary-only correction breaks down right at the boundary itself (an
  integrable singularity in the escaping flux); fixed via a second,
  Theorem-3-style substitution that maps the singular integral onto a fixed,
  smooth domain so automatic differentiation handles it directly.
- **V10–V12** — *(Phase 2, require the future C++ path tracer)* cross-check
  the C++ implementation against this Python oracle, verify multi-strategy
  sampling is unbiased, and confirm the test suite catches a deliberately
  reintroduced bug.
- **V13** — *(Phase 1.1 addition, added after the original Phase 1 close)*
  when two fluorescent species share one TIR-bounded interface, a species
  sitting near the critical-angle boundary measurably perturbs a second,
  spectrally distant species' recovered amplitude — purely through the
  shared Woodbury operator inverse, not any direct spectral overlap between
  the two species. A negative control (moving the second species' absorption
  off the first species' emission band) collapses the effect, confirming the
  mechanism is genuinely spectral-overlap-gated.

## Environment

Conda env `Spectral`:

| Package | Version |
|---|---|
| Python | 3.11.14 |
| pytorch | 2.5.1 (CUDA 12.4 / cuDNN 9) |
| numpy | 2.0.1 |
| scipy | 1.16.0 |
| matplotlib | 3.10.8 |

```
conda activate Spectral
python src/check_env.py
```

All numerical code runs in `torch` `float64`
(`torch.set_default_dtype(torch.float64)` is set globally).

## Running tests

```
conda activate Spectral
python -m tests.test_A
python -m tests.test_T
python -m tests.test_G
python -m tests.test_V
```
