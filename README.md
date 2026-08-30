# Differentiable Bispectral Rendering

Research code for a differentiable renderer that correctly handles materials
which move light from one wavelength to another, fluorescence and thin-film
interference (soap bubbles, oil slicks, iridescent coatings), rather than
assuming every wavelength reflects straight back at itself.

## Why this matters

Most differentiable renderers assume light of wavelength λ only ever comes
back out at wavelength λ. That holds for ordinary reflective surfaces, but
not for anything that shifts colour: a fluorescent dye absorbs blue and
re-emits green; a thin film mixes wavelengths depending on its thickness.
Handling this needs a scattering description `K(λ, λ')` that is *dense*, any
incoming wavelength can feed any outgoing wavelength, instead of one that
only connects each wavelength to itself.

Once wavelengths are coupled, taking the derivative of a rendered image with
respect to a scene parameter (film thickness, dye concentration) becomes
delicate. A naive automatic-differentiation pass differentiates the quantity
being integrated but misses that the region of integration also moves as the
parameter changes; for instance, the angle past which light is trapped by
total internal reflection shifts. The dropped term makes the gradient
*wrong*, not just inaccurate. This project works out and checks the correct
gradient, including that missing moving-boundary term.

## What's here

A Python research prototype. Each result is derived by hand in closed form
and then cross-checked against automatic differentiation and finite
differences, all in `float64`. It includes a real multi-bounce forward
renderer and sensor model, deterministic (it solves the transport equation
directly rather than sampling paths), used across the V-series to check
gradients on whole scenes, not just isolated pieces. There is no random path
tracer yet; that is the Phase 2 C++ build this code is groundwork for.

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
applicable, a test checks a quantity three independent ways: the closed-form
result, `torch` automatic differentiation, and central finite differences,
which should agree to numerical precision.

## What each test checks

**TIR** (total internal reflection) below means the angle past which light
hitting an interface from the denser side stops passing through and reflects
completely. Several of the formulas, and their derivatives, are easy to get
wrong right at that angle; a number of tests exist just to pin that edge down.

**A-series, closed-form proofs** (symbolic algebra, no rendering):
- **A1** the thin-film reflectance formula never leaves the physical range
  [0, 1].
- **A2** a film of zero thickness has no effect, and its
  reflectance-vs-thickness slope goes smoothly to zero there (no kink).
- **A3** the largest factor by which the fluorescence kernel can amplify a
  spectrum matches the predicted bound, and is reached by the input the
  theory predicts.
- **A4** the shortcut form of the adjoint kernel is only correct when
  absorption and emission have the same shape (no wavelength shift);
  otherwise it is provably wrong.
- **A5** near the TIR angle the transmitted-ray angle falls off like the
  square root of the distance from that angle, not linearly.
- **A6** the refraction Jacobian takes its expected values at straight-on
  incidence and diverges or vanishes correctly near grazing angles.
- **A7** the two TIR-limit formulas for the refraction factor (one per
  polarization) agree exactly when the two media have the same index.
- **A8** Brewster's angle (where reflection drops to zero) and the TIR angle
  are different angles and never coincide.
- **A9** absorbed fluorescence power goes cleanly to zero as the dye's
  spectral width shrinks, with no spurious floor.
- **A10** the trapped-light cutoff wavelength responds to the dispersion
  parameters consistently as the geometry approaches straight-on incidence.
- **A11** the rendered result does not depend on which excitation wavelength
  is assumed, for any number of fluorescent species.
- **A12** the well-posedness condition used elsewhere is safe but
  conservative: it can reject scenes that would actually be fine.
- **A13** the sampling-probability terms in a Monte Carlo gradient cancel
  under the standard estimator; not relevant to this repo yet (no random
  sampling), noted as a requirement for the future C++ tracer.
- **A14** at the moving TIR angle, the raw parameter-derivative of the
  integrand blows up with no common bound as the angle is approached, but a
  square-root change of variable that pins the critical angle in place
  removes the blow-up entirely; checked symbolically and for both
  polarizations across the dispersion parameters.

**T-series, small numerical checks** (one component at a time, no full
scene; `T0` is the build gate):
- **T0** energy conservation: reflected plus transmitted power is exactly 1
  at a lossless interface.
- **T1** the reflectance step behaves correctly even for wavelengths right
  at the edge of the sampled range.
- **T2** shows that "every column sums to 1" is not enough to limit how much
  a matrix can amplify a signal, which is why a stronger condition is used
  elsewhere.
- **T4** the refraction determinant formula holds on both sides of the TIR
  angle.
- **T5** the refraction factor stays finite exactly at the TIR onset but
  must not be evaluated past it without a guard.
- **T6** film thickness becomes impossible to recover once the substrate
  index matches the film index (no contrast, no signal).
- **T9** the number of recoverable parameters actually drops, not just gets
  harder, when fluorescence width and film thickness both shrink to zero.
- **T11** parameter recovery becomes exactly impossible, not just hard, when
  two dyes' emission peaks coincide.
- **T12** a scene with both a near-TIR interface and a moving integration
  edge needs both correction terms, and they combine correctly.
- **T13** a numerical clamp in the code hides a real gradient jump at a
  substrate-side TIR angle.
- **T14** the excitation-wavelength independence from A11 breaks down near
  the edge of the simulated wavelength window, a real limit, not a bug.
- **T15** the moving-boundary term goes smoothly to zero at straight-on
  incidence instead of producing `0 * inf`.

**G-series, parameter sweeps** (each saves a CSV to
[`results/figures/`](results/figures) meant to become a paper figure):
- **G1** confirms the TIR-limit refraction formula lands exactly on its
  predicted value for both polarizations, with no kink.
- **G2** refraction Jacobian components swept over incidence angle for
  several index ratios.
- **G3** the gradient stays smooth as the trapped-light cutoff wavelength
  sweeps across the edges of the measurement window.
- **G4** how ill-determined recovery gets as two dye emission peaks are
  brought together (smoothly worse, no plateau).
- **G5** a heatmap of recovery difficulty over emission-peak spacing and
  number of dyes.
- **G6** more measurement angles make parameters easier to recover, with
  diminishing returns.
- **G7** how recovery difficulty behaves as substrate/film index contrast
  goes to zero.
- **G8** recovery difficulty vs. film thickness shows periodic structure
  matching the interference fringe spacing.
- **G9** the sensitivity of absorbed power to excitation wavelength grows
  smoothly from exactly zero, with no jump.
- **G10** how many parameters are recoverable as the measured wavelength
  band widens.
- **G11** using the wrong (transposed) adjoint kernel gives a gradient error
  that grows with the fluorescence wavelength shift and vanishes only when
  there is no shift.
- **G12** *(deferred to Phase 2)* would check excitation-wavelength
  independence in a rendered image under Monte Carlo noise, but there is no
  random estimator here yet; a toy version would only test noise around a
  known constant. Covered for now by G9 and T14.

**V-series, whole-scene checks against the forward renderer** (V1, V2, V3,
V5, V6, V8, V9, V13 run today; V4 and V7 need real sampling noise and are
Phase-2 stubs; V10-V12 need the future C++ renderer):
- **V1** a sealed cavity with a TIR interface settles to a uniform
  temperature (a "furnace test"); catches bugs specific to the TIR limit.
- **V2** a rendered image is exactly unchanged when the assumed excitation
  wavelength is shifted, in a flat-lit single-bounce fluorescent scene.
- **V3** on a full scene, automatic differentiation, the correct adjoint
  gradient, and finite differences all agree for every parameter; the wrong
  adjoint fails on all of them.
- **V4** *(Phase 2, needs real sampling noise)* naive vs. corrected gradient
  near the TIR angle together with a moving cutoff wavelength.
- **V5** parameter recovery under measurement noise scatters the way the
  problem's conditioning predicts, for both the substrate-confound and the
  hard-to-recover-B cases.
- **V6** an "inverse crime" check: fitting a shortened-bounce model to data
  from the full model gives a structurally bad fit, so V5's success was not
  circular.
- **V7** *(Phase 2, needs real sampling strategies)* compares three
  gradient-estimator designs for bias near the difficult (TIR /
  hard-to-recover) region.
- **V8** looks for spurious beat patterns in a rendered image when film
  thickness is swept near resonance with the wavelength sampling; none
  found, at any production sampling rate.
- **V9** the main end-to-end falsifier: a fluorescent-behind-glass scene
  with the cutoff wavelength swept through the emission band. The simple
  boundary correction breaks down right at the boundary (the escaping light
  is infinite there, but integrably so); fixed with a change of variable
  that turns the singular integral into a smooth one automatic
  differentiation can handle.
- **V10-V12** *(Phase 2, need the C++ path tracer)* cross-check the C++
  build against this Python version, check that multi-strategy sampling is
  unbiased, and confirm the suite catches a deliberately reintroduced bug.
- **V13** *(Phase 1.1)* when two fluorescent species share one TIR-bounded
  interface, a species sitting near the TIR angle measurably shifts a
  second, spectrally distant species' recovered brightness, purely through
  the shared feedback in the solver, not any direct spectral overlap. A
  control that moves the second species' absorption off the first's emission
  band removes the effect.

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
