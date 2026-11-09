# ParticleAHFinderX

[![AMReX](https://amrex-codes.github.io/badges/powered%20by-AMReX-red.svg)](https://amrex-codes.github.io)
[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-1fa669?style=flat&colorA=080f12&colorB=1fa669)](https://deepwiki.com/jaykalinani/ParticleAHFinderX)

**ParticleAHFinderX** is a GPU-accelerated apparent-horizon finder for
dynamical spacetimes. It is written in C++ for the
[Einstein Toolkit](https://einsteintoolkit.org/) and uses the
[CarpetX](https://github.com/EinsteinToolkit/CarpetX) driver and
[AMReX](https://amrex-codes.github.io) portable accelerator APIs.

## Overview

- Represents each trial surface with persistent AMReX particles
- Keeps surface evolution, interpolation, expansion evaluation, and reductions
  on the accelerator
- Samples the finest valid AMR level with device-side cubic interpolation
- Solves for star-shaped marginally outer trapped surfaces using GPU
  hyperbolic relaxation with SSPRK3 or RK4
- Tracks existing horizons and manages provisional common-horizon candidates,
  including bounded hierarchical composition for multi-hole systems
- Records a compact candidate decision trail for close encounters, including
  birth, failed attempts, promotion, and retirement reasons
- Supports regridding, checkpoint/restart, stable surface IDs, and explicit
  retirement when a horizon leaves the domain
- Writes native diagnostics and particle data for visualization without a
  fixed SphericalSurface handoff
- Provides dynamic stable-ID quasi-local records without fixed
  SphericalSurface slots through the companion `AH_analysis` thorn

The production horizon pipeline is `ParticleAHFinderX -> AH_analysis`.
AHFinderDirect, SphericalSurface, and QuasiLocalMeasures are independent
validation references and are not production finder or horizon-analysis
dependencies. SphericalSurface may still be activated separately by a
gravitational-wave extraction workflow without coupling it to either thorn.

## Available Modules

- `ParticleAHFinderX` — Apparent-horizon finding, tracking, diagnostics, and
  particle output
- `AH_analysis` — GPU-native area, mass, coordinate momentum/spin, intrinsic
  scalar curvature, normal-bundle diagnostics, and optional variational
  invariant spin, invariant coordinate, degree-eight mass/spin multipoles, and
  finite-radius coordinate fluxes for configured and dynamically created
  horizons

## Getting Started

- Follow the [CarpetX getting-started guide](https://github.com/EinsteinToolkit/CarpetX/wiki/Getting-Started)
  to build an accelerator-enabled Einstein Toolkit configuration.
- Add `ParticleAHFinderX/ParticleAHFinderX` and
  `ParticleAHFinderX/AH_analysis` to the configuration thornlist.
- See the [thorn README](ParticleAHFinderX/README) for runtime options and
  center-tracking policies.
- See the [analysis README](AH_analysis/README.md) for the implemented
  quasi-local formulas and current scope.
- See the [invariant-analysis contract](docs/invariant-analysis.md) for the
  geometry, sign, eigensolver, and validation design of the GPU-native
  approximate-Killing-vector spin layer.
- See the [QuasiLocalMeasures parity matrix](docs/qlm-parity.md) for the exact
  implemented, intentionally different, and still-open analysis quantities.
- See the [four-dimensional analysis contract](docs/four-dimensional-analysis.md)
  for the remaining tetrad, Newman--Penrose, Ricci/matter, and Weinberg layers.
- See the [implementation plan](PLAN.md) for the ordered validation and
  GPU-native analysis goals.
- Example parameter files are available under
  [`ParticleAHFinderX/test`](ParticleAHFinderX/test/).

Surface `i` uses `ntheta[i] * nphi[i]` persistent particles, with
`nphi[i] = 2 * ntheta[i]`. The default `32 * 64` surface contains 2,048
particles. Coarse-to-fine angular continuation can solve at `8 * 16`, then
`16 * 32`, and finally `32 * 64`.

## Current Status

The current CUDA-validated build supports persistent surface populations,
four-hole acquisition and tracking, synchronized active-level-band searches,
moving-AMR checkpoint/recovery, domain-exit retirement, and dynamic stable-ID
analysis. Schema 6 analysis has passed the final four-hole same-slice Oracle
comparison, including intrinsic geometry, invariant spin, degree-eight
multipoles, and coordinate fluxes. The production source no longer contains a
SphericalSurface export adapter; checkpoint schema 15 records that interface
change. The schema-15 source, including the native output-format audit, builds
successfully in the CUDA `asterx_bhc` configuration. Four-GPU job `3437664`
passed its startup, output-format, accelerator-residency, and same-slice
correctness smoke test. Four-GPU job `3437992` then passed exact schema-15
checkpoint/recovery against an uninterrupted control.

The close-four-hole refinement study now separates angular self-convergence
from agreement with the independently discretized reference finder. Strict
job `3438372` reached `(M*L2,M*Linf)=(1e-7,1e-6)` at `16x32` and `32x64` on
two mesh resolutions. Job `3438673` added `64x128`: particle-only angular
differences decreased by factors `0.136--0.291`, while level-11 mesh
refinement reduced the reference differences by factors `0.012--0.024`.
Every completed target used zero point-host staging. The first fine job timed
out only after starting a redundant third search; the corrected harness stops
at the first shared reference slice, and job `3438788` passed cleanly in
9m19s. See the validation reports for the exact residuals and the important
performance limitation at `64x128`.

The current CUDA executable validates schema-6 intrinsic, spin, multipole,
coordinate-flux, Weyl, Weinberg, and point output on dynamic stable IDs.
Physical common-horizon birth and promotion pass with native analysis; moving
AMR, active-level-band operation, same-rank restart, and changed-rank particle
recovery also have runtime evidence. Current unbuilt source repairs the
negative candidate-analysis route and adds an active-layer-only recentered
GPU geometry cache. All 274 repository tests and the portable-source audit
pass, but a fresh CUDA build and runtime comparison remain required.

Time-evolved fly-by expiry, promotion-boundary recovery, matched production
performance, large-surface scaling, HIP, and current-source CPU portability
remain open. See [the validation index](validation/README.md),
[the ordered plan](PLAN.md), and [the handoff](HANDOFF.md) for exact job
evidence and release gates.

## Useful Repositories

- [AsterX](https://github.com/EinsteinToolkit/AsterX) — GPU-accelerated GRMHD
- [CarpetX](https://github.com/EinsteinToolkit/CarpetX) — AMReX-based Einstein
  Toolkit driver
- [FlowTracerX](https://github.com/jaykalinani/FlowTracerX) — Persistent
  AMReX-particle infrastructure
- [SpacetimeX](https://github.com/EinsteinToolkit/SpacetimeX) — Spacetime
  evolution modules

## License

ParticleAHFinderX is distributed under the GNU General Public License v3.0.
