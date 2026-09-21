# ParticleAHFinderX

[![AMReX](https://amrex-codes.github.io/badges/powered%20by-AMReX-red.svg)](https://amrex-codes.github.io)

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
- Tracks existing horizons and manages provisional pairwise common-horizon
  candidates
- Supports regridding, checkpoint/restart, stable surface IDs, and explicit
  retirement when a horizon leaves the domain
- Writes diagnostics and particle data for visualization, with optional
  SphericalSurface compatibility output

## Available Module

- `ParticleAHFinderX` — Apparent-horizon finding, tracking, diagnostics, and
  particle output

## Getting Started

- Follow the [CarpetX getting-started guide](https://github.com/EinsteinToolkit/CarpetX/wiki/Getting-Started)
  to build an accelerator-enabled Einstein Toolkit configuration.
- Add `ParticleAHFinderX/ParticleAHFinderX` to the configuration thornlist.
- See the [thorn README](ParticleAHFinderX/README) for runtime options and
  center-tracking policies.
- Example parameter files are available under
  [`ParticleAHFinderX/test`](ParticleAHFinderX/test/).

Surface `i` uses `ntheta[i] * nphi[i]` persistent particles, with
`nphi[i] = 2 * ntheta[i]`. The default `32 * 64` surface contains 2,048
particles. Coarse-to-fine angular continuation can solve at `8 * 16`, then
`16 * 32`, and finally `32 * 64`.

## Current Status

The code has been exercised with CUDA and CPU/OpenMP AMReX builds. Searches
currently run at fully synchronized CarpetX hierarchy times when subcycling is
enabled. Moving-merger validation, active-level-band searches, and non-CUDA
accelerator validation remain in progress.

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
