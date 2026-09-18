# ParticleAHFinderX

ParticleAHFinderX is a CarpetX apparent-horizon finder under development. Its
surface points are persistent AMReX particles and its production search path
uses portable AMReX accelerator APIs rather than the CPU global interpolator.

The current implementation contains persistent mesh-owned particles, a
separate owner-GPU logical sphere, finest-valid-level native ADM gathering,
GPU-aware routing, sixth-order angular operators, outgoing-expansion
evaluation, internal SSPRK3/RK4 relaxation, tracking and pairwise provisional
common-surface candidates, restart sidecars, native and compatibility
diagnostics, openPMD/TSV visualization output, and an optional bounded
SphericalSurface export. Proposed tracking motion is preflighted against the
non-periodic physical domain; a repeatedly unsupported surface is explicitly
retired instead of being lost during redistribution. Configured populations
are born once and reused in place; a new population is allocated only when a
new candidate is born and is removed only by an explicit retirement
transition.

- Start with [HANDOFF.md](HANDOFF.md) for the objective, reviewed references,
  proposed algorithm, constraints, decisions, and open questions.
- See [PLAN.md](PLAN.md) for milestone and validation gates.
- See [docs/architecture.md](docs/architecture.md) for the proposed data flow
  and ownership model.
- See [docs/reference-audit.md](docs/reference-audit.md) for the FlowTracerX,
  AHFinderDirect, ET_BHaHAHAX, AHFinderX, CarpetX, SphericalSurface, and
  QuasiLocalMeasures review.
- See [docs/validation.md](docs/validation.md) for correctness, restart, AMR,
  coupling, and scaling requirements.
- See [docs/output.md](docs/output.md) for parallel BP5 particle snapshots,
  debug TSV, and the ParaView/movie conversion path.
- See [docs/configuration.md](docs/configuration.md) for surface counts,
  solver controls, candidate lifecycle, center providers, and coupling.
- See [docs/solver-debugging.md](docs/solver-debugging.md) for the current
  close-four-hole failure analysis and the required next validation sequence.
- See [docs/angular-continuation.md](docs/angular-continuation.md) for the
  persistent coarse-to-fine particle-layer design.
- See [validation/README.md](validation/README.md) for the CPU/GPU executable
  distinction and accelerator smoke-test launchers.

Surface resolution is configured independently per horizon with `ntheta[i]`
and `nphi[i]`. The current latitude/longitude topology requires
`nphi = 2*ntheta` and creates exactly `ntheta*nphi` persistent particles. The
default `32*64` surface therefore has 2,048 particles. This count is distinct
from the cubic mesh interpolation stencil: each particle reads 64 mesh samples
from a local `4*4*4` stencil without creating additional particles.

The current source revision compiles with both CUDA-enabled and CPU/OpenMP
AMReX configurations. GPU-first validation on DeltaAI exercised persistent
creation/reuse, fused interpolation and live ADM gathering, angular operators,
both internal relaxation integrators, all twelve available physical ODE
methods, provisional-candidate rejection, domain-exit destruction, 256
simultaneous surfaces, one- and two-GPU BP5 output, and changed-rank
checkpoint/recovery. The final CPU run repeated the portable primitives,
domain-exit path, and 256-surface batch. The current development source uses
checkpoint schema 12, particle-output schema 6, and native-diagnostic schema
7. Persistent coarse-to-fine angular layers, owner-GPU cubic prolongation,
per-layer timesteps, active-layer-only mesh gathering/routing, and a
high-frequency resolution-adequacy gate are implemented in the current
development source. Direct fine-grid physical relaxation approaches each
individual horizon. In the first continuation run, the 8x16 individual layer
reached tolerance before a stale reverse-routing address aborted its 16x32
promotion. The compact device metadata-refresh fix passed that transition;
the rollback-aware redistribution and ordered target-layer bookkeeping fixes
then passed four-GPU job 3168269. That run converged the 8x16 and 16x32 layers,
and reduced the 32x64 individual residual to `M*L2=6.33e-5` and
`M*Linf=9.10e-5` in 3,000 target steps, with mean radius `0.213659` versus the
independent Newton result `0.213641`. The full GPU solve took 47.7 seconds but
the strict L2 target was not yet reached. Portable, accepted-only trajectory
extrapolation now advances the full shape on continuation source layers and
only the sin(theta)-weighted constant mode on the target layer. Four-GPU job
3172404 preserved nonconstant target height modes, reduced the target endpoint
to `M*L2=2.000e-5`, `M*Linf=3.512e-5`, and took 29.6 seconds. This remains an
improved 3,000-step cutoff rather than target convergence. Four-GPU job
3172580 used a bounded 4,000-step ceiling and converged all four individual
horizons near target step 3,802 in 34.3 seconds; it exported all four through
the SphericalSurface bridge. The speculative common seed correctly remained
unpublished. Moving and merger cases, the new integration tests, and HIP/SYCL
builds remain validation work. The
launchers under `validation/` are machine-specific test harnesses, not
production backend dependencies.

Convert a production BP5 output directory to connected ParaView surfaces with:

```bash
python3 tools/particleahfinderx_to_vtk.py particle-output \
  --output-dir particleahfinderx-vtk
```

This requires the `openpmd-api` Python module with ADIOS2 support. For a small
debug run, the same topology path can consume the bounded rank-local TSV files
with `--input-format tsv`. The generated `.pvd` file opens the complete time
series in ParaView.

`AHFinderDirect/` and `ET_BHaHAHAX/` are read-only reference snapshots. Project
workflow and safety rules are in [AGENTS.md](AGENTS.md).
