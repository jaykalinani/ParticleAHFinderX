# AH_analysis

`AH_analysis` computes bounded quasi-local records directly from converged
`ParticleAHFinderX` surfaces. Surfaces are addressed by persistent stable ID,
so horizons created after a merger do not require a fixed
`SphericalSurface` slot.

The default path reuses the metric `gamma_ij`, extrinsic curvature `K_ij`,
outward physical unit normal `s^i`, and physical area weight `dA` already
resident on each logical-owner accelerator after a successful search. It
performs no additional mesh interpolation and transfers only a fixed-size
record per surface for output. Optional curvature diagnostics make one
separate GPU interpolation pass only when explicitly enabled.

## Quantities

The covariant coordinate momentum is

```text
P_i = (1 / 8 pi) integral [(K_ij - K gamma_ij) s^j] dA,
K = gamma^ij K_ij.
```

For any chosen coordinate origin `C`, the three coordinate-rotation spins are

```text
J_a = (1 / 8 pi) integral phi_a^i K_ij s^j dA,
phi_x = (0, -(z-C_z), y-C_y),
phi_y = (z-C_z, 0, -(x-C_x)),
phi_z = (-(y-C_y), x-C_x, 0).
```

From `A` and `J = sqrt(J_x^2 + J_y^2 + J_z^2)`, the thorn reports

```text
R_A   = sqrt(A / 4 pi),
M_irr = sqrt(A / 16 pi),
M_H   = sqrt(M_irr^2 + J^2 / (4 M_irr^2)),
chi   = J / M_H^2.
```

Both the radial-graph center used by the finder and the physical area centroid
are reported as origins. The former permits a direct comparison with the
existing SphericalSurface/QLM coordinate-spin diagnostic; the latter is used
for `J`, `M_H`, and `chi` above.

These spins use coordinate rotation vectors, matching a useful existing
quasi-local diagnostic but not the invariant approximate-Killing-vector spin.

The schema-2 intrinsic layer also forms

```text
q_AB    = gamma_ij e_A^i e_B^j,
omega_A = -e_A^i K_ij s^j,
```

and reports `integral R^(2) dA`, its signed Gauss--Bonnet residual relative to
`8 pi`, and the area-normalized L2 norm of `omega_A`. A separate validity flag
and invalid-point count prevent failed intrinsic geometry from being mistaken
for a physical zero. The scalar-curvature operator differentiates the regular
connection difference relative to the unit sphere, not the polar-singular raw
Christoffel symbols.

When `compute_invariant_spin=yes`, schema 3 additionally minimizes the
trace-free Killing deformation in a configurable real-spherical-harmonic trial
space, solves the projected generalized eigenproblem on the owner GPU, fixes
the axial period through `z_max-z_min=A/(2 pi)`, and reports signed invariant
spin, physical axis, Christodoulou mass, dimensionless spin, eigen residual,
Killing residual, and normalization residual. Failure is explicit; coordinate
spin is never substituted.

When `compute_multipoles=yes`, schema 4 also forms a mean-free dimensionless
invariant coordinate `zeta` whose extrema span two and computes

```text
I_l = integral [R^(2)/4] P_l(zeta) dA,
L_l = integral [phi^i K_ij s^j] P'_l(zeta) dA,
M_l = I_l M_H R_A^l / (2 pi),
J_l = L_l R_A^(l-1) / (8 pi),       0 <= l <= 8.
```

The Legendre polynomial and derivative recurrence is pole regular. The output
checks `M_0`, `J_0`, and `J_1=J_AKV` identities explicitly. Multipoles require
`compute_invariant_spin=yes`; parameter checking rejects an inconsistent
configuration. Curvature quantities remain a separately selectable layer.

When `compute_coordinate_fluxes=yes`, schema 5 appends finite-radius ADM and
conformal-ADM energy, linear momentum, and angular momentum. These are
coordinate-dependent surface fluxes, not intrinsic horizon charges. The
physical-normal ADM angular momentum uses the global Cartesian origin; the
conformal form uses the tracked graph center. The definitions and orientation
contract are in
[finite-radius coordinate fluxes](../docs/coordinate-fluxes.md). This option
reuses the resident metric, metric derivatives, curvature, normal, and area
weight, so it needs no new interpolation or point workspace.

When `compute_weyl_curvature=yes`, schema 6 makes one optional cubic mesh-owner
pass for metric Hessians and curvature gradients, forms the electric and
magnetic Weyl tensors there, and routes only their twelve independent real
components to the logical-owner GPU. It reports area-weighted means of
`Psi_0`--`Psi_4`, `I`, `J`, the speciality index, and a normalized speciality
difference. Core Weyl and both speciality diagnostics have independent
validity counts. The 54 higher derivatives are transient kernel values: they
are neither persistent particle fields nor MPI payloads. See
[the Weyl-curvature contract](../docs/weyl-curvature.md).

When `compute_weinberg_fluxes=yes`, a separate optional GPU gather samples the
native vertex-centred lapse, shift, and their coordinate-time derivatives,
while analytic derivatives of the same cubic interpolant supply the spatial
gauge derivatives. The logical-owner GPU reconstructs the full four-metric and
its first derivatives consistently with the evolved gauge, then integrates
Weinberg energy, Cartesian momentum, and Cartesian angular momentum. These are
coordinate diagnostics, not intrinsic horizon charges. They are written to
`weinberg_fluxes.tsv` under the same dynamic stable IDs; angular point data do
not reach the host. The legacy unit-lapse/zero-shift expression is not used in
general gauge.

## Output conventions

`ah_analysis.tsv` contains one row per dynamic `(iteration, stable_id)`, not a
startup-sized array of fixed surface slots. The numbered header retains the
established AHFinder/QLM names and definitions for equivalent quantities:
`time` is `cctk_time`, `mean_expansion` is the outgoing
`expansion Theta_(l)`, `irreducible_mass` is AHFinderDirect's
`m_irreducible` and QLM's `qlm_irreducible_mass`, and
`momentum_covariant_*` corresponds to QLM's
`qlm_quasilocal_momentum_covariant_*`. Coordinate, invariant-spin, multipole,
ADM-flux, and Weyl families remain explicitly distinguished where their
definitions differ.

The file also decodes lifecycle, role, and invariant-spin status codes. Every
optional family begins with requested/valid/status fields. A disabled family
writes a finite zero payload with `requested=0` and `valid=0`; the header makes
clear that those values are unavailable rather than measured physical zeros.

## Numerical path

1. `ParticleAHFinderX` converges a target angular layer and refreshes its
   geometry on the accelerator.
2. Portable AMReX kernels form the coordinate and intrinsic surface densities,
   reduce 80 reals and eleven integer/status values per logical-owner
   surface, and pack them into stable-ID order without leaving the accelerator.
3. AMReX combines only those compact device records. GPU-aware MPI uses the
   device buffers directly; a platform without GPU-aware MPI stages only the
   bounded records through pinned host memory. Angular particle data are never
   gathered to rank zero. The collective contains only surfaces searched and
   not yet consumed in the current iteration; retired candidate history does
   not make future analysis calls progressively larger.
4. One bounded device-to-host copy exposes the completed records. Rank zero
   appends dynamic stable-ID rows to `ah_analysis.tsv`; recovery preserves the
   committed checkpoint iteration and removes only later abandoned rows.

Set `print_analysis_timing=yes` for validation runs to report kernel,
collective, bounded-copy, host-record, and TSV-I/O timing and byte counters.
The report explicitly records zero angular-point host bytes; only compact
per-surface records may cross to the host.

CarpetX may traverse several synchronized level bands during one Cactus
iteration. A successful surface is consumed once in the band that searched
it, so later band traversals cannot repeat its collective or output row.

Enable the thorn with

```text
ActiveThorns = "ParticleAHFinderX AH_analysis"
ParticleAHFinderX::active = yes
AH_analysis::active = yes
AH_analysis::compute_invariant_spin = yes
AH_analysis::compute_multipoles = yes
AH_analysis::compute_coordinate_fluxes = yes
AH_analysis::compute_weyl_curvature = yes
AH_analysis::compute_weinberg_fluxes = yes
```

The invariant solve defaults to `axial_harmonic_degree=4` (24 nonconstant
real modes), at most 64 cyclic Jacobi sweeps, relative eigen tolerance
`1e-12`, and a pole-separation guard of `0.1` areal radii. The feature is off
by default. The six-case spinning-Kerr and aligned-axis QLM gates pass;
the matching six-case Schwarzschild gate also keeps spin and momentum at
roundoff. Non-axial reference-method and translation-covariance harnesses are
staged and pass their focused checker tests, but their accelerator runs remain
before production enablement. The schema-4 invariant-coordinate/multipole
extension is included in CUDA `asterx_bhc` executable
`64be8e2d77b721e7ae9cd1f73df70f0efc428c76e46f682b117080968dca6fb1`,
which has passed the Schwarzschild, Kerr/QLM, physical restart, partial
active-band, and synthetic dynamic-ID gates. Physical common-horizon promotion
and its promotion-boundary restart remain open.

The quantity-by-quantity [QLM parity matrix](../docs/qlm-parity.md) records
which reference diagnostics are implemented, definitionally different, or
still open. This thorn is not yet a full `QuasiLocalMeasures` replacement.
The open four-dimensional null-tetrad/spin-coefficient, Ricci/matter,
NP/Weyl spin, and corresponding per-point-output layers are required
capabilities before
making that claim;
they are not optional project deliverables. They will be independently
selectable at runtime so disabling an expensive layer avoids its additional
mesh gather, device workspace, communication, and output cost.
The definitions, gauge choices, field requirements, and validation gates for
those layers and for the source-complete but not yet runtime-validated
Weinberg path are recorded in
[`docs/four-dimensional-analysis.md`](../docs/four-dimensional-analysis.md).

The schema-5 coordinate-flux layer is built and passed the schema-6 four-hole
CUDA run `3437334` with valid records and zero angular-point host staging.

The schema-6 Weyl layer is built and CUDA-runtime tested. Four-GPU job
`3470370` enabled it for three complete-hierarchy searches and validated all
compact records plus three complete 8,192-point frames with zero angular-point
host staging. Width-`1.0` control job `3470457` validated the same scalar and
point fields at the iteration-128 partial active band. Matched-reference and
refinement comparisons, restart reconciliation, HIP, and CPU evidence remain
required.
