/**
 * \file analysis_api.hxx
 * \brief Bounded public analysis records for dynamic apparent horizons.
 */
#ifndef PARTICLEAHFINDERX_ANALYSIS_API_HXX
#define PARTICLEAHFINDERX_ANALYSIS_API_HXX

#include <cctk.h>
#include <cctk_Types.h>

#include <array>
#include <cstdint>
#include <vector>

namespace ParticleAHFinderX {

constexpr int surface_analysis_schema = 6;
constexpr int surface_multipole_count = 9;
constexpr int surface_analysis_point_schema = 1;

struct SurfaceComplexValue {
  CCTK_REAL real = 0;
  CCTK_REAL imag = 0;
};

enum class SurfaceRicciSource : int {
  off = 0,
  vacuum = 1,
  tmunu = 2,
};

struct SurfaceAnalysisOptions {
  bool compute_invariant_spin = false;
  bool compute_multipoles = false;
  bool compute_coordinate_fluxes = false;
  bool compute_weyl_curvature = false;
  bool compute_weinberg_fluxes = false;
  SurfaceRicciSource ricci_source = SurfaceRicciSource::off;
  int axial_harmonic_degree = 4;
  int axial_maximum_jacobi_sweeps = 64;
  CCTK_REAL axial_eigen_tolerance = 1.0e-12;
  CCTK_REAL axial_minimum_axis_separation = 0.1;
};

/**
 * One completed analysis record identified by the finder's stable surface ID.
 *
 * The finder computes the surface integrals on the logical-owner GPU from its
 * resident final metric, curvature, normal, and area-weight samples.  Only
 * this bounded record is replicated across ranks.  No angular point data are
 * exposed or transferred by this API.
 */
struct SurfaceAnalysisRecord {
  std::int64_t stable_id = 0;
  std::array<std::int64_t, 2> parent_ids{{0, 0}};
  int generation = 0;
  int lifecycle = 0;
  int role = 0;
  int ntheta = 0;
  int nphi = 0;
  int iteration = -1;
  bool published = false;
  CCTK_REAL time = 0;
  std::array<CCTK_REAL, 3> center{{0, 0, 0}};
  std::array<CCTK_REAL, 3> centroid{{0, 0, 0}};
  CCTK_REAL area = 0;
  CCTK_REAL areal_radius = 0;
  CCTK_REAL irreducible_mass = 0;
  std::array<CCTK_REAL, 3> momentum_covariant{{0, 0, 0}};
  /// Coordinate-rotation angular momentum about the radial-graph center.
  std::array<CCTK_REAL, 3> coordinate_spin_center{{0, 0, 0}};
  /// The same coordinate diagnostic translated to the area centroid.
  std::array<CCTK_REAL, 3> coordinate_spin_centroid{{0, 0, 0}};
  /// Magnitude of coordinate_spin_centroid.
  CCTK_REAL coordinate_spin_magnitude = 0;
  CCTK_REAL christodoulou_mass = 0;
  CCTK_REAL dimensionless_spin = 0;
  std::array<CCTK_REAL, 3> proper_circumference{{0, 0, 0}};
  CCTK_REAL mean_expansion = 0;
  CCTK_REAL expansion_l2 = 0;
  CCTK_REAL expansion_linf = 0;
  std::int64_t invalid_points = 0;
  bool intrinsic_geometry_valid = false;
  CCTK_REAL intrinsic_scalar_curvature_integral = 0;
  CCTK_REAL gauss_bonnet_residual = 0;
  CCTK_REAL normal_connection_l2 = 0;
  std::int64_t intrinsic_invalid_points = 0;
  bool invariant_spin_requested = false;
  bool invariant_spin_valid = false;
  int invariant_spin_status = 0;
  CCTK_REAL invariant_spin_signed = 0;
  CCTK_REAL invariant_spin_magnitude = 0;
  std::array<CCTK_REAL, 3> invariant_spin_axis{{0, 0, 0}};
  CCTK_REAL invariant_christodoulou_mass = 0;
  CCTK_REAL invariant_dimensionless_spin = 0;
  CCTK_REAL axial_eigenvalue = 0;
  CCTK_REAL axial_eigen_residual = 0;
  CCTK_REAL axial_killing_residual = 0;
  CCTK_REAL axial_normalization_residual = 0;
  int axial_iterations = 0;
  int axial_harmonic_degree = 0;
  bool invariant_coordinate_valid = false;
  CCTK_REAL invariant_coordinate_mean = 0;
  CCTK_REAL invariant_coordinate_minimum = 0;
  CCTK_REAL invariant_coordinate_maximum = 0;
  bool multipoles_requested = false;
  bool multipoles_valid = false;
  int multipole_maximum_degree = surface_multipole_count - 1;
  std::int64_t multipole_invalid_points = 0;
  std::array<CCTK_REAL, surface_multipole_count> mass_multipoles{};
  std::array<CCTK_REAL, surface_multipole_count> spin_multipoles{};
  bool coordinate_fluxes_requested = false;
  bool coordinate_fluxes_valid = false;
  std::int64_t coordinate_flux_invalid_points = 0;
  CCTK_REAL adm_energy = 0;
  std::array<CCTK_REAL, 3> adm_momentum{{0, 0, 0}};
  /// Uses the global Cartesian origin, matching the reference quantity.
  std::array<CCTK_REAL, 3> adm_angular_momentum{{0, 0, 0}};
  CCTK_REAL conformal_adm_energy = 0;
  std::array<CCTK_REAL, 3> conformal_adm_momentum{{0, 0, 0}};
  /// Uses the tracked radial-graph center as the coordinate origin.
  std::array<CCTK_REAL, 3> conformal_adm_angular_momentum{{0, 0, 0}};
  bool weyl_curvature_requested = false;
  bool weyl_curvature_valid = false;
  std::int64_t weyl_curvature_invalid_points = 0;
  std::array<SurfaceComplexValue, 5> mean_weyl_scalar{};
  SurfaceComplexValue mean_weyl_invariant_i;
  SurfaceComplexValue mean_weyl_invariant_j;
  bool speciality_index_valid = false;
  std::int64_t speciality_index_invalid_points = 0;
  SurfaceComplexValue mean_speciality_index;
  bool speciality_difference_valid = false;
  std::int64_t speciality_difference_invalid_points = 0;
  SurfaceComplexValue mean_speciality_difference;
  bool weinberg_fluxes_requested = false;
  bool weinberg_fluxes_valid = false;
  std::int64_t weinberg_flux_invalid_points = 0;
  CCTK_REAL weinberg_energy = 0;
  std::array<CCTK_REAL, 3> weinberg_momentum{{0, 0, 0}};
  std::array<CCTK_REAL, 3> weinberg_angular_momentum{{0, 0, 0}};
  SurfaceRicciSource ricci_source = SurfaceRicciSource::off;
  bool ricci_scalars_valid = false;
  std::int64_t ricci_invalid_points = 0;
  CCTK_REAL mean_ricci_phi00 = 0;
  SurfaceComplexValue mean_ricci_phi01;
  SurfaceComplexValue mean_ricci_phi02;
  CCTK_REAL mean_ricci_phi11 = 0;
  SurfaceComplexValue mean_ricci_phi12;
  CCTK_REAL mean_ricci_phi22 = 0;
  CCTK_REAL mean_ricci_lambda = 0;
};

struct SurfaceAnalysisRuntimeStatus {
  bool ready = false;
  bool recovered = false;
  int recovered_iteration = -1;
  /// True for any AMReX accelerator backend (CUDA, HIP, or SYCL).
  bool accelerator_backend = false;
  /// True when AMReX will pass the compact device buffers directly to MPI.
  bool gpu_aware_mpi = false;
};

/** Compact transport and timing data for one collective analysis call. */
struct SurfaceAnalysisPerformance {
  int global_surface_count = 0;
  int local_surface_count = 0;
  int returned_record_count = 0;
  int device_kernel_launches = 0;
  int intrinsic_geometry_kernel_launches = 0;
  int axial_symmetry_kernel_launches = 0;
  int weyl_gather_kernel_launches = 0;
  int weyl_routing_kernel_launches = 0;
  int gauge_gather_kernel_launches = 0;
  int gauge_routing_kernel_launches = 0;
  int matter_gather_kernel_launches = 0;
  int matter_routing_kernel_launches = 0;
  int axial_trial_modes = 0;
  int multipole_maximum_degree = 0;
  int device_collectives = 0;
  int explicit_device_synchronizations = 0;
  int compact_collective_host_staging_copies = 0;
  int device_to_host_copies = 0;
  int host_to_device_copies = 0;
  bool direct_device_collective = false;
  std::int64_t local_angular_points = 0;
  std::int64_t intrinsic_workspace_bytes = 0;
  std::int64_t axial_point_workspace_bytes = 0;
  std::int64_t axial_projected_workspace_bytes = 0;
  std::int64_t angular_point_host_bytes = 0;
  std::int64_t weyl_mesh_workspace_bytes = 0;
  std::int64_t weyl_logical_workspace_bytes = 0;
  std::int64_t weyl_device_payload_bytes = 0;
  std::int64_t gauge_mesh_workspace_bytes = 0;
  std::int64_t gauge_logical_workspace_bytes = 0;
  std::int64_t gauge_device_payload_bytes = 0;
  std::int64_t matter_mesh_workspace_bytes = 0;
  std::int64_t matter_logical_workspace_bytes = 0;
  std::int64_t matter_device_payload_bytes = 0;
  std::int64_t compact_collective_bytes = 0;
  std::int64_t compact_collective_host_staging_bytes = 0;
  std::int64_t compact_device_to_host_bytes = 0;
  std::int64_t compact_host_to_device_bytes = 0;
  CCTK_REAL device_kernel_seconds = 0;
  CCTK_REAL weyl_gather_seconds = 0;
  CCTK_REAL weyl_routing_seconds = 0;
  CCTK_REAL gauge_gather_seconds = 0;
  CCTK_REAL gauge_routing_seconds = 0;
  CCTK_REAL matter_gather_seconds = 0;
  CCTK_REAL matter_routing_seconds = 0;
  CCTK_REAL collective_seconds = 0;
  CCTK_REAL compact_device_to_host_seconds = 0;
  CCTK_REAL host_record_seconds = 0;
  CCTK_REAL total_seconds = 0;
};

/** Controls the optional point-field frame written after scalar analysis. */
struct SurfaceAnalysisPointOutputOptions {
  const char *directory = nullptr;
  const char *backend = "bp5";
  int chunk_points = 65536;
  bool invariant_spin_requested = false;
  bool weyl_curvature_requested = false;
};

/** Output-only work and transfer accounting for one point-field frame. */
struct SurfaceAnalysisPointOutputPerformance {
  bool wrote_frame = false;
  bool existing_frame = false;
  int stable_id_count = 0;
  int selection_kernel_launches = 0;
  int packing_kernel_launches = 0;
  int metadata_collectives = 0;
  int explicit_device_synchronizations = 0;
  int device_to_host_copies = 0;
  std::int64_t local_point_count = 0;
  std::int64_t global_point_count = 0;
  std::int64_t device_workspace_bytes = 0;
  std::int64_t pinned_host_bytes = 0;
  CCTK_REAL selection_seconds = 0;
  CCTK_REAL packing_seconds = 0;
  CCTK_REAL device_to_host_seconds = 0;
  CCTK_REAL output_seconds = 0;
  CCTK_REAL total_seconds = 0;
};

/// Return bounded restart metadata without exposing finder implementation.
SurfaceAnalysisRuntimeStatus surface_analysis_runtime_status();

/**
 * Collect current due target layers successfully found at `iteration`.
 *
 * This is an MPI collective and must be called in the same order on every
 * rank. CarpetX may traverse several active bands at one iteration, so a
 * surface is returned immediately after the band that searched it, not again
 * during later traversals. Surface point integrals run on accelerators; the
 * returned vector is intentionally O(number of surfaces), not O(number of
 * particles).
 */
std::vector<SurfaceAnalysisRecord>
current_surface_analysis(const cGH *cctkGH, int iteration,
                         bool published_only = true,
                         SurfaceAnalysisPerformance *performance = nullptr,
                         const SurfaceAnalysisOptions &options = {});

/** Whether this build can write the native point-field dataset. */
bool surface_analysis_point_output_available();

/**
 * Write one dynamic stable-ID point frame from current logical-owner buffers.
 *
 * This is collective across the Cactus communicator. `stable_ids` must be the
 * same sorted, unique set on every rank and must refer to scalar records
 * committed by the caller for this iteration. Existing committed frames are
 * preserved, making the operation restart-idempotent.
 */
void write_current_surface_analysis_points(
    const cGH *cctkGH, int iteration, CCTK_REAL time,
    const std::vector<std::int64_t> &stable_ids,
    const SurfaceAnalysisPointOutputOptions &options,
    SurfaceAnalysisPointOutputPerformance *performance = nullptr);

/** Remove only recognized point frames newer than a recovered checkpoint. */
void reconcile_surface_analysis_point_output(
    const cGH *cctkGH, const char *directory, const char *backend,
    int recovered_iteration);

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_ANALYSIS_API_HXX
