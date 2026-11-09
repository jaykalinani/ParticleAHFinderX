/**
 * \file logical_surface_storage.hxx
 * \brief Persistent owner-rank device storage for logical angular surfaces.
 */
#ifndef PARTICLEAHFINDERX_LOGICAL_SURFACE_STORAGE_HXX
#define PARTICLEAHFINDERX_LOGICAL_SURFACE_STORAGE_HXX

#include "angular_derivatives.hxx"
#include "axial_symmetry.hxx"
#include "coordinate_fluxes.hxx"
#include "expansion.hxx"
#include "four_dimensional_analysis.hxx"
#include "relaxation.hxx"
#include "surface_particle_container.hxx"
#include "weyl_curvature.hxx"

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <array>
#include <vector>

namespace ParticleAHFinderX {

enum class SurfaceRelaxationState : int {
  idle,
  active,
  converged,
  refine,
  failed,
};

struct LogicalSurfaceSeed {
  amrex::Long stable_id = 0;
  amrex::Long first_particle_id = 0;
  amrex::Long owner_offset = 0;
  amrex::Long particle_count = 0;
  int generation = 0;
  int angular_level = 0;
  int angular_levels = 1;
  int ntheta = 0;
  int nphi = 0;
  int lifecycle = 0;
  int role = 0;
  std::array<amrex::Real, 3> center{{0, 0, 0}};
  amrex::Real radius = 0;
  amrex::Real mass_scale = 1;
  amrex::Real theta_l2_tolerance = 0;
  amrex::Real theta_linf_tolerance = 0;
};

// Some accelerator compilers reject an extended-lambda capture whose pointee
// is a private nested class type, so keep the device descriptor at namespace
// scope.
struct LogicalSurfaceDeviceDescriptor {
  amrex::Long stable_id;
  amrex::Long first_particle_id;
  amrex::Long offset;
  amrex::Long count;
  int generation;
  int angular_level;
  int angular_levels;
  int ntheta;
  int nphi;
  int lifecycle;
  int role;
  amrex::Real center[3];
  amrex::Real radius;
  amrex::Real mass_scale;
  amrex::Real theta_l2_tolerance;
  amrex::Real theta_linf_tolerance;
  int due;
  int relaxed_helper_tolerance;
  int relaxation_state;
};

struct LogicalSurfaceDeviceView {
  const LogicalSurfaceDeviceDescriptor *descriptors = nullptr;
  const int *surface_ordinal = nullptr;
  amrex::Real *position[3]{nullptr, nullptr, nullptr};
  amrex::Real *height = nullptr;
  amrex::Real *relaxation_velocity = nullptr;
  amrex::Real *expansion = nullptr;
  amrex::Real *angular = nullptr;
  amrex::Real *adm = nullptr;
  amrex::Real *normal = nullptr;
  amrex::Real *area_weight = nullptr;
  amrex::Long *particle_id = nullptr;
  amrex::Long *surface_id = nullptr;
  int *generation = nullptr;
  int *angular_level = nullptr;
  int *angular_levels = nullptr;
  int *itheta = nullptr;
  int *iphi = nullptr;
  int *lifecycle = nullptr;
  int *role = nullptr;
  int *arrival = nullptr;
  int *geometry_status = nullptr;
  int *sampled_level = nullptr;
  int *sampled_level_changed = nullptr;
  int *source_rank = nullptr;
  amrex::Long *source_index = nullptr;
  amrex::Long point_count = 0;
  int surface_count = 0;

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real &
  angular_component(const int component, const amrex::Long point) const
      noexcept {
    return angular[amrex::Long(component) * point_count + point];
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real &
  gamma(const int component, const amrex::Long point) const noexcept {
    return adm[amrex::Long(component) * point_count + point];
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real &
  d_gamma(const int d, const int component,
          const amrex::Long point) const noexcept {
    return adm[amrex::Long(6 + 6 * d + component) * point_count + point];
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real &
  curv(const int component, const amrex::Long point) const noexcept {
    return adm[amrex::Long(24 + component) * point_count + point];
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real &
  normal_component(const int d, const amrex::Long point) const noexcept {
    return normal[amrex::Long(d) * point_count + point];
  }
};

struct LogicalArrivalStats {
  amrex::Long missing = 0;
  amrex::Long duplicate = 0;
};

struct LogicalWeylDeviceView {
  amrex::Real *electric = nullptr;
  amrex::Real *magnetic = nullptr;
  int *arrival = nullptr;
  amrex::Long point_count = 0;

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real &
  electric_component(const int component,
                     const amrex::Long point) const noexcept {
    return electric[amrex::Long(component) * point_count + point];
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real &
  magnetic_component(const int component,
                     const amrex::Long point) const noexcept {
    return magnetic[amrex::Long(component) * point_count + point];
  }
};

struct LogicalGaugeDeviceView {
  amrex::Real *value = nullptr;
  int *arrival = nullptr;
  amrex::Long point_count = 0;

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real &
  component(const int component, const amrex::Long point) const noexcept {
    return value[amrex::Long(component) * point_count + point];
  }
};

struct LogicalMatterDeviceView {
  amrex::Real *value = nullptr;
  int *arrival = nullptr;
  amrex::Long point_count = 0;

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real &
  component(const int component, const amrex::Long point) const noexcept {
    return value[amrex::Long(component) * point_count + point];
  }
};

/** Read-only point fields produced by the immediately preceding analysis. */
struct LogicalAnalysisPointDeviceView {
  LogicalSurfaceDeviceView surface;
  const amrex::Real *intrinsic_metric = nullptr;
  const amrex::Real *intrinsic_normal_connection = nullptr;
  const amrex::Real *intrinsic_scalar_curvature = nullptr;
  const int *intrinsic_status = nullptr;
  const amrex::Real *axial_potential = nullptr;
  const amrex::Real *axial_vector = nullptr;
  const amrex::Real *surface_axial_values = nullptr;
  const int *surface_axial_status = nullptr;
  const amrex::Real *weyl_electric = nullptr;
  const amrex::Real *weyl_magnetic = nullptr;
  const int *weyl_arrival = nullptr;
  bool axial_workspace_available = false;
  bool weyl_workspace_available = false;
};

struct SolverConvergenceStats {
  amrex::Real maximum_ratio = 0;
  amrex::Real maximum_l2_times_mass = 0;
  amrex::Real maximum_linf_times_mass = 0;
  amrex::Real maximum_cutoff_indicator = 0;
  amrex::Long invalid = 0;
  amrex::Long active = 0;
  amrex::Long converged = 0;
  amrex::Long refine = 0;
  amrex::Long failed = 0;
};

struct LocalSolverSurfaceState {
  amrex::Long stable_id = 0;
  int angular_level = 0;
  SurfaceRelaxationState relaxation_state = SurfaceRelaxationState::idle;
  bool due = false;
  amrex::Real expansion_l2 = 0;
  amrex::Real expansion_linf = 0;
  amrex::Long invalid_points = 0;
};

struct LocalSurfaceDiagnostics {
  amrex::Long stable_id = 0;
  int angular_level = 0;
  int angular_levels = 1;
  SurfaceRelaxationState relaxation_state = SurfaceRelaxationState::idle;
  bool due = false;
  std::array<amrex::Real, 3> centroid{{0, 0, 0}};
  amrex::Real area = 0;
  amrex::Real mean_radius = 0;
  amrex::Real minimum_radius = 0;
  amrex::Real maximum_radius = 0;
  amrex::Real expansion_l2 = 0;
  amrex::Real expansion_linf = 0;
  amrex::Real mean_expansion = 0;
  std::array<amrex::Real, 3> proper_circumference{{0, 0, 0}};
  std::array<amrex::Real, 6> coordinate_quadrupole{{0, 0, 0, 0, 0, 0}};
  std::array<amrex::Real, 3> position_minimum{{0, 0, 0}};
  std::array<amrex::Real, 3> position_maximum{{0, 0, 0}};
  amrex::Long invalid_points = 0;
  int minimum_sampled_level = -1;
  int maximum_sampled_level = -1;
  amrex::Long sampled_level_changes = 0;
};

// Existing schema-3 values (20), invariant-coordinate diagnostics (3), raw
// mass and spin moments l=0,...,8 (18), physical-normal and conformal
// finite-radius coordinate fluxes (14), nine complex Weyl means (18), then
// gauge-consistent Weinberg energy, momentum, and angular momentum (7), then
// ten real Newman--Penrose Ricci projections.
// Host normalization acts only on these compact per-surface records.
constexpr int surface_analysis_coordinate_flux_offset = 41;
constexpr int surface_analysis_weyl_offset =
    surface_analysis_coordinate_flux_offset +
    CoordinateFluxLayout::component_count;
constexpr int surface_analysis_weinberg_offset =
    surface_analysis_weyl_offset + WeylIntegralLayout::component_count;
constexpr int surface_analysis_weinberg_components = 7;
constexpr int surface_analysis_ricci_offset =
    surface_analysis_weinberg_offset + surface_analysis_weinberg_components;
constexpr int surface_analysis_real_components =
    surface_analysis_ricci_offset + RicciIntegralLayout::component_count;
// Owner count, base/intrinsic invalid points, axial status/sweeps, and
// multipole/coordinate-flux/Weyl/speciality/Weinberg/Ricci invalid points.
constexpr int surface_analysis_long_components = 12;

struct SurfaceExtrapolationControl {
  amrex::Long stable_id = 0;
  int angular_level = 0;
  amrex::Real overstep = 1;
  RelaxationExtrapolationMode mode =
      RelaxationExtrapolationMode::full_shape;
};

/**
 * One rank/GPU owns the complete logical stencil for each assigned surface.
 * This allocation is independent of AMReX's spatial particle ownership and
 * survives ordinary searches and non-search iterations.
 */
class LogicalSurfaceStorage final {
public:
  void initialize(const std::vector<LogicalSurfaceSeed> &local_surfaces);
  void append_surfaces(
      const std::vector<LogicalSurfaceSeed> &new_local_surfaces);
  void update_surface_centers(
      const std::vector<SurfaceCenterUpdate> &updates);
  void update_surface_states(const std::vector<SurfaceStateUpdate> &updates);
  void reset_surface_spheres(const std::vector<SurfaceSphereReset> &resets);
  /// Reclaim retired owner storage while preserving every live device field.
  void compact(const std::vector<LogicalSurfaceSeed> &live_local_surfaces);

  bool initialized() const { return initialized_; }
  amrex::Long local_point_count() const { return local_point_count_; }
  int local_surface_count() const { return local_surface_count_; }
  amrex::Long creation_events() const { return creation_events_; }

  LogicalSurfaceDeviceView device_view();
  void set_due_surface_ids(const std::vector<amrex::Long> &stable_ids);
  void set_due_surface_layers(
      const std::vector<SurfaceLayerSelection> &selections);
  /// Select established tracking surfaces whose non-target layers may use
  /// the internal helper tolerance. Acquisition and reacquisition remain
  /// strict so a poor coarse solve cannot define the nonlinear basin.
  void set_relaxed_helper_surface_ids(
      const std::vector<amrex::Long> &stable_ids);
  /// Restrict each selected target shape into all persistent coarse layers.
  void restrict_target_to_coarse_layers(
      const std::vector<amrex::Long> &stable_ids);
  /// Prolong converged selected layers into their existing next-finer layer.
  void prolong_surface_layers(
      const std::vector<SurfaceLayerProlongation> &source_layers);
  void begin_sample_routing();
  LogicalArrivalStats local_arrival_stats() const;
  /// Allocate and clear the optional logical-owner Weyl receive workspace.
  LogicalWeylDeviceView begin_weyl_routing();
  LogicalArrivalStats local_weyl_arrival_stats() const;
  /// Allocate and clear the optional logical-owner gauge receive workspace.
  LogicalGaugeDeviceView begin_gauge_routing();
  LogicalArrivalStats local_gauge_arrival_stats() const;
  /// Allocate and clear the optional logical-owner matter receive workspace.
  LogicalMatterDeviceView begin_matter_routing();
  LogicalArrivalStats local_matter_arrival_stats() const;
  void evaluate_surface_geometry();
  /// Refresh compact owner-device residual arrays without changing state.
  void evaluate_solver_residuals();
  SolverConvergenceStats
  update_solver_convergence(bool accept_convergence,
                            amrex::Real cutoff_indicator_threshold,
                            amrex::Real helper_tolerance_factor,
                            bool fail_invalid_geometry = true);
  /// Update owner-local relaxation states without a host or MPI reduction.
  void update_solver_convergence_on_device(
      bool accept_convergence, amrex::Real cutoff_indicator_threshold,
      amrex::Real helper_tolerance_factor,
      bool fail_invalid_geometry = true);
  /// Reopen cached converged/refine decisions for one final direct gather.
  void reactivate_due_layers_for_direct_validation();
  /// Copy one compact residual/state record per owner-local surface.
  std::vector<LocalSolverSurfaceState> local_solver_surface_states();
  std::vector<LocalSurfaceDiagnostics> local_surface_diagnostics();
  /**
   * Form owner-local quasi-local integrals and pack them into global stable-ID
   * order without leaving the accelerator.
   *
   * `global_stable_ids` and both result buffers are device pointers. Real
   * records contain area, covariant momentum, curvature flux, coordinate
   * angular momentum, integrated intrinsic scalar curvature, integrated
   * normal-connection norm, normalized axial-mode diagnostics, invariant-
   * coordinate diagnostics, raw mass/spin multipoles, coordinate fluxes, and
   * complex Weyl means, and Weinberg coordinate fluxes. Long records contain
   * owner count, base/intrinsic
   * invalid counts, axial status, eigensolver sweeps, and feature-level
   * invalid counts. The final long is an additional mapping error counter.
   * The caller performs the portable AMReX MPI reduction.
   */
  void prepare_surface_analysis(const amrex::Long *global_stable_ids,
                                int global_surface_count,
                                amrex::Real *global_reals,
                                amrex::Long *global_longs,
                                const AxialSymmetryControl &axial_control,
                                bool compute_coordinate_fluxes,
                                bool compute_weyl_curvature,
                                bool compute_weinberg_fluxes,
                                bool compute_ricci_scalars,
                                bool use_matter_ricci);
  /**
   * Return the transient point fields made by prepare_surface_analysis().
   *
   * The view aliases logical-owner accelerator storage and remains valid only
   * until the next finder or analysis operation mutates that storage. It is
   * intended solely for immediate, bounded output packing.
   */
  LogicalAnalysisPointDeviceView
  analysis_point_device_view(bool invariant_spin_requested,
                             bool weyl_curvature_requested);
  /// Refresh device-only intrinsic workspaces for due target layers.
  ///
  /// This is public because CUDA extended device lambdas cannot be enclosed
  /// by a private member function. LogicalSurfaceStorage remains an internal
  /// implementation class; callers should use prepare_surface_analysis().
  void prepare_intrinsic_geometry();
  /// Solve and normalize the variational axial mode on due target layers.
  void prepare_axial_symmetry(const AxialSymmetryControl &control);
  amrex::Real local_minkowski_expansion_error() const;
  void begin_relaxation_search(amrex::Real eta_times_mass);
  void finish_relaxation_search();
  /// Restore every due layer and leave it idle for a later synchronized band.
  void defer_relaxation_search();
  void begin_relaxation_step();
  /// Store selected active shapes as the trajectory reference on the GPU.
  void store_extrapolation_reference(
      const std::vector<SurfaceLayerSelection> &selections);
  /// Preserve selected current h/v fields while accepted trials are tested.
  void begin_extrapolation_trials(
      const std::vector<SurfaceLayerSelection> &selections);
  /// Reconstruct selected trials from the preserved base and reference.
  void set_extrapolation_trial(
      const std::vector<SurfaceExtrapolationControl> &controls,
      amrex::Real eta_times_mass, amrex::Real minimum_radius_factor,
      amrex::Real maximum_radius_factor, bool reset_accepted_velocity);
  /// Compute one stable pseudo-time step per active logical surface.
  void prepare_relaxation_time_steps(amrex::Real cfl_factor);
  /// Return the owner-local minimum active-surface step for direct sampling.
  amrex::Real local_minimum_relaxation_time_step() const;
  /// Advance the search-local pseudo-time of every accepted active surface.
  void accumulate_relaxation_pseudo_time();
  /// Return the owner-local minimum pseudo-time over active surfaces.
  amrex::Real local_minimum_relaxation_pseudo_time() const;
  void advance_relaxation_stage(RelaxationIntegrator integrator, int stage,
                                amrex::Real eta_times_mass,
                                amrex::Real dissipation_strength,
                                amrex::Real minimum_radius_factor,
                                amrex::Real maximum_radius_factor);
  amrex::Long local_relaxation_invalid() const;
  /// Mark and restore invalid surfaces without a host synchronization.
  void reject_invalid_relaxation_steps_on_device();
  amrex::Long reject_invalid_relaxation_steps();
  amrex::Real local_minimum_angular_spacing();
  void rebuild_positions();

private:
  amrex::Gpu::DeviceVector<LogicalSurfaceDeviceDescriptor> descriptors_;
  std::array<amrex::Gpu::DeviceVector<amrex::Real>, 3> position_;
  amrex::Gpu::DeviceVector<amrex::Real> height_;
  amrex::Gpu::DeviceVector<amrex::Real> relaxation_velocity_;
  amrex::Gpu::DeviceVector<amrex::Real> expansion_;
  // Component-major arrays: (h_theta,h_phi,h_tt,h_tp,h_pp), then the
  // 30-value ADM record, Cartesian unit normal, and angular area weight.
  amrex::Gpu::DeviceVector<amrex::Real> angular_derivatives_;
  amrex::Gpu::DeviceVector<amrex::Real> adm_;
  amrex::Gpu::DeviceVector<amrex::Real> normal_;
  amrex::Gpu::DeviceVector<amrex::Real> area_weight_;
  amrex::Gpu::DeviceVector<amrex::Long> particle_id_;
  amrex::Gpu::DeviceVector<amrex::Long> surface_id_;
  amrex::Gpu::DeviceVector<int> generation_;
  amrex::Gpu::DeviceVector<int> angular_level_;
  amrex::Gpu::DeviceVector<int> angular_levels_;
  amrex::Gpu::DeviceVector<int> itheta_;
  amrex::Gpu::DeviceVector<int> iphi_;
  amrex::Gpu::DeviceVector<int> lifecycle_;
  amrex::Gpu::DeviceVector<int> role_;
  amrex::Gpu::DeviceVector<int> surface_ordinal_;
  // Zero means missing, one means exactly one routed record, and values above
  // one reveal duplicate logical indices without a host population gather.
  amrex::Gpu::DeviceVector<int> arrival_;
  amrex::Gpu::DeviceVector<int> geometry_status_;
  amrex::Gpu::DeviceVector<int> sampled_level_;
  amrex::Gpu::DeviceVector<int> sampled_level_changed_;
  amrex::Gpu::DeviceVector<int> relaxation_status_;
  // Component-major base (h,v) and up to four pairs of stage right-hand sides.
  amrex::Gpu::DeviceVector<amrex::Real> relaxation_base_;
  amrex::Gpu::DeviceVector<amrex::Real> relaxation_rhs_;
  amrex::Gpu::DeviceVector<amrex::Real> committed_state_;
  amrex::Gpu::DeviceVector<amrex::Real> extrapolation_reference_;
  // sin(theta)-weighted l<=1 displacement of h between the stored trajectory
  // reference and pre-trial state. Component-major order is 1,x,y,z.
  amrex::Gpu::DeviceVector<amrex::Real> surface_extrapolation_displacement_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_extrapolation_weight_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_area_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_theta_squared_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_theta_integral_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_theta_linf_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_cutoff_indicator_;
  amrex::Gpu::DeviceVector<amrex::Long> surface_invalid_;
  amrex::Gpu::DeviceVector<int> surface_minimum_sampled_level_;
  amrex::Gpu::DeviceVector<int> surface_maximum_sampled_level_;
  amrex::Gpu::DeviceVector<amrex::Long> surface_sampled_level_changes_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_position_integral_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_position_second_integral_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_position_minimum_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_position_maximum_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_height_integral_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_height_minimum_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_height_maximum_;
  // Proper great-circle lengths in the xy, xz, and yz coordinate planes.
  amrex::Gpu::DeviceVector<amrex::Real> surface_circumference_;
  // Component-major surface analysis records: area, covariant quasi-local
  // momentum, curvature flux, coordinate angular momentum, integrated
  // intrinsic scalar curvature, and normal-connection squared norm.
  amrex::Gpu::DeviceVector<amrex::Real> surface_analysis_integral_;
  amrex::Gpu::DeviceVector<amrex::Long> surface_analysis_invalid_;
  amrex::Gpu::DeviceVector<amrex::Long> surface_intrinsic_invalid_;
  amrex::Gpu::DeviceVector<amrex::Long> surface_multipole_invalid_;
  amrex::Gpu::DeviceVector<amrex::Long> surface_coordinate_flux_invalid_;
  amrex::Gpu::DeviceVector<amrex::Long> surface_weyl_invalid_;
  amrex::Gpu::DeviceVector<amrex::Long> surface_weinberg_invalid_;
  amrex::Gpu::DeviceVector<amrex::Long> surface_ricci_invalid_;
  amrex::Gpu::DeviceVector<amrex::Long>
      surface_speciality_index_invalid_;
  amrex::Gpu::DeviceVector<amrex::Long>
      surface_speciality_difference_invalid_;
  // Optional transient pointwise E/B data routed only at analysis cadence.
  amrex::Gpu::DeviceVector<amrex::Real> weyl_electric_;
  amrex::Gpu::DeviceVector<amrex::Real> weyl_magnetic_;
  amrex::Gpu::DeviceVector<int> weyl_arrival_;
  // Optional transient lapse/shift data routed only at analysis cadence.
  amrex::Gpu::DeviceVector<amrex::Real> gauge_value_;
  amrex::Gpu::DeviceVector<int> gauge_arrival_;
  // Optional transient covariant stress-energy routed only in tmunu mode.
  amrex::Gpu::DeviceVector<amrex::Real> matter_value_;
  amrex::Gpu::DeviceVector<int> matter_arrival_;
  // Reusable pointwise workspaces for the reference-connection intrinsic
  // geometry operator. These never leave the logical owner GPU.
  amrex::Gpu::DeviceVector<amrex::Real> intrinsic_metric_;
  amrex::Gpu::DeviceVector<amrex::Real> intrinsic_normal_connection_;
  amrex::Gpu::DeviceVector<amrex::Real>
      intrinsic_connection_difference_;
  amrex::Gpu::DeviceVector<amrex::Real> intrinsic_scalar_curvature_;
  amrex::Gpu::DeviceVector<int> intrinsic_point_status_;
  amrex::Gpu::DeviceVector<int> intrinsic_connection_status_;
  // Reusable owner-GPU workspaces for the variational axial-symmetry solve.
  // Point fields are mode-major; projected matrices are descriptor-major.
  amrex::Gpu::DeviceVector<amrex::Real> axial_basis_;
  amrex::Gpu::DeviceVector<amrex::Real> axial_gradient_;
  amrex::Gpu::DeviceVector<amrex::Real> axial_covector_;
  amrex::Gpu::DeviceVector<amrex::Real> axial_deformation_;
  amrex::Gpu::DeviceVector<amrex::Real> axial_stiffness_;
  amrex::Gpu::DeviceVector<amrex::Real> axial_mass_;
  amrex::Gpu::DeviceVector<amrex::Real> axial_factor_;
  amrex::Gpu::DeviceVector<amrex::Real> axial_work_;
  amrex::Gpu::DeviceVector<amrex::Real> axial_eigenvectors_;
  amrex::Gpu::DeviceVector<amrex::Real> axial_coefficients_;
  amrex::Gpu::DeviceVector<amrex::Real> axial_potential_;
  amrex::Gpu::DeviceVector<amrex::Real> axial_vector_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_axial_values_;
  amrex::Gpu::DeviceVector<int> surface_axial_status_;
  amrex::Gpu::DeviceVector<int> surface_axial_iterations_;
  amrex::Gpu::DeviceVector<LocalSolverSurfaceState> solver_surface_state_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_time_step_;
  amrex::Gpu::DeviceVector<amrex::Real> surface_pseudo_time_;
  amrex::Gpu::DeviceVector<int> source_rank_;
  amrex::Gpu::DeviceVector<amrex::Long> source_index_;

  amrex::Long local_point_count_ = 0;
  int local_surface_count_ = 0;
  amrex::Long creation_events_ = 0;
  bool initialized_ = false;
};

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_LOGICAL_SURFACE_STORAGE_HXX
