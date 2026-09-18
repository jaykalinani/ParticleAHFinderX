/**
 * \file logical_surface_storage.hxx
 * \brief Persistent owner-rank device storage for logical angular surfaces.
 */
#ifndef PARTICLEAHFINDERX_LOGICAL_SURFACE_STORAGE_HXX
#define PARTICLEAHFINDERX_LOGICAL_SURFACE_STORAGE_HXX

#include "angular_derivatives.hxx"
#include "expansion.hxx"
#include "relaxation.hxx"
#include "surface_particle_container.hxx"

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

struct ExpansionStats {
  amrex::Real area = 0;
  amrex::Real theta_squared_integral = 0;
  amrex::Real theta_linf = 0;
  amrex::Long invalid = 0;
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
  /// Prolong converged selected layers into their existing next-finer layer.
  void prolong_surface_layers(
      const std::vector<SurfaceLayerSelection> &source_layers);
  void begin_sample_routing();
  LogicalArrivalStats local_arrival_stats() const;
  void evaluate_angular_derivatives();
  void evaluate_expansion();
  ExpansionStats local_expansion_stats() const;
  SolverConvergenceStats
  update_solver_convergence(bool accept_convergence,
                            amrex::Real cutoff_indicator_threshold);
  std::vector<LocalSurfaceDiagnostics> local_surface_diagnostics();
  amrex::Real local_minkowski_expansion_error() const;
  void begin_relaxation_search(amrex::Real eta_times_mass);
  void finish_relaxation_search();
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
  amrex::Real prepare_relaxation_time_steps(amrex::Real cfl_factor);
  void advance_relaxation_stage(RelaxationIntegrator integrator, int stage,
                                amrex::Real eta_times_mass,
                                amrex::Real dissipation_strength,
                                amrex::Real minimum_radius_factor,
                                amrex::Real maximum_radius_factor);
  amrex::Long local_relaxation_invalid() const;
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
  // sin(theta)-weighted constant-mode displacement of h between the stored
  // trajectory reference and the pre-trial state, one value per surface.
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
  amrex::Gpu::DeviceVector<amrex::Real> surface_time_step_;
  amrex::Gpu::DeviceVector<int> source_rank_;
  amrex::Gpu::DeviceVector<amrex::Long> source_index_;

  amrex::Long local_point_count_ = 0;
  int local_surface_count_ = 0;
  amrex::Long creation_events_ = 0;
  bool initialized_ = false;
};

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_LOGICAL_SURFACE_STORAGE_HXX
