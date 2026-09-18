/**
 * \file runtime.hxx
 * \brief Process-local persistent surface and configuration state.
 */
#ifndef PARTICLEAHFINDERX_RUNTIME_HXX
#define PARTICLEAHFINDERX_RUNTIME_HXX

#include "carpetx_adapter.hxx"
#include "logical_surface_storage.hxx"
#include "surface_particle_container.hxx"

#include <AMReX_REAL.H>

#include <array>
#include <memory>
#include <vector>

namespace ParticleAHFinderX {

constexpr int max_configured_surfaces = 256;
constexpr int max_compatibility_slots = 256;
constexpr int max_angular_layers = 8;

enum class CenterPolicy : int {
  manual,
  external,
  centroid,
  external_then_centroid,
  centroid_then_external,
  blend_external_centroid,
  parent_midpoint,
  puncture_tracker_displacement,
};

struct ResidualTolerances {
  double l2_times_mass = 0.0;
  double linf_times_mass = 0.0;
};

struct CenterProviderRef {
  std::array<int, 3> variable_indices{{-1, -1, -1}};
  std::array<int, 3> array_indices{{0, 0, 0}};
};

struct CenterHistory {
  std::array<amrex::Real, 3> times{{0, 0, 0}};
  std::array<std::array<amrex::Real, 3>, 3> centers{{
      {{0, 0, 0}},
      {{0, 0, 0}},
      {{0, 0, 0}},
  }};
  std::array<amrex::Real, 3> mean_radii{{0, 0, 0}};
  std::array<std::array<amrex::Real, 3>, 3> external_centers{{
      {{0, 0, 0}},
      {{0, 0, 0}},
      {{0, 0, 0}},
  }};
  std::array<int, 3> external_valid{{0, 0, 0}};
  int count = 0;
};

struct AngularLayerRecord {
  amrex::Long first_particle_id = 0;
  amrex::Long particle_count = 0;
  amrex::Long logical_owner_offset = 0;
  int ntheta = 0;
  int nphi = 0;
  SurfaceRelaxationState relaxation_state = SurfaceRelaxationState::idle;
  int last_relaxation_steps = 0;
  amrex::Real expansion_l2 = 0;
  amrex::Real expansion_linf = 0;
};

struct SurfaceRecord {
  amrex::Long stable_id = 0;
  amrex::Long first_particle_id = 0;
  amrex::Long particle_count = 0;
  int configured_slot = -1;
  int generation = 0;
  int logical_owner_rank = -1;
  amrex::Long logical_owner_offset = 0;
  std::array<amrex::Long, 2> parent_ids{{0, 0}};
  int compatibility_slot = -1;
  int search_every = 0;
  int next_search_iteration = 0;
  int consecutive_failures = 0;
  int consecutive_domain_exit_checks = 0;
  int last_domain_exit_check_iteration = -1;
  bool retired_outside_domain = false;
  int birth_iteration = -1;
  int last_state_change_iteration = -1;
  int candidate_attempts = 0;
  int promoted_iteration = -1;
  int retire_after_iteration = -1;
  int ntheta = 0;
  int nphi = 0;
  int angular_levels = 1;
  int active_angular_level = 0;
  std::array<AngularLayerRecord, max_angular_layers> angular_layer{};
  CenterPolicy center_policy = CenterPolicy::manual;
  bool predict_center = true;
  amrex::Real external_center_weight = 0.5;
  amrex::Real birth_time = 0;
  amrex::Real candidate_radius_floor = 0;
  amrex::Real candidate_radius_ceiling = 0;
  amrex::Real candidate_trigger_distance = 0;
  amrex::Real candidate_expiry_distance = 0;
  bool published = false;
  std::array<amrex::Real, 3> manual_center{{0, 0, 0}};
  std::array<amrex::Real, 3> center{{0, 0, 0}};
  amrex::Real radius = 0;
  amrex::Real mass_scale = 1;
  ResidualTolerances tolerances;
  ParticleLifecycle lifecycle = ParticleLifecycle::candidate;
  SurfaceRole role = SurfaceRole::individual;
  SurfaceRelaxationState relaxation_state = SurfaceRelaxationState::idle;
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
  amrex::Long cumulative_sampled_level_changes = 0;
  int last_search_iteration = -1;
  int last_success_iteration = -1;
  int last_history_iteration = -1;
  amrex::Real last_search_time = 0;
  int last_search_steps = 0;
  CenterHistory center_history;
};

struct Runtime {
  // Containers are attached once and survive every ordinary search and
  // non-search iteration. Only shutdown destroys them.
  std::vector<std::unique_ptr<SurfaceParticleContainer>> containers;
  std::unique_ptr<LogicalSurfaceStorage> logical_surfaces;
  std::vector<SurfaceRecord> surfaces;
  std::array<MeshFieldRef, 12> adm_fields{};
  std::array<CenterProviderRef, max_configured_surfaces> center_providers{};
  std::array<ResidualTolerances, max_configured_surfaces> tolerances{};
  amrex::Long next_surface_id = 1;
  amrex::Long next_particle_id = 1;
  amrex::Long global_particle_count = 0;
  amrex::Long population_creation_events = 0;
  int num_patches = 0;
  int last_particle_output_iteration = -1;
  int last_diagnostics_iteration = -1;
  int last_spherical_export_iteration = -1;
  int last_checkpoint_iteration = -1;
  int last_checkpoint_runtime = -1;
  int recovered_iteration = -1;
  int last_adm_gather_iteration = -1;
  int last_logical_route_iteration = -1;
  int last_center_prediction_iteration = -1;
  int last_relaxation_steps = 0;
  amrex::Real last_expansion_l2 = 0;
  amrex::Real last_expansion_linf = 0;
  bool setup = false;
  bool initial_surfaces_created = false;
  bool reported_adm_centering = false;
  bool adm_gather_self_test_complete = false;
  bool expansion_self_test_complete = false;
  bool last_search_converged = false;
  bool reported_solver_pending = false;
  bool solver_in_flight = false;
  bool checkpoint_in_progress = false;
  bool recovered = false;
  amrex::Long successful_searches = 0;
  amrex::Long failed_searches = 0;
  amrex::Long regrid_events = 0;
  int last_regrid_iteration = -1;
};

Runtime &runtime();

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_RUNTIME_HXX
