/**
 * \file checkpoint.cxx
 * \brief Quiescent AMReX particle checkpoint and decomposition-safe recovery.
 */
#include "runtime.hxx"
#include "surface_manager.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_Config.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Vector.H>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ParticleAHFinderX {
namespace {

constexpr int checkpoint_schema = 12;
constexpr const char *particle_dataset_name = "surface_particles";
constexpr const char *metadata_name = "ParticleAHFinderX.meta";

std::string iteration_string(const int iteration) {
  std::ostringstream stream;
  stream << std::setw(8) << std::setfill('0') << iteration;
  return stream.str();
}

std::string checkpoint_root(const std::string &directory,
                            const std::string &file,
                            const int iteration) {
  return directory + "/" + file + ".it" + iteration_string(iteration) +
         ".particleahfinderx";
}

std::string patch_name(const std::size_t patch) {
  std::ostringstream stream;
  stream << "patch" << std::setw(2) << std::setfill('0') << patch;
  return stream.str();
}

void hash_bytes(std::uint64_t &hash, const void *const data,
                const std::size_t size) {
  const auto *const bytes = static_cast<const unsigned char *>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= UINT64_C(1099511628211);
  }
}

template <typename T> void hash_value(std::uint64_t &hash, const T &value) {
  static_assert(std::is_trivially_copyable<T>::value,
                "Checkpoint hash values must be trivially copyable");
  hash_bytes(hash, &value, sizeof(value));
}

void hash_string(std::uint64_t &hash, const char *const value) {
  const std::string string(value ? value : "");
  hash_bytes(hash, string.data(), string.size());
  const unsigned char terminator = 0;
  hash_bytes(hash, &terminator, 1);
}

std::uint64_t configuration_hash() {
  DECLARE_CCTK_PARAMETERS;
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const int dimension = AMREX_SPACEDIM;
  const int real_size = sizeof(amrex::Real);
  hash_value(hash, dimension);
  hash_value(hash, real_size);
  hash_value(hash, num_surfaces);
  hash_value(hash, interpolation_order);
  hash_value(hash, enable_angular_continuation);
  hash_value(hash, angular_continuation_max_levels);
  hash_value(hash, angular_continuation_min_ntheta);
  hash_value(hash, candidate_angular_continuation_min_ntheta);
  hash_value(hash, angular_continuation_cutoff_threshold);
  hash_value(hash, maximum_relaxation_steps);
  hash_value(hash, minimum_relaxation_steps);
  hash_value(hash, relaxation_cfl_factor);
  hash_value(hash, eta_damping_times_mass);
  hash_value(hash, angular_dissipation_strength);
  hash_value(hash, minimum_radius_factor);
  hash_value(hash, maximum_radius_factor);
  hash_value(hash, maximum_center_extrapolation_intervals);
  hash_value(hash, retire_on_domain_exit);
  hash_value(hash, domain_exit_grace_searches);
  hash_value(hash, find_every);
  hash_value(hash, export_spherical_surfaces);
  for (int slot = 0; slot < max_compatibility_slots; ++slot) {
    hash_value(hash, spherical_surface_index[slot]);
    hash_string(hash, spherical_surface_name[slot]);
  }
  hash_value(hash, enable_dynamic_common_horizons);
  hash_value(hash, candidate_find_every);
  hash_value(hash, candidate_ntheta);
  hash_value(hash, candidate_nphi);
  hash_value(hash, maximum_active_candidates);
  hash_value(hash, maximum_new_candidates_per_search);
  hash_value(hash, candidate_trigger_factor);
  hash_value(hash, candidate_expiry_factor);
  hash_value(hash, candidate_initial_radius_factor);
  hash_value(hash, candidate_radius_growth_factor);
  hash_value(hash, candidate_maximum_radius_factor);
  hash_value(hash, maximum_candidate_attempts);
  hash_value(hash, candidate_retry_base_iterations);
  hash_value(hash, candidate_retry_max_iterations);
  hash_value(hash, candidate_containment_tolerance);
  hash_value(hash, candidate_deduplication_center_tolerance);
  hash_value(hash, candidate_deduplication_area_tolerance);
  hash_value(hash, retire_parents_after_common);
  hash_value(hash, common_parent_grace_iterations);
  hash_value(hash, candidate_theta_l2_tolerance);
  hash_value(hash, candidate_theta_linf_tolerance);
  hash_string(hash, angular_topology);
  hash_string(hash, logical_ownership);
  hash_string(hash, relaxation_integrator);
  for (int surface = 0; surface < num_surfaces; ++surface) {
    hash_value(hash, surface_enabled[surface]);
    hash_string(hash, initial_role[surface]);
    hash_value(hash, find_every_surface[surface]);
    hash_value(hash, ntheta[surface]);
    hash_value(hash, nphi[surface]);
    hash_value(hash, initial_center_x[surface]);
    hash_value(hash, initial_center_y[surface]);
    hash_value(hash, initial_center_z[surface]);
    hash_value(hash, initial_radius[surface]);
    hash_value(hash, mass_scale[surface]);
    hash_value(hash, external_center_weight[surface]);
    hash_value(hash, predict_centers[surface]);
    hash_value(hash, custom_theta_l2_tolerance[surface]);
    hash_value(hash, custom_theta_linf_tolerance[surface]);
    hash_string(hash, center_policy[surface]);
    hash_string(hash, external_center_x[surface]);
    hash_string(hash, external_center_y[surface]);
    hash_string(hash, external_center_z[surface]);
    hash_string(hash, residual_profile[surface]);
  }
  return hash;
}

void validate_quiescent_state(const Runtime &state) {
  if (!state.setup || !state.initial_surfaces_created ||
      !state.logical_surfaces || !state.logical_surfaces->initialized())
    CCTK_ERROR("ParticleAHFinderX checkpoint requested before initialization");
  if (state.solver_in_flight || state.checkpoint_in_progress)
    CCTK_ERROR("ParticleAHFinderX checkpoint requested while surface work is "
               "in flight");
}

void write_metadata(const std::string &root, const cGH *const cctkGH,
                    const Runtime &state) {
  const std::string temporary = root + "/" + metadata_name + ".tmp";
  const std::string final = root + "/" + metadata_name;
  std::ofstream metadata(temporary, std::ios::out | std::ios::trunc);
  if (!metadata)
    CCTK_VERROR("Could not create ParticleAHFinderX checkpoint manifest '%s'",
                temporary.c_str());

  metadata << std::setprecision(std::numeric_limits<double>::max_digits10);
  metadata << "schema " << checkpoint_schema << '\n';
  metadata << "iteration " << cctkGH->cctk_iteration << '\n';
  metadata << "time " << cctkGH->cctk_time << '\n';
  metadata << "patches " << state.containers.size() << '\n';
  metadata << "dimension " << AMREX_SPACEDIM << '\n';
  metadata << "real_size " << sizeof(amrex::Real) << '\n';
  metadata << "topology latitude_longitude_cell_centered\n";
  metadata << "configuration_hash " << configuration_hash() << '\n';
  metadata << "next_surface_id " << state.next_surface_id << '\n';
  metadata << "next_particle_id " << state.next_particle_id << '\n';
  metadata << "global_particle_count " << state.global_particle_count << '\n';
  metadata << "population_creation_events "
           << state.population_creation_events << '\n';
  metadata << "last_particle_output_iteration "
           << state.last_particle_output_iteration << '\n';
  metadata << "last_diagnostics_iteration "
           << state.last_diagnostics_iteration << '\n';
  metadata << "last_spherical_export_iteration "
           << state.last_spherical_export_iteration << '\n';
  metadata << "successful_searches " << state.successful_searches << '\n';
  metadata << "failed_searches " << state.failed_searches << '\n';
  metadata << "regrid_events " << state.regrid_events << '\n';
  metadata << "last_regrid_iteration " << state.last_regrid_iteration
           << '\n';
  metadata << "surfaces " << state.surfaces.size() << '\n';
  for (const auto &surface : state.surfaces) {
    metadata << "surface " << surface.stable_id << ' '
             << surface.first_particle_id << ' ' << surface.particle_count
             << ' ' << surface.configured_slot << ' ' << surface.generation
             << ' ' << surface.ntheta << ' ' << surface.nphi << ' '
             << surface.angular_levels << ' '
             << surface.active_angular_level;
    for (int level = 0; level < surface.angular_levels; ++level) {
      const auto &layer = surface.angular_layer[level];
      metadata << ' ' << layer.first_particle_id << ' '
               << layer.particle_count << ' ' << layer.logical_owner_offset
               << ' ' << layer.ntheta << ' ' << layer.nphi << ' '
               << static_cast<int>(layer.relaxation_state) << ' '
               << layer.last_relaxation_steps << ' ' << layer.expansion_l2
               << ' ' << layer.expansion_linf;
    }
    metadata << ' '
             << surface.center[0] << ' ' << surface.center[1] << ' '
             << surface.center[2] << ' ' << surface.radius << ' '
             << surface.mass_scale << ' '
             << surface.tolerances.l2_times_mass << ' '
             << surface.tolerances.linf_times_mass << ' '
             << static_cast<int>(surface.lifecycle) << ' '
             << static_cast<int>(surface.role) << ' '
             << surface.parent_ids[0] << ' ' << surface.parent_ids[1] << ' '
             << surface.compatibility_slot << ' '
             << static_cast<int>(surface.relaxation_state) << ' '
             << surface.centroid[0] << ' ' << surface.centroid[1] << ' '
             << surface.centroid[2] << ' ' << surface.area << ' '
             << surface.mean_radius << ' ' << surface.minimum_radius << ' '
             << surface.maximum_radius << ' ' << surface.expansion_l2 << ' '
             << surface.expansion_linf << ' ' << surface.mean_expansion;
    for (const auto value : surface.proper_circumference)
      metadata << ' ' << value;
    for (const auto value : surface.coordinate_quadrupole)
      metadata << ' ' << value;
    for (const auto value : surface.position_minimum)
      metadata << ' ' << value;
    for (const auto value : surface.position_maximum)
      metadata << ' ' << value;
    metadata << ' ' << surface.invalid_points << ' '
             << surface.minimum_sampled_level << ' '
             << surface.maximum_sampled_level << ' '
             << surface.sampled_level_changes << ' '
             << surface.cumulative_sampled_level_changes << ' '
             << surface.last_search_iteration << ' '
             << surface.last_success_iteration << ' '
             << surface.last_search_time << ' ' << surface.last_search_steps
             << ' ' << surface.search_every << ' '
             << surface.next_search_iteration << ' '
             << surface.consecutive_failures << ' '
             << static_cast<int>(surface.center_policy) << ' '
             << surface.predict_center << ' '
             << surface.external_center_weight;
    for (const auto value : surface.manual_center)
      metadata << ' ' << value;
    metadata << ' ' << surface.last_history_iteration << ' '
             << surface.center_history.count;
    for (int sample = 0; sample < 3; ++sample) {
      metadata << ' ' << surface.center_history.times[sample];
      for (const auto value : surface.center_history.centers[sample])
        metadata << ' ' << value;
      metadata << ' ' << surface.center_history.mean_radii[sample] << ' '
               << surface.center_history.external_valid[sample];
      for (const auto value :
           surface.center_history.external_centers[sample])
        metadata << ' ' << value;
    }
    metadata << ' ' << surface.birth_iteration << ' '
             << surface.last_state_change_iteration << ' '
             << surface.candidate_attempts << ' '
             << surface.promoted_iteration << ' '
             << surface.retire_after_iteration << ' ' << surface.birth_time
             << ' ' << surface.candidate_radius_floor << ' '
             << surface.candidate_radius_ceiling << ' '
             << surface.candidate_trigger_distance << ' '
             << surface.candidate_expiry_distance << ' '
             << surface.published << ' '
             << surface.consecutive_domain_exit_checks << ' '
             << surface.last_domain_exit_check_iteration << ' '
             << surface.retired_outside_domain;
    metadata << '\n';
  }
  metadata << "complete 1\n";
  metadata.close();
  if (!metadata)
    CCTK_VERROR("Could not complete ParticleAHFinderX checkpoint manifest '%s'",
                temporary.c_str());
  if (std::rename(temporary.c_str(), final.c_str()) != 0)
    CCTK_VERROR("Could not publish ParticleAHFinderX checkpoint manifest '%s'",
                final.c_str());
}

void write_checkpoint(const cGH *const cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  validate_quiescent_state(state);
  if (cctkGH->cctk_iteration == state.recovered_iteration ||
      cctkGH->cctk_iteration <= state.last_checkpoint_iteration)
    return;

  state.checkpoint_in_progress = true;
  amrex::Gpu::streamSynchronize();
  const std::string root = checkpoint_root(
      checkpoint_dir, checkpoint_file, cctkGH->cctk_iteration);
  if (CCTK_CreateDirectory(0755, root.c_str()) < 0)
    CCTK_VERROR("Could not create ParticleAHFinderX checkpoint directory '%s'",
                root.c_str());

  const amrex::Vector<std::string> real_names;
  const amrex::Vector<std::string> int_names;
  for (std::size_t patch = 0; patch < state.containers.size(); ++patch)
    state.containers[patch]->Checkpoint(root + "/" + patch_name(patch),
                                        particle_dataset_name, real_names,
                                        int_names);
  amrex::ParallelDescriptor::Barrier();
  if (amrex::ParallelDescriptor::IOProcessor())
    write_metadata(root, cctkGH, state);
  amrex::ParallelDescriptor::Barrier();
  state.last_checkpoint_iteration = cctkGH->cctk_iteration;
  state.checkpoint_in_progress = false;
}

struct RecoveredState {
  int schema = -1;
  int iteration = -1;
  double time = 0;
  std::size_t patches = 0;
  int dimension = -1;
  int real_size = -1;
  std::string topology;
  std::uint64_t configuration = 0;
  amrex::Long next_surface_id = 0;
  amrex::Long next_particle_id = 0;
  amrex::Long global_particle_count = -1;
  amrex::Long population_creation_events = -1;
  int last_particle_output_iteration = -1;
  int last_diagnostics_iteration = -1;
  int last_spherical_export_iteration = -1;
  amrex::Long successful_searches = 0;
  amrex::Long failed_searches = 0;
  amrex::Long regrid_events = 0;
  int last_regrid_iteration = -1;
  std::size_t declared_surfaces = 0;
  std::vector<SurfaceRecord> surfaces;
  int complete = 0;
};

RecoveredState read_metadata(const std::string &filename) {
  std::ifstream metadata(filename);
  if (!metadata)
    CCTK_VERROR("ParticleAHFinderX recovery manifest '%s' is missing",
                filename.c_str());
  RecoveredState recovered;
  std::string key;
  while (metadata >> key) {
    if (key == "schema")
      metadata >> recovered.schema;
    else if (key == "iteration")
      metadata >> recovered.iteration;
    else if (key == "time")
      metadata >> recovered.time;
    else if (key == "patches")
      metadata >> recovered.patches;
    else if (key == "dimension")
      metadata >> recovered.dimension;
    else if (key == "real_size")
      metadata >> recovered.real_size;
    else if (key == "topology")
      metadata >> recovered.topology;
    else if (key == "configuration_hash")
      metadata >> recovered.configuration;
    else if (key == "next_surface_id")
      metadata >> recovered.next_surface_id;
    else if (key == "next_particle_id")
      metadata >> recovered.next_particle_id;
    else if (key == "global_particle_count")
      metadata >> recovered.global_particle_count;
    else if (key == "population_creation_events")
      metadata >> recovered.population_creation_events;
    else if (key == "last_particle_output_iteration")
      metadata >> recovered.last_particle_output_iteration;
    else if (key == "last_diagnostics_iteration")
      metadata >> recovered.last_diagnostics_iteration;
    else if (key == "last_spherical_export_iteration")
      metadata >> recovered.last_spherical_export_iteration;
    else if (key == "successful_searches")
      metadata >> recovered.successful_searches;
    else if (key == "failed_searches")
      metadata >> recovered.failed_searches;
    else if (key == "regrid_events")
      metadata >> recovered.regrid_events;
    else if (key == "last_regrid_iteration")
      metadata >> recovered.last_regrid_iteration;
    else if (key == "surfaces")
      metadata >> recovered.declared_surfaces;
    else if (key == "surface") {
      SurfaceRecord surface;
      int lifecycle = -1;
      int role = -1;
      int relaxation_state = -1;
      int center_policy = -1;
      metadata >> surface.stable_id >> surface.first_particle_id >>
          surface.particle_count >> surface.configured_slot >>
          surface.generation >> surface.ntheta >> surface.nphi >>
          surface.angular_levels >> surface.active_angular_level;
      if (surface.angular_levels <= 0 ||
          surface.angular_levels > max_angular_layers)
        CCTK_ERROR("ParticleAHFinderX checkpoint has an invalid angular "
                   "layer count");
      for (int level = 0; level < surface.angular_levels; ++level) {
        auto &layer = surface.angular_layer[level];
        int layer_state = -1;
        metadata >> layer.first_particle_id >> layer.particle_count >>
            layer.logical_owner_offset >> layer.ntheta >> layer.nphi >>
            layer_state >> layer.last_relaxation_steps >> layer.expansion_l2 >>
            layer.expansion_linf;
        layer.relaxation_state =
            static_cast<SurfaceRelaxationState>(layer_state);
      }
      metadata >>
          surface.center[0] >> surface.center[1] >> surface.center[2] >>
          surface.radius >> surface.mass_scale >>
          surface.tolerances.l2_times_mass >>
          surface.tolerances.linf_times_mass >> lifecycle >> role >>
          surface.parent_ids[0] >> surface.parent_ids[1] >>
          surface.compatibility_slot >> relaxation_state >>
          surface.centroid[0] >> surface.centroid[1] >>
          surface.centroid[2] >> surface.area >> surface.mean_radius >>
          surface.minimum_radius >> surface.maximum_radius >>
          surface.expansion_l2 >> surface.expansion_linf >>
          surface.mean_expansion;
      for (auto &value : surface.proper_circumference)
        metadata >> value;
      for (auto &value : surface.coordinate_quadrupole)
        metadata >> value;
      for (auto &value : surface.position_minimum)
        metadata >> value;
      for (auto &value : surface.position_maximum)
        metadata >> value;
      metadata >> surface.invalid_points >> surface.minimum_sampled_level >>
          surface.maximum_sampled_level >> surface.sampled_level_changes >>
          surface.cumulative_sampled_level_changes >>
          surface.last_search_iteration >>
          surface.last_success_iteration >> surface.last_search_time >>
          surface.last_search_steps >> surface.search_every >>
          surface.next_search_iteration >> surface.consecutive_failures >>
          center_policy >> surface.predict_center >>
          surface.external_center_weight;
      for (auto &value : surface.manual_center)
        metadata >> value;
      metadata >> surface.last_history_iteration >>
          surface.center_history.count;
      for (int sample = 0; sample < 3; ++sample) {
        metadata >> surface.center_history.times[sample];
        for (auto &value : surface.center_history.centers[sample])
          metadata >> value;
        metadata >> surface.center_history.mean_radii[sample] >>
            surface.center_history.external_valid[sample];
        for (auto &value : surface.center_history.external_centers[sample])
          metadata >> value;
      }
      metadata >> surface.birth_iteration >>
          surface.last_state_change_iteration >> surface.candidate_attempts >>
          surface.promoted_iteration >> surface.retire_after_iteration >>
          surface.birth_time >> surface.candidate_radius_floor >>
          surface.candidate_radius_ceiling >>
          surface.candidate_trigger_distance >>
          surface.candidate_expiry_distance >> surface.published;
      metadata >> surface.consecutive_domain_exit_checks >>
          surface.last_domain_exit_check_iteration >>
          surface.retired_outside_domain;
      surface.lifecycle = static_cast<ParticleLifecycle>(lifecycle);
      surface.role = static_cast<SurfaceRole>(role);
      surface.relaxation_state =
          static_cast<SurfaceRelaxationState>(relaxation_state);
      surface.center_policy = static_cast<CenterPolicy>(center_policy);
      recovered.surfaces.push_back(surface);
    } else if (key == "complete")
      metadata >> recovered.complete;
    else {
      std::string ignored;
      std::getline(metadata, ignored);
    }
    if (!metadata)
      CCTK_VERROR("ParticleAHFinderX recovery manifest '%s' is malformed",
                  filename.c_str());
  }
  return recovered;
}

void validate_metadata(const RecoveredState &recovered,
                       const cGH *const cctkGH, const Runtime &state) {
  const double time_scale =
      std::max({1.0, std::abs(recovered.time),
                std::abs(static_cast<double>(cctkGH->cctk_time))});
  if (recovered.schema != checkpoint_schema ||
      recovered.iteration != cctkGH->cctk_iteration ||
      std::abs(recovered.time - cctkGH->cctk_time) >
          32 * std::numeric_limits<double>::epsilon() * time_scale ||
      recovered.patches != state.containers.size() ||
      recovered.dimension != AMREX_SPACEDIM ||
      recovered.real_size != int(sizeof(amrex::Real)) ||
      recovered.topology != "latitude_longitude_cell_centered" ||
      recovered.configuration != configuration_hash() ||
      recovered.declared_surfaces != recovered.surfaces.size() ||
      recovered.complete != 1)
    CCTK_ERROR("ParticleAHFinderX checkpoint manifest is incompatible with "
               "this run");

  amrex::Long expected_particles = 0;
  amrex::Long maximum_surface_id = 0;
  amrex::Long maximum_particle_end = 0;
  std::array<amrex::Long, max_compatibility_slots + 1>
      compatibility_surface_ids{};
  for (std::size_t i = 0; i < recovered.surfaces.size(); ++i) {
    const auto &surface = recovered.surfaces[i];
    const bool valid_lifecycle =
        surface.lifecycle == ParticleLifecycle::candidate ||
        surface.lifecycle == ParticleLifecycle::active ||
        surface.lifecycle == ParticleLifecycle::retired;
    const bool valid_role = surface.role == SurfaceRole::individual ||
                            surface.role == SurfaceRole::common ||
                            surface.role == SurfaceRole::other;
    const bool valid_relaxation_state =
        surface.relaxation_state == SurfaceRelaxationState::idle ||
        surface.relaxation_state == SurfaceRelaxationState::converged ||
        surface.relaxation_state == SurfaceRelaxationState::failed;
    bool valid_layers =
        surface.angular_levels > 0 &&
        surface.angular_levels <= max_angular_layers &&
        surface.active_angular_level >= 0 &&
        surface.active_angular_level < surface.angular_levels;
    amrex::Long layer_particles = 0;
    for (int level = 0; level < surface.angular_levels; ++level) {
      const auto &layer = surface.angular_layer[level];
      const bool valid_layer_state =
          layer.relaxation_state == SurfaceRelaxationState::idle ||
          layer.relaxation_state == SurfaceRelaxationState::converged ||
          layer.relaxation_state == SurfaceRelaxationState::refine ||
          layer.relaxation_state == SurfaceRelaxationState::failed;
      valid_layers &=
          layer.first_particle_id > 0 && layer.particle_count > 0 &&
          layer.logical_owner_offset >= 0 && layer.ntheta > 0 &&
          layer.nphi > 0 &&
          layer.particle_count == amrex::Long(layer.ntheta) * layer.nphi &&
          valid_layer_state && layer.last_relaxation_steps >= 0 &&
          std::isfinite(layer.expansion_l2) &&
          std::isfinite(layer.expansion_linf) && layer.expansion_l2 >= 0 &&
          layer.expansion_linf >= 0 &&
          layer_particles <= std::numeric_limits<amrex::Long>::max() -
                                 layer.particle_count;
      if (valid_layers)
        layer_particles += layer.particle_count;
      if (layer.first_particle_id > 0 && layer.particle_count > 0 &&
          layer.first_particle_id <=
              std::numeric_limits<amrex::Long>::max() -
                  layer.particle_count)
        maximum_particle_end = std::max(
            maximum_particle_end,
            layer.first_particle_id + layer.particle_count);
    }
    if (valid_layers) {
      const auto &target =
          surface.angular_layer[surface.angular_levels - 1];
      valid_layers &= surface.first_particle_id == target.first_particle_id &&
                      surface.particle_count == target.particle_count &&
                      surface.logical_owner_offset ==
                          target.logical_owner_offset &&
                      surface.ntheta == target.ntheta &&
                      surface.nphi == target.nphi;
    }
    const bool valid_center_policy =
        surface.center_policy == CenterPolicy::manual ||
        surface.center_policy == CenterPolicy::external ||
        surface.center_policy == CenterPolicy::centroid ||
        surface.center_policy == CenterPolicy::external_then_centroid ||
        surface.center_policy == CenterPolicy::centroid_then_external ||
        surface.center_policy == CenterPolicy::blend_external_centroid ||
        surface.center_policy == CenterPolicy::parent_midpoint ||
        surface.center_policy ==
            CenterPolicy::puncture_tracker_displacement;
    bool valid_history = surface.center_history.count >= 0 &&
                         surface.center_history.count <= 3;
    for (int sample = 0; sample < 3; ++sample) {
      valid_history &= surface.center_history.external_valid[sample] == 0 ||
                       surface.center_history.external_valid[sample] == 1;
      for (const auto value :
           surface.center_history.external_centers[sample])
        valid_history &= std::isfinite(value);
    }
    for (int sample = 0; sample < surface.center_history.count; ++sample) {
      valid_history &=
          std::isfinite(surface.center_history.times[sample]) &&
          surface.center_history.mean_radii[sample] > 0 &&
          std::isfinite(surface.center_history.mean_radii[sample]);
      for (const auto value : surface.center_history.centers[sample])
        valid_history &= std::isfinite(value);
      if (sample > 0)
        valid_history &= surface.center_history.times[sample - 1] <
                         surface.center_history.times[sample];
    }
    const bool is_common_candidate =
        surface.role == SurfaceRole::common && !surface.published;
    const bool valid_publication_state =
        (surface.lifecycle != ParticleLifecycle::candidate ||
         (!surface.published && surface.role == SurfaceRole::common &&
          surface.promoted_iteration < 0)) &&
        (surface.promoted_iteration < 0 ||
         (surface.published && surface.role == SurfaceRole::common));
    const bool valid_candidate_state =
        surface.candidate_attempts >= 0 &&
        surface.consecutive_domain_exit_checks >= 0 &&
        surface.last_domain_exit_check_iteration <= recovered.iteration &&
        surface.last_domain_exit_check_iteration >= -1 &&
        ((surface.consecutive_domain_exit_checks == 0) ==
         (surface.last_domain_exit_check_iteration == -1)) &&
        (!surface.retired_outside_domain ||
         surface.lifecycle == ParticleLifecycle::retired) &&
        surface.birth_iteration >= 0 &&
        surface.birth_iteration <= recovered.iteration &&
        surface.last_state_change_iteration >= surface.birth_iteration &&
        surface.last_state_change_iteration <= recovered.iteration &&
        surface.promoted_iteration <= recovered.iteration &&
        surface.retire_after_iteration >= -1 &&
        std::isfinite(surface.birth_time) &&
        surface.candidate_radius_floor >= 0 &&
        surface.candidate_radius_ceiling >=
            surface.candidate_radius_floor &&
        surface.candidate_trigger_distance >= 0 &&
        surface.candidate_expiry_distance >=
            surface.candidate_trigger_distance &&
        (!is_common_candidate ||
         (surface.parent_ids[0] > 0 && surface.parent_ids[1] > 0 &&
          surface.candidate_radius_floor > 0 &&
          surface.candidate_trigger_distance > 0));
    if (surface.stable_id <= 0 || surface.first_particle_id <= 0 ||
        surface.particle_count <= 0 || surface.ntheta <= 0 ||
        surface.nphi <= 0 ||
        surface.particle_count !=
            amrex::Long(surface.ntheta) * surface.nphi ||
        !(surface.radius > 0) || !(surface.mass_scale > 0) ||
        !(surface.tolerances.l2_times_mass > 0) ||
        !(surface.tolerances.linf_times_mass > 0) || !valid_lifecycle ||
        !valid_role || !valid_relaxation_state || !valid_layers ||
        !valid_center_policy ||
        !valid_history || !valid_candidate_state ||
        !valid_publication_state || surface.search_every < 0 ||
        surface.next_search_iteration < 0 ||
        surface.consecutive_failures < 0 ||
        !(surface.external_center_weight >= 0 &&
          surface.external_center_weight <= 1) ||
        !std::isfinite(surface.manual_center[0]) ||
        !std::isfinite(surface.manual_center[1]) ||
        !std::isfinite(surface.manual_center[2]) ||
        surface.configured_slot < -1 ||
        surface.configured_slot >= max_configured_surfaces ||
        surface.compatibility_slot == 0 || surface.compatibility_slot < -1 ||
        surface.compatibility_slot > max_compatibility_slots ||
        surface.parent_ids[0] < 0 || surface.parent_ids[1] < 0 ||
        surface.last_search_iteration > recovered.iteration ||
        surface.last_success_iteration > recovered.iteration ||
        surface.last_history_iteration > recovered.iteration ||
        surface.last_search_steps < 0 || surface.invalid_points < 0 ||
        !((surface.minimum_sampled_level == -1 &&
           surface.maximum_sampled_level == -1) ||
          (surface.minimum_sampled_level >= 0 &&
           surface.maximum_sampled_level >=
               surface.minimum_sampled_level)) ||
        surface.sampled_level_changes < 0 ||
        surface.sampled_level_changes > surface.particle_count ||
        surface.cumulative_sampled_level_changes <
            surface.sampled_level_changes ||
        surface.proper_circumference[0] < 0 ||
        surface.proper_circumference[1] < 0 ||
        surface.proper_circumference[2] < 0 ||
        !std::isfinite(surface.proper_circumference[0]) ||
        !std::isfinite(surface.proper_circumference[1]) ||
        !std::isfinite(surface.proper_circumference[2]) ||
        (i > 0 && recovered.surfaces[i - 1].stable_id >=
                      surface.stable_id) ||
        surface.first_particle_id >
            std::numeric_limits<amrex::Long>::max() -
                surface.particle_count ||
        (surface.lifecycle != ParticleLifecycle::retired &&
         expected_particles > std::numeric_limits<amrex::Long>::max() -
                                  layer_particles))
      CCTK_ERROR("ParticleAHFinderX checkpoint contains invalid surface "
                 "metadata");
    if (surface.compatibility_slot > 0) {
      auto &slot = compatibility_surface_ids[surface.compatibility_slot];
      if (slot != 0)
        CCTK_ERROR("ParticleAHFinderX checkpoint assigns one compatibility "
                   "slot to multiple stable surfaces");
      slot = surface.stable_id;
    }
    if (surface.lifecycle != ParticleLifecycle::retired)
      expected_particles += layer_particles;
    maximum_surface_id = surface.stable_id;
  }
  amrex::Long expected_creation_events = 0;
  for (const auto &surface : recovered.surfaces)
    expected_creation_events += surface.angular_levels;
  if (expected_particles != recovered.global_particle_count ||
      recovered.population_creation_events !=
          expected_creation_events ||
      recovered.next_surface_id <= maximum_surface_id ||
      recovered.next_particle_id < maximum_particle_end ||
      recovered.last_particle_output_iteration > recovered.iteration ||
      recovered.last_diagnostics_iteration > recovered.iteration ||
      recovered.last_spherical_export_iteration > recovered.iteration ||
      recovered.regrid_events < 0 ||
      recovered.last_regrid_iteration > recovered.iteration ||
      recovered.last_regrid_iteration < -1 ||
      ((recovered.regrid_events == 0) !=
       (recovered.last_regrid_iteration == -1)))
    CCTK_ERROR("ParticleAHFinderX checkpoint counters are inconsistent");
}

void recover_checkpoint(const cGH *const cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  if (!state.setup)
    CCTK_ERROR("ParticleAHFinderX recovery ran before setup");
  if (state.solver_in_flight || state.checkpoint_in_progress)
    CCTK_ERROR("ParticleAHFinderX recovery requested while work is in flight");

  const std::string root =
      checkpoint_root(recover_dir, recover_file, cctkGH->cctk_iteration);
  RecoveredState recovered = read_metadata(root + "/" + metadata_name);
  validate_metadata(recovered, cctkGH, state);

  for (std::size_t patch = 0; patch < state.containers.size(); ++patch)
    state.containers[patch]->Restart(root + "/" + patch_name(patch),
                                     particle_dataset_name);

  state.surfaces = std::move(recovered.surfaces);
  assign_logical_owners(state.surfaces);
  state.logical_surfaces = std::make_unique<LogicalSurfaceStorage>();
  state.logical_surfaces->initialize(local_logical_seeds(state.surfaces));
  const auto ownership =
      ParticleAHFinderX::logical_ownership(state.surfaces);
  std::vector<amrex::Long> surface_counts(ownership.size(), 0);
  for (auto &container : state.containers) {
    const auto local_counts = container->remap_logical_ownership(ownership);
    for (std::size_t surface = 0; surface < surface_counts.size(); ++surface)
      surface_counts[surface] += local_counts[surface];
    container->rebuild_amr_masks();
    container->redistribute_global();
  }
  if (!surface_counts.empty())
    amrex::ParallelDescriptor::ReduceLongSum(
        surface_counts.data(), static_cast<int>(surface_counts.size()));
  for (std::size_t surface = 0; surface < surface_counts.size(); ++surface) {
    const auto record = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(),
        ownership[surface].surface_id,
        [](const SurfaceRecord &candidate, const amrex::Long stable_id) {
          return candidate.stable_id < stable_id;
        });
    if (record == state.surfaces.end() ||
        record->stable_id != ownership[surface].surface_id)
      CCTK_ERROR("ParticleAHFinderX recovered ownership has an unknown "
                 "surface ID");
    const int level = ownership[surface].angular_level;
    if (level < 0 || level >= record->angular_levels)
      CCTK_ERROR("ParticleAHFinderX recovered ownership has an invalid "
                 "angular layer");
    const amrex::Long expected = record->angular_layer[level].particle_count;
    if (surface_counts[surface] != expected)
      CCTK_VERROR("ParticleAHFinderX restored %lld particles for surface %lld "
                  "angular level %d, expected %lld",
                  static_cast<long long>(surface_counts[surface]),
                  static_cast<long long>(record->stable_id),
                  level, static_cast<long long>(expected));
  }

  amrex::Long global_particles = 0;
  for (const auto &container : state.containers)
    global_particles += container->local_particle_count();
  amrex::ParallelDescriptor::ReduceLongSum(global_particles);
  if (global_particles != recovered.global_particle_count)
    CCTK_VERROR("ParticleAHFinderX restored %lld particles, expected %lld",
                static_cast<long long>(global_particles),
                static_cast<long long>(recovered.global_particle_count));

  state.next_surface_id = recovered.next_surface_id;
  state.next_particle_id = recovered.next_particle_id;
  state.global_particle_count = global_particles;
  state.population_creation_events = recovered.population_creation_events;
  state.last_particle_output_iteration =
      recovered.last_particle_output_iteration;
  state.last_diagnostics_iteration = recovered.last_diagnostics_iteration;
  state.last_spherical_export_iteration =
      recovered.last_spherical_export_iteration;
  state.successful_searches = recovered.successful_searches;
  state.failed_searches = recovered.failed_searches;
  state.regrid_events = recovered.regrid_events;
  state.last_regrid_iteration = recovered.last_regrid_iteration;
  state.last_checkpoint_iteration = cctkGH->cctk_iteration;
  state.recovered_iteration = cctkGH->cctk_iteration;
  state.initial_surfaces_created = true;
  state.recovered = true;
  state.last_checkpoint_runtime = CCTK_RunTime();
  state.last_adm_gather_iteration = -1;
  state.last_logical_route_iteration = -1;

  if (verbose)
    CCTK_VINFO("Recovered %zu persistent surface population(s) and %lld "
               "particles; logical ownership was rebuilt for %d rank(s)",
               state.surfaces.size(),
               static_cast<long long>(state.global_particle_count),
               amrex::ParallelDescriptor::NProcs());
}

} // namespace

extern "C" void ParticleAHFinderX_CheckpointInitial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  if (checkpoint_ID) {
    write_checkpoint(cctkGH);
    runtime().last_checkpoint_runtime = CCTK_RunTime();
  }
}

extern "C" void ParticleAHFinderX_Checkpoint(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  int run_time = CCTK_RunTime();
  MPI_Bcast(&run_time, 1, MPI_INT, 0,
            amrex::ParallelDescriptor::Communicator());
  const bool by_iteration =
      checkpoint_every > 0 && cctk_iteration % checkpoint_every == 0;
  const bool by_wall_time =
      checkpoint_every_walltime_hours > 0 &&
      run_time >= runtime().last_checkpoint_runtime +
                      std::lrint(checkpoint_every_walltime_hours * 3600);
  if (by_iteration || by_wall_time) {
    write_checkpoint(cctkGH);
    runtime().last_checkpoint_runtime = run_time;
  }
}

extern "C" void ParticleAHFinderX_CheckpointTerminate(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  if (checkpoint_on_terminate)
    write_checkpoint(cctkGH);
}

extern "C" void ParticleAHFinderX_Recover(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  if (!CCTK_EQUALS(recover, "no"))
    recover_checkpoint(cctkGH);
}

} // namespace ParticleAHFinderX
