/**
 * \file candidate_manager.cxx
 * \brief Pairwise provisional common-surface discovery and lifecycle.
 */
#include "candidate_manager.hxx"

#include "surface_manager.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace ParticleAHFinderX {
namespace {

std::array<amrex::Real, 3> tracking_center(const SurfaceRecord &surface) {
  return surface.last_success_iteration >= 0 ? surface.centroid
                                              : surface.center;
}

amrex::Real bounding_radius(const SurfaceRecord &surface) {
  return surface.maximum_radius > 0 ? surface.maximum_radius
                                    : surface.radius;
}

amrex::Real separation(const SurfaceRecord &first,
                       const SurfaceRecord &second) {
  const auto first_center = tracking_center(first);
  const auto second_center = tracking_center(second);
  amrex::Real distance_squared = 0;
  for (int d = 0; d < 3; ++d) {
    const amrex::Real difference = first_center[d] - second_center[d];
    distance_squared += difference * difference;
  }
  return std::sqrt(distance_squared);
}

std::array<amrex::Real, 3>
weighted_center(const SurfaceRecord &first, const SurfaceRecord &second) {
  const auto first_center = tracking_center(first);
  const auto second_center = tracking_center(second);
  const amrex::Real total_mass = first.mass_scale + second.mass_scale;
  std::array<amrex::Real, 3> center{{0, 0, 0}};
  for (int d = 0; d < 3; ++d)
    center[d] = (first.mass_scale * first_center[d] +
                 second.mass_scale * second_center[d]) /
                total_mass;
  return center;
}

amrex::Real distance_to(const std::array<amrex::Real, 3> &first,
                        const std::array<amrex::Real, 3> &second) {
  amrex::Real distance_squared = 0;
  for (int d = 0; d < 3; ++d) {
    const amrex::Real difference = first[d] - second[d];
    distance_squared += difference * difference;
  }
  return std::sqrt(distance_squared);
}

bool same_parent_pair(const SurfaceRecord &surface, const amrex::Long first,
                      const amrex::Long second) {
  return surface.parent_ids[0] == first && surface.parent_ids[1] == second;
}

const SurfaceRecord *find_surface(const std::vector<SurfaceRecord> &surfaces,
                                  const amrex::Long stable_id) {
  const auto found = std::lower_bound(
      surfaces.begin(), surfaces.end(), stable_id,
      [](const SurfaceRecord &surface, const amrex::Long id) {
        return surface.stable_id < id;
      });
  return found != surfaces.end() && found->stable_id == stable_id
             ? &*found
             : nullptr;
}

bool candidate_contains_parents(const SurfaceRecord &candidate,
                                const std::vector<SurfaceRecord> &surfaces,
                                const amrex::Real tolerance_fraction) {
  for (const auto parent_id : candidate.parent_ids) {
    const auto *const parent = find_surface(surfaces, parent_id);
    if (!parent || parent->lifecycle == ParticleLifecycle::retired ||
        parent->last_success_iteration < 0 || !(parent->area > 0) ||
        !(candidate.area > parent->area * (1 - tolerance_fraction)))
      return false;
    const amrex::Real coordinate_scale = amrex::max(
        amrex::Real(1),
        amrex::max(candidate.maximum_radius, parent->maximum_radius));
    const amrex::Real tolerance = tolerance_fraction * coordinate_scale;
    for (int d = 0; d < 3; ++d)
      if (candidate.position_minimum[d] >
              parent->position_minimum[d] + tolerance ||
          candidate.position_maximum[d] <
              parent->position_maximum[d] - tolerance)
        return false;
    if (distance_to(candidate.centroid, parent->centroid) >
        candidate.maximum_radius + tolerance)
      return false;
  }
  return true;
}

bool duplicates_published_surface(
    const SurfaceRecord &candidate,
    const std::vector<SurfaceRecord> &surfaces,
    const amrex::Real center_tolerance,
    const amrex::Real area_tolerance) {
  for (const auto &surface : surfaces) {
    if (surface.stable_id == candidate.stable_id || !surface.published ||
        surface.lifecycle != ParticleLifecycle::active ||
        surface.role != SurfaceRole::common || !(surface.area > 0))
      continue;
    const amrex::Real radius =
        amrex::min(candidate.mean_radius, surface.mean_radius);
    if (!(radius > 0) ||
        distance_to(candidate.centroid, surface.centroid) >
            center_tolerance * radius)
      continue;
    const amrex::Real relative_area_difference =
        std::abs(candidate.area - surface.area) /
        amrex::max(candidate.area, surface.area);
    if (relative_area_difference <= area_tolerance)
      return true;
  }
  return false;
}

int next_compatibility_slot(const std::vector<SurfaceRecord> &surfaces) {
  for (int slot = 1; slot <= max_compatibility_slots; ++slot) {
    bool used = false;
    for (const auto &surface : surfaces)
      used |= surface.compatibility_slot == slot;
    if (!used)
      return slot;
  }
  return -1;
}

void birth_candidates(Runtime &state,
                      const std::vector<std::array<std::size_t, 2>> &pairs,
                      const int iteration, const amrex::Real time) {
  DECLARE_CCTK_PARAMETERS;
  if (pairs.empty())
    return;
  std::vector<SurfaceSeed> seeds;
  std::vector<SurfaceRecord> new_records;
  seeds.reserve(pairs.size());
  new_records.reserve(pairs.size());
  amrex::Long total_particles = 0;

  for (const auto &pair : pairs) {
    const auto first = state.surfaces[pair[0]];
    const auto second = state.surfaces[pair[1]];
    if (state.next_surface_id == std::numeric_limits<amrex::Long>::max())
      CCTK_ERROR("ParticleAHFinderX stable surface ID allocator overflowed");
    const auto center = weighted_center(first, second);
    const amrex::Real enclosing_radius = amrex::max(
        distance_to(center, tracking_center(first)) + bounding_radius(first),
        distance_to(center, tracking_center(second)) +
            bounding_radius(second));
    const amrex::Real radius_floor =
        candidate_initial_radius_factor * enclosing_radius;
    const amrex::Real radius_ceiling =
        candidate_maximum_radius_factor * enclosing_radius;

    SurfaceSeed seed;
    seed.surface_id = state.next_surface_id++;
    seed.first_particle_id = state.next_particle_id;
    seed.generation = 0;
    seed.angular_level = 0;
    seed.angular_levels = 1;
    seed.ntheta = candidate_ntheta;
    seed.nphi = candidate_nphi;
    seed.center = center;
    seed.radius = radius_floor;
    seed.lifecycle = ParticleLifecycle::candidate;
    seed.role = SurfaceRole::common;
    const amrex::Long count = seed.particle_count();

    SurfaceRecord record;
    record.stable_id = seed.surface_id;
    record.first_particle_id = seed.first_particle_id;
    record.particle_count = count;
    record.parent_ids = {{first.stable_id, second.stable_id}};
    record.search_every = candidate_find_every;
    record.next_search_iteration = iteration;
    record.birth_iteration = iteration;
    record.last_state_change_iteration = iteration;
    record.ntheta = seed.ntheta;
    record.nphi = seed.nphi;
    record.center_policy = CenterPolicy::parent_midpoint;
    record.manual_center = center;
    record.center = center;
    record.radius = seed.radius;
    record.mass_scale = first.mass_scale + second.mass_scale;
    record.tolerances = {candidate_theta_l2_tolerance,
                         candidate_theta_linf_tolerance};
    record.lifecycle = seed.lifecycle;
    record.role = seed.role;
    record.birth_time = time;
    record.candidate_radius_floor = radius_floor;
    record.candidate_radius_ceiling = radius_ceiling;
    record.candidate_trigger_distance =
        candidate_trigger_factor *
        (bounding_radius(first) + bounding_radius(second));
    record.candidate_expiry_distance =
        candidate_expiry_factor * record.candidate_trigger_distance;
    initialize_angular_layers(record, enable_angular_continuation,
                              candidate_angular_continuation_min_ntheta,
                              angular_continuation_max_levels);
    const amrex::Long total_count = total_angular_particle_count(record);
    if (state.next_particle_id >
            std::numeric_limits<amrex::Long>::max() - total_count ||
        total_particles >
            std::numeric_limits<amrex::Long>::max() - total_count)
      CCTK_ERROR("ParticleAHFinderX dynamic particle ID allocator overflowed");
    assign_new_logical_owner(record, state.surfaces);
    state.surfaces.push_back(record);
    new_records.push_back(record);
    const auto new_seeds = particle_seeds({record});
    seeds.insert(seeds.end(), new_seeds.begin(), new_seeds.end());
    state.next_particle_id += total_count;
    total_particles += total_count;
  }

  if (state.global_particle_count >
      std::numeric_limits<amrex::Long>::max() - total_particles)
    CCTK_ERROR("ParticleAHFinderX dynamic global particle count overflowed");

  for (auto &container : state.containers)
    container->create_surfaces(seeds);
  state.logical_surfaces->append_surfaces(local_logical_seeds(new_records));
  for (auto &container : state.containers)
    container->redistribute_global();
  state.global_particle_count += total_particles;
  if (seeds.size() >
          static_cast<std::size_t>(std::numeric_limits<amrex::Long>::max()) ||
      state.population_creation_events >
          std::numeric_limits<amrex::Long>::max() -
              static_cast<amrex::Long>(seeds.size()))
    CCTK_ERROR("ParticleAHFinderX population creation counter overflowed");
  state.population_creation_events += static_cast<amrex::Long>(seeds.size());

  amrex::Long actual_particles = 0;
  for (const auto &container : state.containers)
    actual_particles += container->local_particle_count();
  amrex::ParallelDescriptor::ReduceLongSum(actual_particles);
  if (actual_particles != state.global_particle_count)
    CCTK_ERROR("ParticleAHFinderX dynamic birth produced an inconsistent "
               "global particle count");
}

} // namespace

void manage_candidates_before_search(Runtime &state, const int iteration,
                                     const amrex::Real time) {
  DECLARE_CCTK_PARAMETERS;
  if (!enable_dynamic_common_horizons && !retire_parents_after_common)
    return;

  std::vector<amrex::Long> retire_ids;
  for (auto &surface : state.surfaces) {
    if (surface.lifecycle == ParticleLifecycle::active &&
        surface.retire_after_iteration >= 0 &&
        iteration >= surface.retire_after_iteration) {
      surface.last_state_change_iteration = iteration;
      retire_ids.push_back(surface.stable_id);
      continue;
    }
    if (surface.lifecycle != ParticleLifecycle::candidate ||
        surface.role != SurfaceRole::common)
      continue;
    const auto *const first =
        find_surface(state.surfaces, surface.parent_ids[0]);
    const auto *const second =
        find_surface(state.surfaces, surface.parent_ids[1]);
    if (!first || !second || first->lifecycle == ParticleLifecycle::retired ||
        second->lifecycle == ParticleLifecycle::retired ||
        separation(*first, *second) > surface.candidate_expiry_distance) {
      surface.next_search_iteration =
          iteration + candidate_retry_max_iterations;
      surface.last_state_change_iteration = iteration;
      retire_ids.push_back(surface.stable_id);
    }
  }
  if (!retire_ids.empty())
    retire_surface_populations(state, retire_ids);
  if (!enable_dynamic_common_horizons)
    return;

  int active_candidates = 0;
  for (const auto &surface : state.surfaces)
    active_candidates +=
        surface.lifecycle == ParticleLifecycle::candidate &&
        surface.role == SurfaceRole::common;
  if (active_candidates >= maximum_active_candidates)
    return;

  std::vector<std::array<std::size_t, 2>> pairs;
  const std::size_t existing_surface_count = state.surfaces.size();
  for (std::size_t first_index = 0;
       first_index < existing_surface_count &&
       pairs.size() <
           static_cast<std::size_t>(maximum_new_candidates_per_search) &&
       active_candidates + static_cast<int>(pairs.size()) <
           maximum_active_candidates;
       ++first_index) {
    const auto &first = state.surfaces[first_index];
    if (first.lifecycle != ParticleLifecycle::active || !first.published ||
        first.role != SurfaceRole::individual ||
        first.last_success_iteration < 0)
      continue;
    for (std::size_t second_index = first_index + 1;
         second_index < existing_surface_count &&
         pairs.size() <
             static_cast<std::size_t>(maximum_new_candidates_per_search) &&
         active_candidates + static_cast<int>(pairs.size()) <
             maximum_active_candidates;
         ++second_index) {
      const auto &second = state.surfaces[second_index];
      if (second.lifecycle != ParticleLifecycle::active || !second.published ||
          second.role != SurfaceRole::individual ||
          second.last_success_iteration < 0)
        continue;
      const amrex::Long first_id = first.stable_id;
      const amrex::Long second_id = second.stable_id;
      bool already_live = false;
      int cooldown_until = -1;
      for (const auto &surface : state.surfaces) {
        if (!same_parent_pair(surface, first_id, second_id))
          continue;
        if (surface.lifecycle != ParticleLifecycle::retired)
          already_live = true;
        else
          cooldown_until = amrex::max(cooldown_until,
                                      surface.next_search_iteration);
      }
      if (already_live || iteration < cooldown_until)
        continue;
      const amrex::Real trigger_distance =
          candidate_trigger_factor *
          (bounding_radius(first) + bounding_radius(second));
      if (separation(first, second) <= trigger_distance)
        pairs.push_back({first_index, second_index});
    }
  }
  birth_candidates(state, pairs, iteration, time);
}

void finalize_candidates_after_search(Runtime &state, const int iteration,
                                      const amrex::Real time) {
  DECLARE_CCTK_PARAMETERS;
  static_cast<void>(time);
  std::vector<SurfaceStateUpdate> promotions;
  std::vector<SurfaceSphereReset> resets;
  std::vector<amrex::Long> retire_ids;

  for (auto &candidate : state.surfaces) {
    if (candidate.lifecycle != ParticleLifecycle::candidate ||
        candidate.role != SurfaceRole::common ||
        candidate.last_search_iteration != iteration)
      continue;
    ++candidate.candidate_attempts;
    bool promote =
        candidate.relaxation_state == SurfaceRelaxationState::converged &&
        candidate.invalid_points == 0 && candidate.area > 0 &&
        candidate_contains_parents(candidate, state.surfaces,
                                   candidate_containment_tolerance);
    if (promote)
      promote = !duplicates_published_surface(
          candidate, state.surfaces,
          candidate_deduplication_center_tolerance,
          candidate_deduplication_area_tolerance);
    if (promote) {
      candidate.published = true;
      candidate.promoted_iteration = iteration;
      candidate.last_state_change_iteration = iteration;
      candidate.compatibility_slot = next_compatibility_slot(state.surfaces);
      promotions.push_back({candidate.stable_id, ParticleLifecycle::active,
                            SurfaceRole::common});
      if (retire_parents_after_common)
        for (const auto parent_id : candidate.parent_ids) {
          auto parent = std::lower_bound(
              state.surfaces.begin(), state.surfaces.end(), parent_id,
              [](const SurfaceRecord &surface, const amrex::Long stable_id) {
                return surface.stable_id < stable_id;
              });
          if (parent != state.surfaces.end() && parent->stable_id == parent_id &&
              parent->lifecycle == ParticleLifecycle::active)
            parent->retire_after_iteration =
                iteration + common_parent_grace_iterations;
        }
      continue;
    }

    candidate.relaxation_state = SurfaceRelaxationState::failed;
    if (candidate.candidate_attempts >= maximum_candidate_attempts) {
      candidate.next_search_iteration =
          iteration + candidate_retry_max_iterations;
      candidate.last_state_change_iteration = iteration;
      retire_ids.push_back(candidate.stable_id);
      continue;
    }
    int delay = candidate_retry_base_iterations;
    for (int attempt = 1; attempt < candidate.candidate_attempts &&
                          delay < candidate_retry_max_iterations;
         ++attempt)
      delay = amrex::min(candidate_retry_max_iterations,
                         delay > candidate_retry_max_iterations / 2
                             ? candidate_retry_max_iterations
                             : 2 * delay);
    candidate.next_search_iteration = iteration + delay;
    candidate.last_state_change_iteration = iteration;
    const amrex::Real next_radius =
        amrex::min(candidate.candidate_radius_ceiling,
                   candidate.radius * candidate_radius_growth_factor);
    resets.push_back({candidate.stable_id, candidate.center, next_radius});
  }

  if (!promotions.empty())
    update_surface_states(state, promotions);
  if (!resets.empty())
    reset_surface_spheres(state, resets);
  if (!retire_ids.empty())
    retire_surface_populations(state, retire_ids);
}

} // namespace ParticleAHFinderX
