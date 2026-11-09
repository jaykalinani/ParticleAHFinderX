/**
 * \file candidate_manager.cxx
 * \brief Bounded hierarchical common-surface discovery and lifecycle.
 */
#include "candidate_manager.hxx"

#include "candidate_events.hxx"
#include "surface_manager.hxx"

#include <cctk.h>
#include <cctk_Parameters.h>

#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <utility>
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

bool has_composed_lineage(const SurfaceRecord &surface) {
  return surface.role == SurfaceRole::common && surface.parent_ids[0] > 0 &&
         surface.parent_ids[1] > 0;
}

bool is_discovery_parent(const SurfaceRecord &surface) {
  return surface.lifecycle == ParticleLifecycle::active &&
         surface.published && surface.last_success_iteration >= 0 &&
         (surface.role == SurfaceRole::individual ||
          has_composed_lineage(surface));
}

std::vector<std::vector<amrex::Long>>
constituent_lineages(const std::vector<SurfaceRecord> &surfaces) {
  std::vector<std::vector<amrex::Long>> result(surfaces.size());
  for (std::size_t index = 0; index < surfaces.size(); ++index) {
    const auto &surface = surfaces[index];
    if (!has_composed_lineage(surface)) {
      result[index] = {surface.stable_id};
      continue;
    }
    std::array<std::size_t, 2> parent_index;
    for (int parent = 0; parent < 2; ++parent) {
      const auto found = std::lower_bound(
          surfaces.begin(), surfaces.end(), surface.parent_ids[parent],
          [](const SurfaceRecord &candidate, const amrex::Long stable_id) {
            return candidate.stable_id < stable_id;
          });
      if (found == surfaces.end() ||
          found->stable_id != surface.parent_ids[parent] ||
          found >= surfaces.begin() + index)
        CCTK_ERROR("ParticleAHFinderX common-surface lineage must reference "
                   "two older surfaces");
      parent_index[parent] =
          static_cast<std::size_t>(found - surfaces.begin());
    }
    std::set_union(
        result[parent_index[0]].begin(), result[parent_index[0]].end(),
        result[parent_index[1]].begin(), result[parent_index[1]].end(),
        std::back_inserter(result[index]));
    if (result[index].empty())
      CCTK_ERROR("ParticleAHFinderX produced an empty common-surface "
                 "constituent lineage");
  }
  return result;
}

bool disjoint_lineages(const std::vector<amrex::Long> &first,
                       const std::vector<amrex::Long> &second) {
  auto first_it = first.begin();
  auto second_it = second.begin();
  while (first_it != first.end() && second_it != second.end()) {
    if (*first_it == *second_it)
      return false;
    if (*first_it < *second_it)
      ++first_it;
    else
      ++second_it;
  }
  return true;
}

std::vector<amrex::Long>
merged_lineage(const std::vector<amrex::Long> &first,
               const std::vector<amrex::Long> &second) {
  std::vector<amrex::Long> result;
  result.reserve(first.size() + second.size());
  std::set_union(first.begin(), first.end(), second.begin(), second.end(),
                 std::back_inserter(result));
  return result;
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

amrex::Long duplicate_published_surface(
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
      return surface.stable_id;
  }
  return 0;
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
    if (first.generation == std::numeric_limits<int>::max() ||
        second.generation == std::numeric_limits<int>::max())
      CCTK_ERROR("ParticleAHFinderX dynamic surface generation overflowed");
    seed.generation = amrex::max(first.generation, second.generation) + 1;
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
    record.generation = seed.generation;
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
  for (auto &container : state.containers)
    container->redistribute_global();
  std::vector<amrex::Long> new_surface_ids;
  new_surface_ids.reserve(new_records.size());
  for (const auto &record : new_records)
    new_surface_ids.push_back(record.stable_id);
  const auto owner_stats =
      assign_new_spatial_logical_owners(state, new_surface_ids);
  for (auto &record : new_records) {
    const auto found = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), record.stable_id,
        [](const SurfaceRecord &surface, const amrex::Long stable_id) {
          return surface.stable_id < stable_id;
        });
    if (found == state.surfaces.end() || found->stable_id != record.stable_id)
      CCTK_ERROR("ParticleAHFinderX lost a dynamic surface during spatial "
                 "ownership assignment");
    record = *found;
  }
  state.logical_surfaces->append_surfaces(local_logical_seeds(new_records));
  if (verbose && amrex::ParallelDescriptor::IOProcessor()) {
    const double colocated_fraction =
        owner_stats.particle_count > 0
            ? double(owner_stats.colocated_particles) /
                  double(owner_stats.particle_count)
            : 1.0;
    CCTK_VINFO("Dynamic spatial logical ownership colocates %lld/%lld new "
               "particle(s) (%.2f%%); maximum total owner load=%lld",
               static_cast<long long>(owner_stats.colocated_particles),
               static_cast<long long>(owner_stats.particle_count),
               100.0 * colocated_fraction,
               static_cast<long long>(owner_stats.maximum_owner_load));
  }
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
  for (const auto &record : new_records)
    append_candidate_event(state, record, CandidateEventType::birth,
                           CandidateEventReason::created, iteration, time);
}

} // namespace

void manage_candidates_before_search(Runtime &state, const int iteration,
                                     const amrex::Real time) {
  DECLARE_CCTK_PARAMETERS;
  if (!enable_dynamic_common_horizons && !retire_parents_after_common)
    return;

  std::vector<amrex::Long> retire_ids;
  std::vector<std::pair<amrex::Long, CandidateEventReason>>
      candidate_retirements;
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
    const bool unavailable =
        !first || !second || first->lifecycle == ParticleLifecycle::retired ||
        second->lifecycle == ParticleLifecycle::retired;
    const bool expired =
        !unavailable &&
        separation(*first, *second) > surface.candidate_expiry_distance;
    if (unavailable || expired) {
      surface.next_search_iteration =
          iteration + candidate_retry_max_iterations;
      surface.last_state_change_iteration = iteration;
      retire_ids.push_back(surface.stable_id);
      candidate_retirements.push_back(
          {surface.stable_id,
           unavailable ? CandidateEventReason::parent_unavailable
                       : CandidateEventReason::separation_expired});
    }
  }
  if (!retire_ids.empty()) {
    retire_surface_populations(state, retire_ids);
    for (const auto &retirement : candidate_retirements) {
      const auto *const surface =
          find_surface(state.surfaces, retirement.first);
      if (!surface)
        CCTK_ERROR("ParticleAHFinderX lost retired candidate metadata");
      append_candidate_event(state, *surface,
                             CandidateEventType::retirement,
                             retirement.second, iteration, time);
    }
  }
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
  const auto lineages = constituent_lineages(state.surfaces);
  struct ExistingLineage {
    bool live = false;
    int cooldown_until = -1;
  };
  std::map<std::vector<amrex::Long>, ExistingLineage> existing_lineages;
  for (std::size_t index = 0; index < existing_surface_count; ++index) {
    const auto &surface = state.surfaces[index];
    if (!has_composed_lineage(surface))
      continue;
    auto &status = existing_lineages[lineages[index]];
    if (surface.lifecycle != ParticleLifecycle::retired)
      status.live = true;
    else
      status.cooldown_until =
          amrex::max(status.cooldown_until, surface.next_search_iteration);
  }
  for (std::size_t first_index = 0;
       first_index < existing_surface_count &&
       pairs.size() <
           static_cast<std::size_t>(maximum_new_candidates_per_search) &&
       active_candidates + static_cast<int>(pairs.size()) <
           maximum_active_candidates;
       ++first_index) {
    const auto &first = state.surfaces[first_index];
    if (!is_discovery_parent(first))
      continue;
    for (std::size_t second_index = first_index + 1;
         second_index < existing_surface_count &&
         pairs.size() <
             static_cast<std::size_t>(maximum_new_candidates_per_search) &&
         active_candidates + static_cast<int>(pairs.size()) <
             maximum_active_candidates;
         ++second_index) {
      const auto &second = state.surfaces[second_index];
      if (!is_discovery_parent(second) ||
          !disjoint_lineages(lineages[first_index],
                             lineages[second_index]))
        continue;
      const auto lineage = merged_lineage(lineages[first_index],
                                          lineages[second_index]);
      const auto existing = existing_lineages.find(lineage);
      if (existing != existing_lineages.end() &&
          (existing->second.live ||
           iteration < existing->second.cooldown_until))
        continue;
      const amrex::Real trigger_distance =
          candidate_trigger_factor *
          (bounding_radius(first) + bounding_radius(second));
      if (separation(first, second) <= trigger_distance) {
        pairs.push_back({first_index, second_index});
        existing_lineages[lineage].live = true;
      }
    }
  }
  birth_candidates(state, pairs, iteration, time);
}

void finalize_candidates_after_search(Runtime &state, const int iteration,
                                      const amrex::Real time) {
  DECLARE_CCTK_PARAMETERS;
  std::vector<SurfaceStateUpdate> promotions;
  std::vector<amrex::Long> promoted_ids;
  std::vector<SurfaceSphereReset> resets;
  std::vector<amrex::Long> retire_ids;

  for (auto &candidate : state.surfaces) {
    if (candidate.lifecycle != ParticleLifecycle::candidate ||
        candidate.role != SurfaceRole::common ||
        candidate.last_search_iteration != iteration)
      continue;
    ++candidate.candidate_attempts;
    CandidateEventReason attempt_reason =
        CandidateEventReason::solver_not_converged;
    amrex::Long duplicate_id = 0;
    bool promote = false;
    if (candidate.invalid_points != 0 || !std::isfinite(candidate.area) ||
        !std::isfinite(candidate.expansion_l2) ||
        !std::isfinite(candidate.expansion_linf)) {
      attempt_reason = CandidateEventReason::invalid_geometry;
    } else if (candidate.relaxation_state !=
               SurfaceRelaxationState::converged) {
      attempt_reason = CandidateEventReason::solver_not_converged;
    } else if (!(candidate.area > 0)) {
      attempt_reason = CandidateEventReason::invalid_geometry;
    } else if (!candidate_contains_parents(
                   candidate, state.surfaces,
                   candidate_containment_tolerance)) {
      attempt_reason = CandidateEventReason::parent_containment_failed;
    } else {
      duplicate_id = duplicate_published_surface(
          candidate, state.surfaces,
          candidate_deduplication_center_tolerance,
          candidate_deduplication_area_tolerance);
      if (duplicate_id != 0)
        attempt_reason = CandidateEventReason::duplicate_published_surface;
      else {
        attempt_reason = CandidateEventReason::accepted;
        promote = true;
      }
    }
    if (promote) {
      candidate.published = true;
      candidate.promoted_iteration = iteration;
      candidate.last_state_change_iteration = iteration;
      candidate.compatibility_slot = next_compatibility_slot(state.surfaces);
      promotions.push_back({candidate.stable_id, ParticleLifecycle::active,
                            SurfaceRole::common});
      promoted_ids.push_back(candidate.stable_id);
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
      append_candidate_event(state, candidate, CandidateEventType::attempt,
                             attempt_reason, iteration, time,
                             duplicate_id);
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
    append_candidate_event(state, candidate, CandidateEventType::attempt,
                           attempt_reason, iteration, time, duplicate_id);
    const amrex::Real next_radius =
        amrex::min(candidate.candidate_radius_ceiling,
                   candidate.radius * candidate_radius_growth_factor);
    resets.push_back({candidate.stable_id, candidate.center, next_radius});
  }

  if (!promotions.empty()) {
    update_surface_states(state, promotions);
    for (const auto stable_id : promoted_ids) {
      const auto *const surface = find_surface(state.surfaces, stable_id);
      if (!surface)
        CCTK_ERROR("ParticleAHFinderX lost promoted candidate metadata");
      append_candidate_event(state, *surface,
                             CandidateEventType::promotion,
                             CandidateEventReason::accepted, iteration,
                             time);
    }
  }
  if (!resets.empty())
    reset_surface_spheres(state, resets);
  if (!retire_ids.empty()) {
    retire_surface_populations(state, retire_ids);
    for (const auto stable_id : retire_ids) {
      const auto *const surface = find_surface(state.surfaces, stable_id);
      if (!surface)
        CCTK_ERROR("ParticleAHFinderX lost attempt-limited candidate "
                   "metadata");
      append_candidate_event(state, *surface,
                             CandidateEventType::retirement,
                             CandidateEventReason::attempt_limit, iteration,
                             time);
    }
  }
}

} // namespace ParticleAHFinderX
