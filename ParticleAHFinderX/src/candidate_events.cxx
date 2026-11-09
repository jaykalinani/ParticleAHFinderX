/**
 * \file candidate_events.cxx
 * \brief Bounded audit records for dynamic common-surface decisions.
 */
#include "candidate_events.hxx"

#include "surface_manager.hxx"

#include <algorithm>
#include <cmath>

namespace ParticleAHFinderX {
namespace {

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

std::array<amrex::Real, 3>
tracked_center(const SurfaceRecord &surface) {
  return surface.last_success_iteration >= 0 ? surface.centroid
                                              : surface.center;
}

amrex::Real parent_separation(const Runtime &state,
                              const SurfaceRecord &surface) {
  const auto *const first = find_surface(state.surfaces,
                                         surface.parent_ids[0]);
  const auto *const second = find_surface(state.surfaces,
                                          surface.parent_ids[1]);
  if (!first || !second)
    return -1;
  const auto first_center = tracked_center(*first);
  const auto second_center = tracked_center(*second);
  amrex::Real distance_squared = 0;
  for (int d = 0; d < 3; ++d) {
    const amrex::Real difference = first_center[d] - second_center[d];
    distance_squared += difference * difference;
  }
  if (!std::isfinite(distance_squared))
    return -1;
  return std::sqrt(distance_squared);
}

} // namespace

void append_candidate_event(Runtime &state, const SurfaceRecord &surface,
                            const CandidateEventType type,
                            const CandidateEventReason reason,
                            const int iteration, const amrex::Real time,
                            const amrex::Long related_surface_id) {
  if (surface.stable_id <= 0 || surface.role != SurfaceRole::common ||
      surface.generation <= 0 || surface.parent_ids[0] <= 0 ||
      surface.parent_ids[1] <= 0 || iteration < 0 ||
      !std::isfinite(time) || related_surface_id < 0 ||
      (reason == CandidateEventReason::duplicate_published_surface) !=
          (related_surface_id > 0) ||
      related_surface_id == surface.stable_id)
    amrex::Abort("ParticleAHFinderX cannot record an invalid candidate "
                 "event");
  for (const auto coordinate : surface.center)
    if (!std::isfinite(coordinate))
      amrex::Abort("ParticleAHFinderX candidate event has a non-finite "
                   "center");
  if (!std::isfinite(surface.candidate_trigger_distance) ||
      !std::isfinite(surface.candidate_expiry_distance) ||
      !std::isfinite(surface.radius) ||
      !(surface.candidate_trigger_distance > 0) ||
      !(surface.candidate_expiry_distance >
        surface.candidate_trigger_distance) ||
      !(surface.radius > 0))
    amrex::Abort("ParticleAHFinderX candidate event has invalid encounter "
                 "bounds");
  const auto duplicate = std::find_if(
      state.pending_candidate_events.begin(),
      state.pending_candidate_events.end(), [&](const CandidateEvent &event) {
        return event.iteration == iteration &&
               event.stable_id == surface.stable_id && event.type == type;
      });
  if (duplicate != state.pending_candidate_events.end())
    amrex::Abort("ParticleAHFinderX attempted to record a duplicate candidate "
                 "event");

  CandidateEvent event;
  event.iteration = iteration;
  event.time = time;
  event.stable_id = surface.stable_id;
  event.parent_ids = surface.parent_ids;
  event.related_surface_id = related_surface_id;
  event.generation = surface.generation;
  event.type = type;
  event.reason = reason;
  event.attempt = surface.candidate_attempts;
  event.center = surface.center;
  event.parent_separation = parent_separation(state, surface);
  event.trigger_distance = surface.candidate_trigger_distance;
  event.expiry_distance = surface.candidate_expiry_distance;
  event.radius = surface.radius;
  const bool valid_radius_range =
      std::isfinite(surface.minimum_radius) &&
      std::isfinite(surface.maximum_radius) && surface.minimum_radius >= 0 &&
      surface.maximum_radius >= surface.minimum_radius;
  event.minimum_radius = valid_radius_range ? surface.minimum_radius : 0;
  event.maximum_radius = valid_radius_range ? surface.maximum_radius : 0;
  event.area = std::isfinite(surface.area) && surface.area >= 0
                   ? surface.area
                   : 0;
  event.expansion_l2 =
      std::isfinite(surface.expansion_l2) && surface.expansion_l2 >= 0
          ? surface.expansion_l2
          : 0;
  event.expansion_linf =
      std::isfinite(surface.expansion_linf) && surface.expansion_linf >= 0
          ? surface.expansion_linf
          : 0;
  event.invalid_points = surface.invalid_points;
  event.relaxation_steps = surface.last_search_steps;
  event.next_search_iteration = surface.next_search_iteration;
  event.particle_count = total_angular_particle_count(surface);
  state.pending_candidate_events.push_back(event);
}

} // namespace ParticleAHFinderX
