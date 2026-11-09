/**
 * \file candidate_events.hxx
 * \brief Bounded audit records for dynamic common-surface decisions.
 */
#ifndef PARTICLEAHFINDERX_CANDIDATE_EVENTS_HXX
#define PARTICLEAHFINDERX_CANDIDATE_EVENTS_HXX

#include "runtime.hxx"

namespace ParticleAHFinderX {

/**
 * Snapshot one candidate decision into the replicated bounded event queue.
 *
 * Parent separation is evaluated from the current tracked parent centers.
 * `related_surface_id` identifies the published surface that caused a
 * duplicate rejection, and is zero for all other decisions.
 */
void append_candidate_event(Runtime &state, const SurfaceRecord &surface,
                            CandidateEventType type,
                            CandidateEventReason reason, int iteration,
                            amrex::Real time,
                            amrex::Long related_surface_id = 0);

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_CANDIDATE_EVENTS_HXX
