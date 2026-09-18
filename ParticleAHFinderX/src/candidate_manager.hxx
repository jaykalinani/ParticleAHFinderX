/**
 * \file candidate_manager.hxx
 * \brief Conservative lifecycle management for dynamic common surfaces.
 */
#ifndef PARTICLEAHFINDERX_CANDIDATE_MANAGER_HXX
#define PARTICLEAHFINDERX_CANDIDATE_MANAGER_HXX

#include "runtime.hxx"

namespace ParticleAHFinderX {

/// Expire stale candidates, honor grace retirement, and batch new births.
void manage_candidates_before_search(Runtime &state, int iteration,
                                     amrex::Real time);

/// Promote independently validated candidates or schedule bounded retries.
void finalize_candidates_after_search(Runtime &state, int iteration,
                                      amrex::Real time);

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_CANDIDATE_MANAGER_HXX
