/**
 * \file surface_manager.hxx
 * \brief Surface ownership and device-seed construction utilities.
 */
#ifndef PARTICLEAHFINDERX_SURFACE_MANAGER_HXX
#define PARTICLEAHFINDERX_SURFACE_MANAGER_HXX

#include "runtime.hxx"

#include <cctk.h>

#include <vector>

namespace ParticleAHFinderX {

/// Initialize the compatibility one-layer representation of a new surface.
void initialize_single_angular_layer(SurfaceRecord &surface);

/// Allocate a factor-two persistent hierarchy ending at the target layer.
void initialize_angular_layers(SurfaceRecord &surface, bool enable,
                               int minimum_ntheta, int maximum_levels);

/// Total persistent population across every angular layer of one surface.
amrex::Long total_angular_particle_count(const SurfaceRecord &surface);

/// Construct particle birth descriptors for every live angular layer.
std::vector<SurfaceSeed>
particle_seeds(const std::vector<SurfaceRecord> &surfaces);

/// Balance complete logical surfaces across the currently available ranks.
void assign_logical_owners(std::vector<SurfaceRecord> &surfaces);

/// Append one birth without moving any existing logical surface owner.
void assign_new_logical_owner(SurfaceRecord &surface,
                              const std::vector<SurfaceRecord> &surfaces);

/// Close owner-local offset gaps after explicit retirement.
void rebuild_logical_offsets(std::vector<SurfaceRecord> &surfaces);

/// Remove explicitly retired populations and compact owner-GPU storage.
void retire_surface_populations(Runtime &state,
                                const std::vector<amrex::Long> &stable_ids);

/// Publish bounded lifecycle/role changes to both persistent device views.
void update_surface_states(Runtime &state,
                           const std::vector<SurfaceStateUpdate> &updates);

/// Replace failed candidate shapes with a new spherical radius bracket.
void reset_surface_spheres(Runtime &state,
                           const std::vector<SurfaceSphereReset> &resets);

/// Construct owner-local logical storage descriptors for this rank.
std::vector<LogicalSurfaceSeed>
local_logical_seeds(const std::vector<SurfaceRecord> &surfaces);

/// Construct the decomposition metadata written into persistent particles.
std::vector<LogicalOwnership>
logical_ownership(const std::vector<SurfaceRecord> &surfaces);

/** Gather one bounded owner record per surface and update replicated metadata. */
void update_surface_diagnostics(LogicalSurfaceStorage &logical_surfaces,
                                std::vector<SurfaceRecord> &surfaces,
                                int iteration, amrex::Real time,
                                int relaxation_steps);

/**
 * Preflight and move due persistent seeds according to their center policy.
 *
 * A due ID is removed from the vector while its proposed surface does not fit
 * inside a non-periodic physical domain. Repeated domain-exit checks retire
 * the population through the explicit persistent lifecycle path.
 */
void predict_surface_centers(const cGH *cctkGH, Runtime &state,
                             std::vector<amrex::Long> &due_surface_ids,
                             int iteration, amrex::Real time,
                             amrex::Real maximum_extrapolation_intervals);

/// Conservative radial bound used by domain-exit preflight and its self-test.
bool surface_fits_physical_domain(const SurfaceRecord &surface,
                                  const std::array<amrex::Real, 3> &center,
                                  const PhysicalDomainBounds &domain);

/**
 * Commit successful-search center/radius history and its external-center
 * anchor for future displacement-based seeds.
 */
void update_tracking_history(const cGH *cctkGH, Runtime &state, int iteration,
                             amrex::Real time);

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_SURFACE_MANAGER_HXX
