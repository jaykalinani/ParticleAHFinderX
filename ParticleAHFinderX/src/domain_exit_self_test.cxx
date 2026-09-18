/**
 * \file domain_exit_self_test.cxx
 * \brief Validation-only exercise of explicit domain-exit retirement.
 */
#include "runtime.hxx"
#include "surface_manager.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_ParallelDescriptor.H>

#include <vector>

namespace ParticleAHFinderX {

extern "C" void ParticleAHFinderX_DomainExitSelfTest(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  if (!run_domain_exit_self_test)
    return;
  auto &state = runtime();
  if (!retire_on_domain_exit || domain_exit_grace_searches != 1 ||
      state.surfaces.size() != 1 || state.containers.empty() ||
      !state.logical_surfaces || !state.initial_surfaces_created)
    CCTK_ERROR("ParticleAHFinderX domain-exit self-test requires one "
               "initialized surface, retire_on_domain_exit=yes, and "
               "domain_exit_grace_searches=1");

  auto &surface = state.surfaces.front();
  const PhysicalDomainBounds domain =
      state.containers.front()->physical_domain_bounds();
  if (!surface_fits_physical_domain(surface, surface.center, domain))
    CCTK_ERROR("ParticleAHFinderX domain-exit self-test surface does not "
               "initially fit the physical domain");

  int exit_direction = -1;
  for (int d = 0; d < 3; ++d)
    if (!domain.periodic[d]) {
      exit_direction = d;
      break;
    }
  if (exit_direction < 0)
    CCTK_ERROR("ParticleAHFinderX domain-exit self-test requires one "
               "non-periodic direction");

  auto outside_center = surface.center;
  outside_center[exit_direction] =
      domain.upper[exit_direction] + 2 * surface.radius;
  if (surface_fits_physical_domain(surface, outside_center, domain))
    CCTK_ERROR("ParticleAHFinderX domain-exit self-test accepted an outside "
               "non-periodic radial bound");
  auto periodic_domain = domain;
  periodic_domain.periodic[exit_direction] = 1;
  if (!surface_fits_physical_domain(surface, outside_center, periodic_domain))
    CCTK_ERROR("ParticleAHFinderX domain-exit self-test rejected motion "
               "through a periodic direction");

  const amrex::Long next_surface_id = state.next_surface_id;
  const amrex::Long next_particle_id = state.next_particle_id;
  const amrex::Long creation_events = state.population_creation_events;
  surface.manual_center = outside_center;
  surface.center_policy = CenterPolicy::manual;
  std::vector<amrex::Long> due_surface_ids{surface.stable_id};
  predict_surface_centers(cctkGH, state, due_surface_ids, cctk_iteration,
                          static_cast<amrex::Real>(cctk_time),
                          maximum_center_extrapolation_intervals);

  amrex::Long particle_count = 0;
  for (const auto &container : state.containers)
    particle_count += container->local_particle_count();
  amrex::ParallelDescriptor::ReduceLongSum(particle_count);
  amrex::Long logical_count = state.logical_surfaces->local_point_count();
  amrex::ParallelDescriptor::ReduceLongSum(logical_count);
  if (!due_surface_ids.empty() ||
      surface.lifecycle != ParticleLifecycle::retired ||
      !surface.retired_outside_domain ||
      surface.consecutive_domain_exit_checks != 1 || particle_count != 0 ||
      logical_count != 0 || state.global_particle_count != 0 ||
      state.next_surface_id != next_surface_id ||
      state.next_particle_id != next_particle_id ||
      state.population_creation_events != creation_events)
    CCTK_ERROR("ParticleAHFinderX domain-exit self-test did not explicitly "
               "retire exactly one persistent population");

  CCTK_INFO("Domain-exit retirement self-test passed on the active AMReX "
            "backend: persistent particle and logical counts are zero");
}

} // namespace ParticleAHFinderX
