/**
 * \file surface_router.hxx
 * \brief GPU-aware mesh-owner to logical-owner surface communication.
 */
#ifndef PARTICLEAHFINDERX_SURFACE_ROUTER_HXX
#define PARTICLEAHFINDERX_SURFACE_ROUTER_HXX

#include "logical_surface_storage.hxx"
#include "surface_particle_container.hxx"

#include <AMReX_GpuContainers.H>

#include <memory>
#include <vector>

namespace ParticleAHFinderX {

struct LogicalSourceRecord {
  amrex::Long logical_index;
  amrex::Long particle_id;
  amrex::Long surface_id;
  amrex::Long source_index;
  int source_rank;
  int generation;
  int angular_level;
  int angular_levels;
  int itheta;
  int iphi;
};

struct LogicalSampleRecord {
  amrex::Long logical_index;
  amrex::Long particle_id;
  amrex::Long surface_id;
  amrex::Long source_index;
  int source_rank;
  int generation;
  int angular_level;
  int angular_levels;
  int itheta;
  int iphi;
  int sampled_level;
  int sampled_level_changed;
  amrex::Real position[3];
  amrex::Real height;
  amrex::Real relaxation_velocity;
  amrex::Real expansion;
  // gamma[6], d_gamma[3][6], curv[6]
  amrex::Real adm[30];
};

struct SurfaceRoutingStats {
  amrex::Long sent = 0;
  amrex::Long received = 0;
  amrex::Long missing = 0;
  amrex::Long duplicate = 0;
  amrex::Long invalid = 0;
};

enum class LogicalRoutingMode : int {
  sampled_fields,
  source_metadata,
};

struct MeshUpdateRecord {
  amrex::Long source_index;
  amrex::Long particle_id;
  amrex::Real position[3];
  amrex::Real height;
  amrex::Real relaxation_velocity;
  amrex::Real expansion;
  amrex::Real normal[3];
  amrex::Real area_weight;
};

/**
 * Pack active mesh particles on device, exchange directly from accelerator
 * memory, and scatter one record per immutable logical index. The
 * source_metadata mode refreshes reverse-routing addresses after AMReX has
 * reordered particles without overwriting the logical surface state.
 */
SurfaceRoutingStats route_particles_to_logical_owners(
    const std::vector<std::unique_ptr<SurfaceParticleContainer>> &containers,
    LogicalSurfaceStorage &logical_surfaces, LogicalRoutingMode mode);

/** Return accepted logical h/v/position records to their current mesh owner. */
SurfaceRoutingStats route_updates_to_mesh_owners(
    LogicalSurfaceStorage &logical_surfaces,
    const std::vector<std::unique_ptr<SurfaceParticleContainer>> &containers);

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_SURFACE_ROUTER_HXX
