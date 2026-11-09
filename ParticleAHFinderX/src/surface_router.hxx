/**
 * \file surface_router.hxx
 * \brief GPU-aware mesh-owner to logical-owner surface communication.
 */
#ifndef PARTICLEAHFINDERX_SURFACE_ROUTER_HXX
#define PARTICLEAHFINDERX_SURFACE_ROUTER_HXX

#include "logical_surface_storage.hxx"
#include "surface_particle_container.hxx"

#include <AMReX_GpuContainers.H>

#include <mpi.h>

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

/** Optional twelve-real curvature payload; higher derivatives stay local. */
struct LogicalWeylRecord {
  amrex::Long logical_index;
  amrex::Long particle_id;
  amrex::Long surface_id;
  amrex::Real electric[6];
  amrex::Real magnetic[6];
  int status;
};

/** Optional gauge payload; values stay in device buffers during routing. */
struct LogicalGaugeRecord {
  amrex::Long logical_index;
  amrex::Long particle_id;
  amrex::Long surface_id;
  amrex::Real value[FourGaugeLayout::component_count];
  int status;
};

/** Optional covariant stress-energy payload kept in device buffers. */
struct LogicalMatterRecord {
  amrex::Long logical_index;
  amrex::Long particle_id;
  amrex::Long surface_id;
  amrex::Real value[FourMatterLayout::component_count];
  int status;
};

struct SurfaceRoutingStats {
  amrex::Long sent = 0;
  amrex::Long received = 0;
  amrex::Long missing = 0;
  amrex::Long duplicate = 0;
  amrex::Long invalid = 0;
  amrex::Long outside_active_band = 0;
  // Payload records remain in accelerator memory. These counters distinguish
  // that point-data path from the compact rank-count/error control plane that
  // topology discovery currently maintains on the host.
  amrex::Long device_payload_bytes_sent = 0;
  amrex::Long device_payload_bytes_received = 0;
  amrex::Long device_local_copy_bytes = 0;
  amrex::Long host_point_data_bytes = 0;
  amrex::Long control_device_to_host_bytes = 0;
  amrex::Long control_host_to_device_bytes = 0;
  amrex::Long mpi_control_bytes_sent = 0;
  amrex::Long mpi_control_bytes_received = 0;
  amrex::Long device_to_host_control_copies = 0;
  amrex::Long host_to_device_control_copies = 0;
  amrex::Long mpi_count_collectives = 0;
  amrex::Long mpi_payload_exchange_rounds = 0;
  amrex::Long mpi_payload_messages_sent = 0;
  amrex::Long mpi_payload_messages_received = 0;
  amrex::Long explicit_device_synchronizations = 0;
  amrex::Long forward_topology_reuses = 0;
  amrex::Long forward_topology_rebuilds = 0;
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
 * Transient transpose of one completed mesh-to-logical route.
 *
 * The received forward records are grouped by source mesh rank. Their device
 * receive slots therefore provide the exact packing order for the immediate
 * logical-to-mesh return. The plan remains valid until the next forward route;
 * AMReX redistribution may then have changed particle rank ownership.
 */
struct SurfaceRoutePlan {
  std::vector<int> return_send_counts;
  std::vector<int> return_send_offsets;
  std::vector<int> return_receive_counts;
  std::vector<int> return_receive_offsets;
  // Search-lifetime device workspaces. Resizing after the first route only
  // occurs when the active population grows; ordinary pseudo-steps reuse the
  // existing accelerator allocations.
  amrex::Gpu::DeviceVector<int> rank_counts;
  amrex::Gpu::DeviceVector<int> rank_offsets;
  amrex::Gpu::DeviceVector<int> rank_cursors;
  amrex::Gpu::DeviceVector<int> forward_status;
  amrex::Gpu::DeviceVector<amrex::Long> return_slot;
  amrex::Gpu::DeviceVector<LogicalSampleRecord> sample_send_records;
  amrex::Gpu::DeviceVector<LogicalSampleRecord> sample_receive_records;
  amrex::Gpu::DeviceVector<LogicalSourceRecord> source_send_records;
  amrex::Gpu::DeviceVector<LogicalSourceRecord> source_receive_records;
  amrex::Gpu::DeviceVector<LogicalWeylRecord> weyl_send_records;
  amrex::Gpu::DeviceVector<LogicalWeylRecord> weyl_receive_records;
  amrex::Gpu::DeviceVector<LogicalGaugeRecord> gauge_send_records;
  amrex::Gpu::DeviceVector<LogicalGaugeRecord> gauge_receive_records;
  amrex::Gpu::DeviceVector<LogicalMatterRecord> matter_send_records;
  amrex::Gpu::DeviceVector<LogicalMatterRecord> matter_receive_records;
  amrex::Gpu::DeviceVector<MeshUpdateRecord> update_send_records;
  amrex::Gpu::DeviceVector<MeshUpdateRecord> update_receive_records;
  amrex::Gpu::DeviceVector<MeshUpdateRecord> ordered_updates;
  amrex::Gpu::DeviceVector<int> update_arrival;
  amrex::Gpu::DeviceVector<int> update_expected;
  amrex::Gpu::DeviceVector<int> update_status;
  std::vector<MPI_Request> mpi_requests;
  amrex::Long logical_point_count = 0;
  amrex::Long active_point_count = 0;
  int ranks = 0;
  // Recomputed collectively whenever the topology is rebuilt. If every
  // active point is already on its logical owner's MPI rank, both directions
  // can use device-to-device copies until the topology is invalidated.
  bool payload_uses_mpi = true;
  bool valid = false;
};

enum class LogicalRoutingMode : int {
  sampled_fields,
  source_metadata,
};

/**
 * Pack active mesh particles on device, exchange directly from accelerator
 * memory, and scatter one record per immutable logical index. The
 * source_metadata mode refreshes reverse-routing addresses after AMReX has
 * reordered particles without overwriting the logical surface state.
 */
SurfaceRoutingStats route_particles_to_logical_owners(
    const std::vector<std::unique_ptr<SurfaceParticleContainer>> &containers,
    LogicalSurfaceStorage &logical_surfaces, LogicalRoutingMode mode,
    SurfaceRoutePlan &return_plan, amrex::Long active_point_count);

/** Route optional mesh-owner E/B records through the final sample topology. */
SurfaceRoutingStats route_weyl_to_logical_owners(
    const MeshWeylPoint *mesh_points,
    LogicalSurfaceStorage &logical_surfaces,
    SurfaceRoutePlan &route_plan, amrex::Long active_point_count,
    amrex::Long mesh_point_count);

/** Route optional mesh-owner gauge records through the final sample topology. */
SurfaceRoutingStats route_gauge_to_logical_owners(
    const MeshGaugePoint *mesh_points,
    LogicalSurfaceStorage &logical_surfaces,
    SurfaceRoutePlan &route_plan, amrex::Long active_point_count,
    amrex::Long mesh_point_count);

/** Route optional covariant stress-energy through the final sample topology. */
SurfaceRoutingStats route_matter_to_logical_owners(
    const MeshMatterPoint *mesh_points,
    LogicalSurfaceStorage &logical_surfaces,
    SurfaceRoutePlan &route_plan, amrex::Long active_point_count,
    amrex::Long mesh_point_count);

/** Return accepted logical h/v/position records to their current mesh owner. */
SurfaceRoutingStats route_updates_to_mesh_owners(
    LogicalSurfaceStorage &logical_surfaces,
    const std::vector<std::unique_ptr<SurfaceParticleContainer>> &containers,
    SurfaceRoutePlan &return_plan,
    int minimum_active_level = -1, int maximum_active_level = -1);

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_SURFACE_ROUTER_HXX
