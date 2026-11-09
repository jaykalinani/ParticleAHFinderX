/**
 * \file surface_router.cxx
 * \brief Portable GPU-aware routing between mesh and angular ownership.
 */
#include "surface_router.hxx"

#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <mpi.h>

#include <climits>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ParticleAHFinderX {
namespace {

std::vector<int> exclusive_offsets(const std::vector<int> &counts) {
  std::vector<int> offsets(counts.size(), 0);
  for (std::size_t rank = 1; rank < counts.size(); ++rank) {
    if (offsets[rank - 1] > INT_MAX - counts[rank - 1])
      throw std::runtime_error(
          "ParticleAHFinderX routing record count overflowed");
    offsets[rank] = offsets[rank - 1] + counts[rank - 1];
  }
  return offsets;
}

int total_count(const std::vector<int> &counts,
                const std::vector<int> &offsets) {
  if (counts.empty())
    return 0;
  if (offsets.back() > INT_MAX - counts.back())
    throw std::runtime_error(
        "ParticleAHFinderX routing record count overflowed");
  return offsets.back() + counts.back();
}

int remote_peer_count(const std::vector<int> &counts, const int rank) {
  int peers = 0;
  for (int peer = 0; peer < static_cast<int>(counts.size()); ++peer)
    peers += peer != rank && counts[peer] > 0;
  return peers;
}

void check_mpi(const int error, const char *const operation) {
  if (error != MPI_SUCCESS)
    throw std::runtime_error(std::string("ParticleAHFinderX ") + operation +
                             " failed");
}

template <typename Record>
amrex::Long record_bytes(const int count) {
  if (count < 0 ||
      static_cast<unsigned long long>(count) >
          static_cast<unsigned long long>(
              std::numeric_limits<amrex::Long>::max()) /
              sizeof(Record))
    throw std::runtime_error(
        "ParticleAHFinderX routing byte counter overflowed");
  return static_cast<amrex::Long>(count) * sizeof(Record);
}

template <typename Record>
int mpi_record_bytes(const int count) {
  constexpr std::size_t record_size = sizeof(Record);
  if (count < 0 || record_size > static_cast<std::size_t>(INT_MAX) ||
      static_cast<std::size_t>(count) >
          static_cast<std::size_t>(INT_MAX) / record_size)
    throw std::runtime_error(
        "ParticleAHFinderX routing exceeds the MPI byte-count limit");
  return count * static_cast<int>(record_size);
}

template <typename Record>
SurfaceRoutingStats routing_performance_stats(
    const int ranks, const int send_total, const int receive_total,
    const int local_send_count, const int local_receive_count,
    const int remote_send_peers, const int remote_receive_peers,
    const amrex::Long device_to_host_bytes,
    const amrex::Long device_to_host_copies,
    const amrex::Long host_to_device_bytes,
    const amrex::Long host_to_device_copies,
    const bool exchanged_counts, const bool payload_uses_mpi) {
  SurfaceRoutingStats stats;
  if (local_send_count < 0 || local_send_count > send_total ||
      local_receive_count < 0 || local_receive_count > receive_total ||
      remote_send_peers < 0 || remote_receive_peers < 0)
    throw std::runtime_error(
        "ParticleAHFinderX routing has invalid process-local counts");
  const amrex::Long local_bytes = record_bytes<Record>(local_send_count);
  const amrex::Long remote_sent_bytes =
      record_bytes<Record>(send_total - local_send_count);
  const amrex::Long remote_received_bytes =
      record_bytes<Record>(receive_total - local_receive_count);
#ifdef AMREX_USE_GPU
  stats.control_device_to_host_bytes = device_to_host_bytes;
  stats.control_host_to_device_bytes = host_to_device_bytes;
  stats.device_to_host_control_copies = device_to_host_copies;
  stats.host_to_device_control_copies = host_to_device_copies;
#endif
  if (exchanged_counts) {
    stats.mpi_control_bytes_sent =
        (static_cast<amrex::Long>(ranks) + (ranks > 1)) * sizeof(int);
    stats.mpi_control_bytes_received =
        (static_cast<amrex::Long>(ranks) + (ranks > 1)) * sizeof(int);
    // The topology rebuild exchanges per-rank counts and, for a multi-rank
    // run, collectively determines whether every payload route is local.
    stats.mpi_count_collectives = 1 + (ranks > 1);
  }
  stats.device_local_copy_bytes = local_bytes;
  if (payload_uses_mpi) {
    stats.device_payload_bytes_sent = remote_sent_bytes;
    stats.device_payload_bytes_received = remote_received_bytes;
    stats.mpi_payload_exchange_rounds = 1;
    stats.mpi_payload_messages_sent = remote_send_peers;
    stats.mpi_payload_messages_received = remote_receive_peers;
    stats.explicit_device_synchronizations = 1;
  }
  return stats;
}

template <typename Record>
void exchange_payload_records(
    amrex::Gpu::DeviceVector<Record> &send_records,
    amrex::Gpu::DeviceVector<Record> &receive_records,
    const std::vector<int> &send_counts,
    const std::vector<int> &send_offsets,
    const std::vector<int> &receive_counts,
    const std::vector<int> &receive_offsets,
    SurfaceRoutePlan &route_plan, const char *const operation) {
  const int rank = amrex::ParallelDescriptor::MyProc();
  const int ranks = amrex::ParallelDescriptor::NProcs();
  if (send_counts.size() != static_cast<std::size_t>(ranks) ||
      send_offsets.size() != static_cast<std::size_t>(ranks) ||
      receive_counts.size() != static_cast<std::size_t>(ranks) ||
      receive_offsets.size() != static_cast<std::size_t>(ranks))
    throw std::runtime_error(
        "ParticleAHFinderX payload exchange has an invalid rank layout");
  const int local_send = send_counts[rank];
  const int local_receive = receive_counts[rank];
  if (local_send != local_receive)
    throw std::runtime_error(
        "ParticleAHFinderX process-local route is not its own transpose");
  if (local_send > 0)
    amrex::Gpu::copy(
        amrex::Gpu::deviceToDevice,
        send_records.begin() + send_offsets[rank],
        send_records.begin() + send_offsets[rank] + local_send,
        receive_records.begin() + receive_offsets[rank]);
  if (!route_plan.payload_uses_mpi)
    return;

  // Point-to-point exchanges avoid forcing ranks with only local surfaces
  // through an all-to-all payload collective. The compact count exchange at
  // topology rebuild time proves that every nonzero receive has one matching
  // send. Payload buffers remain accelerator resident.
  amrex::Gpu::streamSynchronize();
  const int tag = amrex::ParallelDescriptor::SeqNum();
  route_plan.mpi_requests.clear();
  route_plan.mpi_requests.reserve(2 * static_cast<std::size_t>(ranks - 1));
  auto *const send_data = reinterpret_cast<unsigned char *>(
      send_records.dataPtr());
  auto *const receive_data = reinterpret_cast<unsigned char *>(
      receive_records.dataPtr());
  for (int peer = 0; peer < ranks; ++peer) {
    if (peer == rank || receive_counts[peer] == 0)
      continue;
    const int byte_count = mpi_record_bytes<Record>(receive_counts[peer]);
    route_plan.mpi_requests.push_back(MPI_REQUEST_NULL);
    check_mpi(
        MPI_Irecv(receive_data +
                      static_cast<std::size_t>(receive_offsets[peer]) *
                          sizeof(Record),
                  byte_count, MPI_BYTE, peer, tag,
                  amrex::ParallelDescriptor::Communicator(),
                  &route_plan.mpi_requests.back()),
        operation);
  }
  for (int peer = 0; peer < ranks; ++peer) {
    if (peer == rank || send_counts[peer] == 0)
      continue;
    const int byte_count = mpi_record_bytes<Record>(send_counts[peer]);
    route_plan.mpi_requests.push_back(MPI_REQUEST_NULL);
    check_mpi(
        MPI_Isend(send_data +
                      static_cast<std::size_t>(send_offsets[peer]) *
                          sizeof(Record),
                  byte_count, MPI_BYTE, peer, tag,
                  amrex::ParallelDescriptor::Communicator(),
                  &route_plan.mpi_requests.back()),
        operation);
  }
  if (!route_plan.mpi_requests.empty())
    check_mpi(MPI_Waitall(
                  static_cast<int>(route_plan.mpi_requests.size()),
                  route_plan.mpi_requests.data(), MPI_STATUSES_IGNORE),
              operation);
}

} // namespace

namespace {

template <bool transfer_sampled_fields>
SurfaceRoutingStats route_particles_to_logical_owners_impl(
    const std::vector<std::unique_ptr<SurfaceParticleContainer>> &containers,
    LogicalSurfaceStorage &logical_surfaces,
    SurfaceRoutePlan &return_plan, const amrex::Long active_point_count) {
  using RoutingRecord =
      std::conditional_t<transfer_sampled_fields, LogicalSampleRecord,
                         LogicalSourceRecord>;
  const int rank = amrex::ParallelDescriptor::MyProc();
  const int ranks = amrex::ParallelDescriptor::NProcs();
  const auto logical = logical_surfaces.device_view();
  const bool reuse_topology =
      return_plan.valid && return_plan.ranks == ranks &&
      return_plan.logical_point_count == logical.point_count &&
      return_plan.active_point_count == active_point_count &&
      return_plan.return_send_counts.size() ==
          static_cast<std::size_t>(ranks) &&
      return_plan.return_send_offsets.size() ==
          static_cast<std::size_t>(ranks) &&
      return_plan.return_receive_counts.size() ==
          static_cast<std::size_t>(ranks) &&
      return_plan.return_receive_offsets.size() ==
          static_cast<std::size_t>(ranks) &&
      return_plan.rank_offsets.size() == static_cast<std::size_t>(ranks);
  return_plan.valid = false;
  amrex::Long local_population = 0;
  for (const auto &container : containers)
    local_population += container->local_particle_count();
  if (local_population > INT_MAX)
    throw std::runtime_error(
        "ParticleAHFinderX routing currently requires at most INT_MAX local "
        "surface particles");

  return_plan.rank_counts.resize(ranks);
  return_plan.forward_status.resize(1);
  int *const count_by_rank = return_plan.rank_counts.dataPtr();
  int *const routing_error = return_plan.forward_status.dataPtr();
  amrex::ParallelFor(
      ranks, [=] AMREX_GPU_DEVICE(const int destination) noexcept {
        count_by_rank[destination] = 0;
      });
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(const int) noexcept {
    routing_error[0] = 0;
  });

  for (const auto &container : containers)
    for (int lev = 0; lev <= container->finestLevel(); ++lev)
      for (ParIter pti(*container, lev); pti.isValid(); ++pti) {
        const int count = pti.numParticles();
        if (count == 0)
          continue;
        const auto particles =
            pti.GetParticleTile().getConstParticleTileData();
        amrex::ParallelFor(count,
                           [=] AMREX_GPU_DEVICE(const int i) noexcept {
          if (!particles.id(i).is_valid() ||
              particles.idata(IntIdx::solver_active)[i] == 0 ||
              (transfer_sampled_fields &&
               static_cast<ParticleValidity>(
                   particles.idata(IntIdx::validity)[i]) !=
                   ParticleValidity::valid))
            return;
          const int owner = particles.idata(IntIdx::logical_owner_rank)[i];
          if (owner < 0 || owner >= ranks) {
            amrex::Gpu::Atomic::Exch(routing_error, 1);
            return;
          }
          amrex::Gpu::Atomic::AddNoRet(count_by_rank + owner, 1);
        });
      }

  std::vector<int> send_counts(ranks, 0);
  std::vector<int> send_offsets;
  int host_error = 0;
  if (reuse_topology) {
    send_counts = return_plan.return_receive_counts;
    send_offsets = return_plan.return_receive_offsets;
    const int cached_send_total = total_count(send_counts, send_offsets);
    const int *const cached_offsets = return_plan.rank_offsets.dataPtr();
    amrex::ParallelFor(
        ranks, [=] AMREX_GPU_DEVICE(const int destination) noexcept {
          const int expected =
              destination + 1 < ranks
                  ? cached_offsets[destination + 1] -
                        cached_offsets[destination]
                  : cached_send_total - cached_offsets[destination];
          if (count_by_rank[destination] != expected)
            amrex::Gpu::Atomic::Exch(routing_error, 1);
        });
  } else {
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                     return_plan.rank_counts.begin(),
                     return_plan.rank_counts.end(), send_counts.begin());
  }
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   return_plan.forward_status.begin(),
                   return_plan.forward_status.end(), &host_error);
  if (host_error)
    throw std::runtime_error(
        reuse_topology
            ? "ParticleAHFinderX cached routing topology changed without "
              "invalidation"
            : "ParticleAHFinderX found an invalid logical owner rank");

  if (!reuse_topology)
    send_offsets = exclusive_offsets(send_counts);
  const int send_total = total_count(send_counts, send_offsets);
  return_plan.rank_offsets.resize(ranks);
  if (!reuse_topology)
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, send_offsets.begin(),
                     send_offsets.end(), return_plan.rank_offsets.begin());
  return_plan.rank_cursors.resize(ranks);
  int *const cursors = return_plan.rank_cursors.dataPtr();
  amrex::ParallelFor(ranks,
                     [=] AMREX_GPU_DEVICE(const int destination) noexcept {
                       cursors[destination] = 0;
                     });
  auto &send_records = [&]() -> amrex::Gpu::DeviceVector<RoutingRecord> & {
    if constexpr (transfer_sampled_fields)
      return return_plan.sample_send_records;
    else
      return return_plan.source_send_records;
  }();
  send_records.resize(send_total > 0 ? send_total : 1);
  const int *const offsets = return_plan.rank_offsets.dataPtr();
  RoutingRecord *const records = send_records.dataPtr();

  amrex::Long source_offset = 0;
  for (const auto &container : containers)
    for (int lev = 0; lev <= container->finestLevel(); ++lev)
      for (ParIter pti(*container, lev); pti.isValid(); ++pti) {
        const int count = pti.numParticles();
        if (count == 0)
          continue;
        const auto particles =
            pti.GetParticleTile().getConstParticleTileData();
        const amrex::Long tile_source_offset = source_offset;
        amrex::ParallelFor(count,
                           [=] AMREX_GPU_DEVICE(const int i) noexcept {
          if (!particles.id(i).is_valid() ||
              particles.idata(IntIdx::solver_active)[i] == 0 ||
              (transfer_sampled_fields &&
               static_cast<ParticleValidity>(
                   particles.idata(IntIdx::validity)[i]) !=
                   ParticleValidity::valid))
            return;
          const int owner = particles.idata(IntIdx::logical_owner_rank)[i];
          const int slot = offsets[owner] +
                           amrex::Gpu::Atomic::Add(cursors + owner, 1);
          auto &record = records[slot];
          record.logical_index = surface_id_from_limbs(
              particles.idata(IntIdx::logical_index_low)[i],
              particles.idata(IntIdx::logical_index_middle)[i],
              particles.idata(IntIdx::logical_index_high)[i]);
          record.particle_id = particles.id(i);
          record.surface_id = surface_id_from_limbs(
              particles.idata(IntIdx::surface_id_low)[i],
              particles.idata(IntIdx::surface_id_middle)[i],
              particles.idata(IntIdx::surface_id_high)[i]);
          record.source_index = tile_source_offset + i;
          record.source_rank = rank;
          record.generation = particles.idata(IntIdx::generation)[i];
          record.angular_level =
              particles.idata(IntIdx::angular_level)[i];
          record.angular_levels =
              particles.idata(IntIdx::angular_levels)[i];
          record.itheta = particles.idata(IntIdx::itheta)[i];
          record.iphi = particles.idata(IntIdx::iphi)[i];
          if constexpr (transfer_sampled_fields) {
            record.sampled_level =
                particles.idata(IntIdx::sampled_level)[i];
            record.sampled_level_changed =
                particles.idata(IntIdx::sampled_level_changed)[i];
            for (int d = 0; d < 3; ++d)
              record.position[d] = particles.pos(d, i);
            record.height = particles.rdata(RealIdx::height)[i];
            record.relaxation_velocity =
                particles.rdata(RealIdx::relaxation_velocity)[i];
            record.expansion = particles.rdata(RealIdx::expansion)[i];
            for (int component = 0; component < 6; ++component) {
              record.adm[component] =
                  particles.rdata(RealIdx::gamma_component(component))[i];
              record.adm[24 + component] =
                  particles.rdata(RealIdx::curv_component(component))[i];
              for (int d = 0; d < 3; ++d)
                record.adm[6 + 6 * d + component] = particles.rdata(
                    RealIdx::d_gamma_component(d, component))[i];
            }
          }
        });
        source_offset += count;
      }

  std::vector<int> receive_counts(ranks, 0);
  std::vector<int> receive_offsets;
  if (reuse_topology) {
    receive_counts = return_plan.return_send_counts;
    receive_offsets = return_plan.return_send_offsets;
  } else {
    check_mpi(MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                           receive_counts.data(), 1, MPI_INT,
                           amrex::ParallelDescriptor::Communicator()),
              "sample count exchange");
    receive_offsets = exclusive_offsets(receive_counts);
    int remote_payload = 0;
    for (int peer = 0; peer < ranks; ++peer)
      if (peer != rank &&
          (send_counts[peer] != 0 || receive_counts[peer] != 0)) {
        remote_payload = 1;
        break;
      }
    if (ranks > 1) {
      int any_remote_payload = 0;
      check_mpi(MPI_Allreduce(
                    &remote_payload, &any_remote_payload, 1, MPI_INT, MPI_MAX,
                    amrex::ParallelDescriptor::Communicator()),
                "payload locality reduction");
      remote_payload = any_remote_payload;
    }
    return_plan.payload_uses_mpi = remote_payload != 0;
  }
  const int receive_total = total_count(receive_counts, receive_offsets);
  auto &receive_records = [&]() -> amrex::Gpu::DeviceVector<RoutingRecord> & {
    if constexpr (transfer_sampled_fields)
      return return_plan.sample_receive_records;
    else
      return return_plan.source_receive_records;
  }();
  receive_records.resize(receive_total > 0 ? receive_total : 1);
  exchange_payload_records(send_records, receive_records, send_counts,
                           send_offsets, receive_counts, receive_offsets,
                           return_plan, "sample payload exchange");

  logical_surfaces.begin_sample_routing();
  return_plan.return_slot.resize(logical.point_count);
  auto *const return_slot = return_plan.return_slot.dataPtr();
  const auto *const received = receive_records.dataPtr();
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(const int) noexcept {
    routing_error[0] = 0;
  });
  int *const invalid_record = routing_error;
  amrex::ParallelFor(
      receive_total, [=] AMREX_GPU_DEVICE(const int i) noexcept {
        const auto &record = received[i];
        const amrex::Long point = record.logical_index;
        if (point < 0 || point >= logical.point_count ||
            logical.particle_id[point] != record.particle_id ||
            logical.surface_id[point] != record.surface_id ||
            logical.generation[point] != record.generation ||
            logical.angular_level[point] != record.angular_level ||
            logical.angular_levels[point] != record.angular_levels ||
            logical.itheta[point] != record.itheta ||
            logical.iphi[point] != record.iphi) {
          amrex::Gpu::Atomic::Exch(invalid_record, 1);
          return;
        }
        const int previous =
            amrex::Gpu::Atomic::Add(logical.arrival + point, 1);
        if (previous != 0)
          return;
        return_slot[point] = i;
        logical.source_rank[point] = record.source_rank;
        logical.source_index[point] = record.source_index;
        if constexpr (transfer_sampled_fields) {
          logical.sampled_level[point] = record.sampled_level;
          logical.sampled_level_changed[point] =
              record.sampled_level_changed;
          for (int d = 0; d < 3; ++d)
            logical.position[d][point] = record.position[d];
          logical.height[point] = record.height;
          logical.relaxation_velocity[point] = record.relaxation_velocity;
          logical.expansion[point] = record.expansion;
          for (int component = 0; component < 6; ++component) {
            logical.gamma(component, point) = record.adm[component];
            logical.curv(component, point) = record.adm[24 + component];
            for (int d = 0; d < 3; ++d)
              logical.d_gamma(d, component, point) =
                  record.adm[6 + 6 * d + component];
          }
        }
      });

  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   return_plan.forward_status.begin(),
                   return_plan.forward_status.end(), &host_error);
  const auto arrivals = logical_surfaces.local_arrival_stats();
  const amrex::Long control_device_to_host_bytes =
      (reuse_topology ? 2 * sizeof(int)
                      : static_cast<amrex::Long>(ranks + 2) * sizeof(int)) +
      2 * sizeof(amrex::Long);
  const amrex::Long control_device_to_host_copies =
      reuse_topology ? 3 : 4;
  auto stats = routing_performance_stats<RoutingRecord>(
      ranks, send_total, receive_total, send_counts[rank],
      receive_counts[rank], remote_peer_count(send_counts, rank),
      remote_peer_count(receive_counts, rank), control_device_to_host_bytes,
      control_device_to_host_copies,
      reuse_topology ? 0 : static_cast<amrex::Long>(ranks) * sizeof(int),
      reuse_topology ? 0 : 1, !reuse_topology,
      return_plan.payload_uses_mpi);
  stats.sent = send_total;
  stats.received = receive_total;
  stats.missing = arrivals.missing;
  stats.duplicate = arrivals.duplicate;
  stats.invalid = host_error;
  stats.forward_topology_reuses = reuse_topology;
  stats.forward_topology_rebuilds = !reuse_topology;
  return_plan.return_send_counts = receive_counts;
  return_plan.return_send_offsets = receive_offsets;
  return_plan.return_receive_counts = send_counts;
  return_plan.return_receive_offsets = send_offsets;
  return_plan.logical_point_count = logical.point_count;
  return_plan.active_point_count = active_point_count;
  return_plan.ranks = ranks;
  return_plan.valid = true;
  return stats;
}

} // namespace

SurfaceRoutingStats route_particles_to_logical_owners(
    const std::vector<std::unique_ptr<SurfaceParticleContainer>> &containers,
    LogicalSurfaceStorage &logical_surfaces, const LogicalRoutingMode mode,
    SurfaceRoutePlan &return_plan, const amrex::Long active_point_count) {
  switch (mode) {
  case LogicalRoutingMode::sampled_fields:
    return route_particles_to_logical_owners_impl<true>(containers,
                                                        logical_surfaces,
                                                        return_plan,
                                                        active_point_count);
  case LogicalRoutingMode::source_metadata:
    return route_particles_to_logical_owners_impl<false>(containers,
                                                         logical_surfaces,
                                                         return_plan,
                                                         active_point_count);
  }
  throw std::runtime_error("ParticleAHFinderX received an unknown logical "
                           "routing mode");
}

SurfaceRoutingStats route_weyl_to_logical_owners(
    const MeshWeylPoint *const mesh_points,
    LogicalSurfaceStorage &logical_surfaces,
    SurfaceRoutePlan &route_plan, const amrex::Long active_point_count,
    const amrex::Long mesh_point_count) {
  const int rank = amrex::ParallelDescriptor::MyProc();
  const int ranks = amrex::ParallelDescriptor::NProcs();
  const auto logical = logical_surfaces.device_view();
  if (!mesh_points || mesh_point_count < 0 || !route_plan.valid ||
      route_plan.ranks != ranks ||
      route_plan.logical_point_count != logical.point_count ||
      route_plan.active_point_count != active_point_count ||
      route_plan.return_send_counts.size() !=
          static_cast<std::size_t>(ranks) ||
      route_plan.return_send_offsets.size() !=
          static_cast<std::size_t>(ranks) ||
      route_plan.return_receive_counts.size() !=
          static_cast<std::size_t>(ranks) ||
      route_plan.return_receive_offsets.size() !=
          static_cast<std::size_t>(ranks))
    throw std::runtime_error(
        "ParticleAHFinderX Weyl analysis has no matching final sample route");

  const auto &send_counts = route_plan.return_receive_counts;
  const auto &send_offsets = route_plan.return_receive_offsets;
  const auto &receive_counts = route_plan.return_send_counts;
  const auto &receive_offsets = route_plan.return_send_offsets;
  const int send_total = total_count(send_counts, send_offsets);
  const int receive_total = total_count(receive_counts, receive_offsets);
  if (route_plan.sample_send_records.size() <
      static_cast<std::size_t>(send_total))
    throw std::runtime_error(
        "ParticleAHFinderX Weyl analysis lost its mesh sample ordering");

  route_plan.weyl_send_records.resize(send_total > 0 ? send_total : 1);
  const auto *const samples = route_plan.sample_send_records.dataPtr();
  auto *const records = route_plan.weyl_send_records.dataPtr();
  amrex::ParallelFor(send_total,
                     [=] AMREX_GPU_DEVICE(const int index) noexcept {
    const auto &sample = samples[index];
    auto &record = records[index];
    record.logical_index = sample.logical_index;
    record.particle_id = sample.particle_id;
    record.surface_id = sample.surface_id;
    if (sample.source_index < 0 ||
        sample.source_index >= mesh_point_count) {
      record.status = static_cast<int>(WeylGatherStatus::inactive);
      for (int component = 0; component < 6; ++component) {
        record.electric[component] = 0;
        record.magnetic[component] = 0;
      }
      return;
    }
    const auto &source = mesh_points[sample.source_index];
    record.status = source.status;
    for (int component = 0; component < 6; ++component) {
      record.electric[component] = source.electric[component];
      record.magnetic[component] = source.magnetic[component];
    }
  });

  route_plan.weyl_receive_records.resize(receive_total > 0 ? receive_total
                                                            : 1);
  exchange_payload_records(
      route_plan.weyl_send_records, route_plan.weyl_receive_records,
      send_counts, send_offsets, receive_counts, receive_offsets, route_plan,
      "Weyl payload exchange");

  const auto weyl = logical_surfaces.begin_weyl_routing();
  route_plan.forward_status.resize(1);
  int *const invalid_record = route_plan.forward_status.dataPtr();
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(const int) noexcept {
    invalid_record[0] = 0;
  });
  const auto *const received = route_plan.weyl_receive_records.dataPtr();
  amrex::ParallelFor(
      receive_total, [=] AMREX_GPU_DEVICE(const int index) noexcept {
        const auto &record = received[index];
        const amrex::Long point = record.logical_index;
        if (point < 0 || point >= logical.point_count ||
            logical.particle_id[point] != record.particle_id ||
            logical.surface_id[point] != record.surface_id ||
            record.status != static_cast<int>(WeylGatherStatus::valid)) {
          amrex::Gpu::Atomic::Exch(invalid_record, 1);
          return;
        }
        const int previous =
            amrex::Gpu::Atomic::Add(weyl.arrival + point, 1);
        if (previous != 0)
          return;
        for (int component = 0; component < 6; ++component) {
          weyl.electric_component(component, point) =
              record.electric[component];
          weyl.magnetic_component(component, point) =
              record.magnetic[component];
        }
      });

  int host_error = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   route_plan.forward_status.begin(),
                   route_plan.forward_status.end(), &host_error);
  const auto arrivals = logical_surfaces.local_weyl_arrival_stats();
  auto stats = routing_performance_stats<LogicalWeylRecord>(
      ranks, send_total, receive_total, send_counts[rank],
      receive_counts[rank], remote_peer_count(send_counts, rank),
      remote_peer_count(receive_counts, rank),
      sizeof(int) + 2 * sizeof(amrex::Long), 2, 0, 0, false,
      route_plan.payload_uses_mpi);
  stats.sent = send_total;
  stats.received = receive_total;
  stats.missing = arrivals.missing;
  stats.duplicate = arrivals.duplicate;
  stats.invalid = host_error;
  return stats;
}

SurfaceRoutingStats route_gauge_to_logical_owners(
    const MeshGaugePoint *const mesh_points,
    LogicalSurfaceStorage &logical_surfaces,
    SurfaceRoutePlan &route_plan, const amrex::Long active_point_count,
    const amrex::Long mesh_point_count) {
  const int rank = amrex::ParallelDescriptor::MyProc();
  const int ranks = amrex::ParallelDescriptor::NProcs();
  const auto logical = logical_surfaces.device_view();
  if (!mesh_points || mesh_point_count < 0 || !route_plan.valid ||
      route_plan.ranks != ranks ||
      route_plan.logical_point_count != logical.point_count ||
      route_plan.active_point_count != active_point_count ||
      route_plan.return_send_counts.size() !=
          static_cast<std::size_t>(ranks) ||
      route_plan.return_send_offsets.size() !=
          static_cast<std::size_t>(ranks) ||
      route_plan.return_receive_counts.size() !=
          static_cast<std::size_t>(ranks) ||
      route_plan.return_receive_offsets.size() !=
          static_cast<std::size_t>(ranks))
    throw std::runtime_error(
        "ParticleAHFinderX gauge analysis has no matching final sample route");

  const auto &send_counts = route_plan.return_receive_counts;
  const auto &send_offsets = route_plan.return_receive_offsets;
  const auto &receive_counts = route_plan.return_send_counts;
  const auto &receive_offsets = route_plan.return_send_offsets;
  const int send_total = total_count(send_counts, send_offsets);
  const int receive_total = total_count(receive_counts, receive_offsets);
  if (route_plan.sample_send_records.size() <
      static_cast<std::size_t>(send_total))
    throw std::runtime_error(
        "ParticleAHFinderX gauge analysis lost its mesh sample ordering");

  route_plan.gauge_send_records.resize(send_total > 0 ? send_total : 1);
  const auto *const samples = route_plan.sample_send_records.dataPtr();
  auto *const records = route_plan.gauge_send_records.dataPtr();
  amrex::ParallelFor(send_total,
                     [=] AMREX_GPU_DEVICE(const int index) noexcept {
    const auto &sample = samples[index];
    auto &record = records[index];
    record.logical_index = sample.logical_index;
    record.particle_id = sample.particle_id;
    record.surface_id = sample.surface_id;
    if (sample.source_index < 0 || sample.source_index >= mesh_point_count) {
      record.status = static_cast<int>(GaugeGatherStatus::inactive);
      for (int component = 0; component < FourGaugeLayout::component_count;
           ++component)
        record.value[component] = 0;
      return;
    }
    const auto &source = mesh_points[sample.source_index];
    record.status = source.status;
    for (int component = 0; component < FourGaugeLayout::component_count;
         ++component)
      record.value[component] = source.value[component];
  });

  route_plan.gauge_receive_records.resize(receive_total > 0 ? receive_total
                                                             : 1);
  exchange_payload_records(
      route_plan.gauge_send_records, route_plan.gauge_receive_records,
      send_counts, send_offsets, receive_counts, receive_offsets, route_plan,
      "gauge payload exchange");

  const auto gauge = logical_surfaces.begin_gauge_routing();
  route_plan.forward_status.resize(1);
  int *const invalid_record = route_plan.forward_status.dataPtr();
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(const int) noexcept {
    invalid_record[0] = 0;
  });
  const auto *const received = route_plan.gauge_receive_records.dataPtr();
  amrex::ParallelFor(
      receive_total, [=] AMREX_GPU_DEVICE(const int index) noexcept {
        const auto &record = received[index];
        const amrex::Long point = record.logical_index;
        if (point < 0 || point >= logical.point_count ||
            logical.particle_id[point] != record.particle_id ||
            logical.surface_id[point] != record.surface_id ||
            record.status != static_cast<int>(GaugeGatherStatus::valid)) {
          amrex::Gpu::Atomic::Exch(invalid_record, 1);
          return;
        }
        const int previous =
            amrex::Gpu::Atomic::Add(gauge.arrival + point, 1);
        if (previous != 0)
          return;
        for (int component = 0;
             component < FourGaugeLayout::component_count; ++component)
          gauge.component(component, point) = record.value[component];
      });

  int host_error = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   route_plan.forward_status.begin(),
                   route_plan.forward_status.end(), &host_error);
  const auto arrivals = logical_surfaces.local_gauge_arrival_stats();
  auto stats = routing_performance_stats<LogicalGaugeRecord>(
      ranks, send_total, receive_total, send_counts[rank],
      receive_counts[rank], remote_peer_count(send_counts, rank),
      remote_peer_count(receive_counts, rank),
      sizeof(int) + 2 * sizeof(amrex::Long), 2, 0, 0, false,
      route_plan.payload_uses_mpi);
  stats.sent = send_total;
  stats.received = receive_total;
  stats.missing = arrivals.missing;
  stats.duplicate = arrivals.duplicate;
  stats.invalid = host_error;
  return stats;
}

SurfaceRoutingStats route_matter_to_logical_owners(
    const MeshMatterPoint *const mesh_points,
    LogicalSurfaceStorage &logical_surfaces,
    SurfaceRoutePlan &route_plan, const amrex::Long active_point_count,
    const amrex::Long mesh_point_count) {
  const int rank = amrex::ParallelDescriptor::MyProc();
  const int ranks = amrex::ParallelDescriptor::NProcs();
  const auto logical = logical_surfaces.device_view();
  if (!mesh_points || mesh_point_count < 0 || !route_plan.valid ||
      route_plan.ranks != ranks ||
      route_plan.logical_point_count != logical.point_count ||
      route_plan.active_point_count != active_point_count ||
      route_plan.return_send_counts.size() !=
          static_cast<std::size_t>(ranks) ||
      route_plan.return_send_offsets.size() !=
          static_cast<std::size_t>(ranks) ||
      route_plan.return_receive_counts.size() !=
          static_cast<std::size_t>(ranks) ||
      route_plan.return_receive_offsets.size() !=
          static_cast<std::size_t>(ranks))
    throw std::runtime_error(
        "ParticleAHFinderX matter analysis has no matching final sample route");

  const auto &send_counts = route_plan.return_receive_counts;
  const auto &send_offsets = route_plan.return_receive_offsets;
  const auto &receive_counts = route_plan.return_send_counts;
  const auto &receive_offsets = route_plan.return_send_offsets;
  const int send_total = total_count(send_counts, send_offsets);
  const int receive_total = total_count(receive_counts, receive_offsets);
  if (route_plan.sample_send_records.size() <
      static_cast<std::size_t>(send_total))
    throw std::runtime_error(
        "ParticleAHFinderX matter analysis lost its mesh sample ordering");

  route_plan.matter_send_records.resize(send_total > 0 ? send_total : 1);
  const auto *const samples = route_plan.sample_send_records.dataPtr();
  auto *const records = route_plan.matter_send_records.dataPtr();
  amrex::ParallelFor(send_total,
                     [=] AMREX_GPU_DEVICE(const int index) noexcept {
    const auto &sample = samples[index];
    auto &record = records[index];
    record.logical_index = sample.logical_index;
    record.particle_id = sample.particle_id;
    record.surface_id = sample.surface_id;
    if (sample.source_index < 0 || sample.source_index >= mesh_point_count) {
      record.status = static_cast<int>(MatterGatherStatus::inactive);
      for (int component = 0; component < FourMatterLayout::component_count;
           ++component)
        record.value[component] = 0;
      return;
    }
    const auto &source = mesh_points[sample.source_index];
    record.status = source.status;
    for (int component = 0; component < FourMatterLayout::component_count;
         ++component)
      record.value[component] = source.value[component];
  });

  route_plan.matter_receive_records.resize(receive_total > 0 ? receive_total
                                                               : 1);
  exchange_payload_records(
      route_plan.matter_send_records, route_plan.matter_receive_records,
      send_counts, send_offsets, receive_counts, receive_offsets, route_plan,
      "matter payload exchange");

  const auto matter = logical_surfaces.begin_matter_routing();
  route_plan.forward_status.resize(1);
  int *const invalid_record = route_plan.forward_status.dataPtr();
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(const int) noexcept {
    invalid_record[0] = 0;
  });
  const auto *const received = route_plan.matter_receive_records.dataPtr();
  amrex::ParallelFor(
      receive_total, [=] AMREX_GPU_DEVICE(const int index) noexcept {
        const auto &record = received[index];
        const amrex::Long point = record.logical_index;
        if (point < 0 || point >= logical.point_count ||
            logical.particle_id[point] != record.particle_id ||
            logical.surface_id[point] != record.surface_id ||
            record.status != static_cast<int>(MatterGatherStatus::valid)) {
          amrex::Gpu::Atomic::Exch(invalid_record, 1);
          return;
        }
        const int previous =
            amrex::Gpu::Atomic::Add(matter.arrival + point, 1);
        if (previous != 0)
          return;
        for (int component = 0;
             component < FourMatterLayout::component_count; ++component)
          matter.component(component, point) = record.value[component];
      });

  int host_error = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   route_plan.forward_status.begin(),
                   route_plan.forward_status.end(), &host_error);
  const auto arrivals = logical_surfaces.local_matter_arrival_stats();
  auto stats = routing_performance_stats<LogicalMatterRecord>(
      ranks, send_total, receive_total, send_counts[rank],
      receive_counts[rank], remote_peer_count(send_counts, rank),
      remote_peer_count(receive_counts, rank),
      sizeof(int) + 2 * sizeof(amrex::Long), 2, 0, 0, false,
      route_plan.payload_uses_mpi);
  stats.sent = send_total;
  stats.received = receive_total;
  stats.missing = arrivals.missing;
  stats.duplicate = arrivals.duplicate;
  stats.invalid = host_error;
  return stats;
}

SurfaceRoutingStats route_updates_to_mesh_owners(
    LogicalSurfaceStorage &logical_surfaces,
    const std::vector<std::unique_ptr<SurfaceParticleContainer>> &containers,
    SurfaceRoutePlan &return_plan,
    const int minimum_active_level, const int maximum_active_level) {
  const int rank = amrex::ParallelDescriptor::MyProc();
  const int ranks = amrex::ParallelDescriptor::NProcs();
  const bool enforce_active_band = minimum_active_level >= 0;
  if (enforce_active_band &&
      (maximum_active_level <= minimum_active_level))
    throw std::runtime_error(
        "ParticleAHFinderX update routing received an invalid active band");
  const auto logical = logical_surfaces.device_view();
  if (logical.point_count > INT_MAX)
    throw std::runtime_error(
        "ParticleAHFinderX update routing currently requires at most "
        "INT_MAX logical points per owner");

  if (!return_plan.valid || return_plan.ranks != ranks ||
      return_plan.logical_point_count != logical.point_count ||
      return_plan.return_send_counts.size() !=
          static_cast<std::size_t>(ranks) ||
      return_plan.return_send_offsets.size() !=
          static_cast<std::size_t>(ranks) ||
      return_plan.return_receive_counts.size() !=
          static_cast<std::size_t>(ranks) ||
      return_plan.return_receive_offsets.size() !=
          static_cast<std::size_t>(ranks) ||
      return_plan.return_slot.size() !=
          static_cast<std::size_t>(logical.point_count))
    throw std::runtime_error(
        "ParticleAHFinderX logical update has no valid return route plan");

  const auto &send_counts = return_plan.return_send_counts;
  const auto &send_offsets = return_plan.return_send_offsets;
  const auto &receive_counts = return_plan.return_receive_counts;
  const auto &receive_offsets = return_plan.return_receive_offsets;
  const int send_total = total_count(send_counts, send_offsets);
  const int receive_total = total_count(receive_counts, receive_offsets);
  return_plan.update_status.resize(2);
  int *const invalid_record = return_plan.update_status.dataPtr();
  int *const outside_active_band = return_plan.update_status.dataPtr() + 1;
  amrex::ParallelFor(2, [=] AMREX_GPU_DEVICE(const int index) noexcept {
    invalid_record[index] = 0;
  });
  const auto *const return_slot = return_plan.return_slot.dataPtr();
  return_plan.update_send_records.resize(send_total > 0 ? send_total : 1);
  auto *const records = return_plan.update_send_records.dataPtr();
  amrex::ParallelFor(
      logical.point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &descriptor =
            logical.descriptors[logical.surface_ordinal[point]];
        if (!descriptor.due)
          return;
        const int destination = logical.source_rank[point];
        const amrex::Long slot = return_slot[point];
        if (logical.arrival[point] != 1 || destination < 0 ||
            destination >= ranks || logical.source_index[point] < 0 ||
            slot < 0 || slot >= send_total) {
          amrex::Gpu::Atomic::Exch(invalid_record, 1);
          return;
        }
        auto &record = records[slot];
        record.source_index = logical.source_index[point];
        record.particle_id = logical.particle_id[point];
        for (int d = 0; d < 3; ++d)
          record.position[d] = logical.position[d][point];
        record.height = logical.height[point];
        record.relaxation_velocity = logical.relaxation_velocity[point];
        record.expansion = logical.expansion[point];
        for (int d = 0; d < 3; ++d)
          record.normal[d] = logical.normal_component(d, point);
        record.area_weight = logical.area_weight[point];
      });

  return_plan.update_receive_records.resize(receive_total > 0 ? receive_total
                                                              : 1);
  exchange_payload_records(
      return_plan.update_send_records, return_plan.update_receive_records,
      send_counts, send_offsets, receive_counts, receive_offsets,
      return_plan, "update payload exchange");

  amrex::Long local_population = 0;
  for (const auto &container : containers)
    local_population += container->local_particle_count();
  return_plan.update_arrival.resize(local_population);
  return_plan.update_expected.resize(local_population);
  return_plan.ordered_updates.resize(local_population);
  auto *const arrival = return_plan.update_arrival.dataPtr();
  auto *const expected = return_plan.update_expected.dataPtr();
  auto *const ordered = return_plan.ordered_updates.dataPtr();
  const auto *const received = return_plan.update_receive_records.dataPtr();
  amrex::ParallelFor(
      local_population,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        arrival[point] = 0;
        expected[point] = 0;
      });
  amrex::ParallelFor(
      receive_total, [=] AMREX_GPU_DEVICE(const int i) noexcept {
        const auto &record = received[i];
        if (record.source_index < 0 ||
            record.source_index >= local_population) {
          amrex::Gpu::Atomic::Exch(invalid_record, 1);
          return;
        }
        const int previous = amrex::Gpu::Atomic::Add(
            arrival + record.source_index, 1);
        if (previous == 0)
          ordered[record.source_index] = record;
      });

  amrex::Long source_offset = 0;
  for (const auto &container : containers) {
    if (enforce_active_band &&
        maximum_active_level > container->finestLevel() + 1)
      throw std::runtime_error(
          "ParticleAHFinderX update routing active band exceeds the particle "
          "hierarchy");
    if (!container->m_particle_locator.isValid(
            container->GetParGDB(), container->do_tiling,
            container->tile_size)) {
      container->m_particle_locator.build(container->GetParGDB(),
                                          container->do_tiling,
                                          container->tile_size);
    }
    container->m_particle_locator.setGeometry(container->GetParGDB());
    const auto assign_grid = container->m_particle_locator.getGridAssignor();
    for (int lev = 0; lev <= container->finestLevel(); ++lev)
      for (ParIter pti(*container, lev); pti.isValid(); ++pti) {
        const int count = pti.numParticles();
        if (count == 0)
          continue;
        auto particles = pti.GetParticleTile().getParticleTileData();
        const amrex::Long tile_source_offset = source_offset;
        amrex::ParallelFor(count,
                           [=] AMREX_GPU_DEVICE(const int i) noexcept {
          const amrex::Long source_index = tile_source_offset + i;
          const bool active = particles.id(i).is_valid() &&
                              particles.idata(IntIdx::solver_active)[i] != 0;
          expected[source_index] = active;
          if (!active) {
            if (arrival[source_index] != 0)
              amrex::Gpu::Atomic::Exch(invalid_record, 1);
            return;
          }
          if (arrival[source_index] != 1 ||
              ordered[source_index].particle_id != particles.id(i)) {
            amrex::Gpu::Atomic::Exch(invalid_record, 1);
            return;
          }
          const auto &record = ordered[source_index];
          if (enforce_active_band) {
            auto candidate = particles.getSuperParticle(i);
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
              candidate.pos(d) = record.position[d];
            const auto destination = assign_grid(
                candidate, minimum_active_level, maximum_active_level - 1, 0);
            if (amrex::get<0>(destination) < 0) {
              amrex::Gpu::Atomic::Exch(outside_active_band, 1);
              return;
            }
          }
          for (int d = 0; d < 3; ++d)
            particles.pos(d, i) = record.position[d];
          particles.rdata(RealIdx::height)[i] = record.height;
          particles.rdata(RealIdx::relaxation_velocity)[i] =
              record.relaxation_velocity;
          particles.rdata(RealIdx::expansion)[i] = record.expansion;
          for (int d = 0; d < 3; ++d)
            particles.rdata(RealIdx::normal_component(d))[i] =
                record.normal[d];
          particles.rdata(RealIdx::area_weight)[i] = record.area_weight;
        });
        source_offset += count;
      }
  }

  amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
  amrex::ReduceData<amrex::Long, amrex::Long> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  reduce_ops.eval(
      local_population, reduce_data,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept -> ReduceTuple {
        return {expected[point] != 0 && arrival[point] == 0,
                arrival[point] > 1};
      });
  const auto reduced = reduce_data.value();
  int host_scatter_status[2]{0, 0};
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   return_plan.update_status.begin(),
                   return_plan.update_status.end(), host_scatter_status);
  const int host_error = host_scatter_status[0];
  auto stats = routing_performance_stats<MeshUpdateRecord>(
      ranks, send_total, receive_total, send_counts[rank],
      receive_counts[rank], remote_peer_count(send_counts, rank),
      remote_peer_count(receive_counts, rank),
      2 * sizeof(int) + 2 * sizeof(amrex::Long), 2, 0, 0, false,
      return_plan.payload_uses_mpi);
  stats.sent = send_total;
  stats.received = receive_total;
  stats.missing = amrex::get<0>(reduced);
  stats.duplicate = amrex::get<1>(reduced);
  stats.invalid = host_error;
  stats.outside_active_band = host_scatter_status[1];
  return stats;
}

} // namespace ParticleAHFinderX
