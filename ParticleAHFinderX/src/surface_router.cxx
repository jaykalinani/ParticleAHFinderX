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
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ParticleAHFinderX {
namespace {

template <typename Record>
void make_byte_layout(const std::vector<int> &counts,
                      const std::vector<int> &offsets,
                      std::vector<int> &byte_counts,
                      std::vector<int> &byte_offsets) {
  const auto record_size = static_cast<int>(sizeof(Record));
  byte_counts.resize(counts.size());
  byte_offsets.resize(offsets.size());
  for (std::size_t rank = 0; rank < counts.size(); ++rank) {
    if (counts[rank] > INT_MAX / record_size ||
        offsets[rank] > INT_MAX / record_size)
      throw std::runtime_error(
          "ParticleAHFinderX routing exceeds the MPI byte-count limit");
    byte_counts[rank] = counts[rank] * record_size;
    byte_offsets[rank] = offsets[rank] * record_size;
  }
}

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

void check_mpi(const int error, const char *const operation) {
  if (error != MPI_SUCCESS)
    throw std::runtime_error(std::string("ParticleAHFinderX ") + operation +
                             " failed");
}

} // namespace

namespace {

template <bool transfer_sampled_fields>
SurfaceRoutingStats route_particles_to_logical_owners_impl(
    const std::vector<std::unique_ptr<SurfaceParticleContainer>> &containers,
    LogicalSurfaceStorage &logical_surfaces) {
  using RoutingRecord =
      std::conditional_t<transfer_sampled_fields, LogicalSampleRecord,
                         LogicalSourceRecord>;
  const int rank = amrex::ParallelDescriptor::MyProc();
  const int ranks = amrex::ParallelDescriptor::NProcs();
  amrex::Long local_population = 0;
  for (const auto &container : containers)
    local_population += container->local_particle_count();
  if (local_population > INT_MAX)
    throw std::runtime_error(
        "ParticleAHFinderX routing currently requires at most INT_MAX local "
        "surface particles");

  amrex::Gpu::DeviceVector<int> device_counts(ranks, 0);
  amrex::Gpu::DeviceVector<int> device_error(1, 0);
  int *const count_by_rank = device_counts.dataPtr();
  int *const routing_error = device_error.dataPtr();

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
  int host_error = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_counts.begin(),
                   device_counts.end(), send_counts.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_error.begin(),
                   device_error.end(), &host_error);
  if (host_error)
    throw std::runtime_error(
        "ParticleAHFinderX found an invalid logical owner rank");

  const auto send_offsets = exclusive_offsets(send_counts);
  const int send_total = total_count(send_counts, send_offsets);
  amrex::Gpu::DeviceVector<int> device_offsets(ranks);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, send_offsets.begin(),
                   send_offsets.end(), device_offsets.begin());
  amrex::Gpu::DeviceVector<int> device_cursors(ranks, 0);
  amrex::Gpu::DeviceVector<RoutingRecord> send_records(
      send_total > 0 ? send_total : 1);
  const int *const offsets = device_offsets.dataPtr();
  int *const cursors = device_cursors.dataPtr();
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
  check_mpi(MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                         receive_counts.data(), 1, MPI_INT,
                         amrex::ParallelDescriptor::Communicator()),
            "sample count exchange");
  const auto receive_offsets = exclusive_offsets(receive_counts);
  const int receive_total = total_count(receive_counts, receive_offsets);
  std::vector<int> send_byte_counts;
  std::vector<int> send_byte_offsets;
  std::vector<int> receive_byte_counts;
  std::vector<int> receive_byte_offsets;
  make_byte_layout<RoutingRecord>(send_counts, send_offsets, send_byte_counts,
                                  send_byte_offsets);
  make_byte_layout<RoutingRecord>(receive_counts, receive_offsets,
                                  receive_byte_counts, receive_byte_offsets);

  amrex::Gpu::DeviceVector<RoutingRecord> receive_records(
      receive_total > 0 ? receive_total : 1);
  if (ranks == 1) {
    amrex::Gpu::copy(amrex::Gpu::deviceToDevice, send_records.begin(),
                     send_records.begin() + send_total,
                     receive_records.begin());
  } else {
    amrex::Gpu::streamSynchronize();
    check_mpi(MPI_Alltoallv(
                  send_records.dataPtr(), send_byte_counts.data(),
                  send_byte_offsets.data(), MPI_BYTE,
                  receive_records.dataPtr(), receive_byte_counts.data(),
                  receive_byte_offsets.data(), MPI_BYTE,
                  amrex::ParallelDescriptor::Communicator()),
              "sample payload exchange");
  }

  logical_surfaces.begin_sample_routing();
  auto logical = logical_surfaces.device_view();
  const auto *const received = receive_records.dataPtr();
  amrex::Gpu::DeviceVector<int> scatter_error(1, 0);
  int *const invalid_record = scatter_error.dataPtr();
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

  amrex::Gpu::copy(amrex::Gpu::deviceToHost, scatter_error.begin(),
                   scatter_error.end(), &host_error);
  const auto arrivals = logical_surfaces.local_arrival_stats();
  return {send_total, receive_total, arrivals.missing, arrivals.duplicate,
          host_error};
}

} // namespace

SurfaceRoutingStats route_particles_to_logical_owners(
    const std::vector<std::unique_ptr<SurfaceParticleContainer>> &containers,
    LogicalSurfaceStorage &logical_surfaces, const LogicalRoutingMode mode) {
  switch (mode) {
  case LogicalRoutingMode::sampled_fields:
    return route_particles_to_logical_owners_impl<true>(containers,
                                                        logical_surfaces);
  case LogicalRoutingMode::source_metadata:
    return route_particles_to_logical_owners_impl<false>(containers,
                                                         logical_surfaces);
  }
  throw std::runtime_error("ParticleAHFinderX received an unknown logical "
                           "routing mode");
}

SurfaceRoutingStats route_updates_to_mesh_owners(
    LogicalSurfaceStorage &logical_surfaces,
    const std::vector<std::unique_ptr<SurfaceParticleContainer>> &containers) {
  const int ranks = amrex::ParallelDescriptor::NProcs();
  const auto logical = logical_surfaces.device_view();
  if (logical.point_count > INT_MAX)
    throw std::runtime_error(
        "ParticleAHFinderX update routing currently requires at most "
        "INT_MAX logical points per owner");

  amrex::Gpu::DeviceVector<int> device_counts(ranks, 0);
  amrex::Gpu::DeviceVector<int> device_error(1, 0);
  int *const count_by_rank = device_counts.dataPtr();
  int *const routing_error = device_error.dataPtr();
  amrex::ParallelFor(
      logical.point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &descriptor =
            logical.descriptors[logical.surface_ordinal[point]];
        if (!descriptor.due)
          return;
        const int destination = logical.source_rank[point];
        if (logical.arrival[point] != 1 || destination < 0 ||
            destination >= ranks || logical.source_index[point] < 0) {
          amrex::Gpu::Atomic::Exch(routing_error, 1);
          return;
        }
        amrex::Gpu::Atomic::AddNoRet(count_by_rank + destination, 1);
      });

  std::vector<int> send_counts(ranks, 0);
  int host_error = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_counts.begin(),
                   device_counts.end(), send_counts.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_error.begin(),
                   device_error.end(), &host_error);
  if (host_error)
    throw std::runtime_error(
        "ParticleAHFinderX logical update has invalid source metadata");

  const auto send_offsets = exclusive_offsets(send_counts);
  const int send_total = total_count(send_counts, send_offsets);
  amrex::Gpu::DeviceVector<int> device_offsets(ranks);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, send_offsets.begin(),
                   send_offsets.end(), device_offsets.begin());
  amrex::Gpu::DeviceVector<int> device_cursors(ranks, 0);
  amrex::Gpu::DeviceVector<MeshUpdateRecord> send_records(
      send_total > 0 ? send_total : 1);
  const int *const offsets = device_offsets.dataPtr();
  int *const cursors = device_cursors.dataPtr();
  auto *const records = send_records.dataPtr();
  amrex::ParallelFor(
      logical.point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &descriptor =
            logical.descriptors[logical.surface_ordinal[point]];
        if (!descriptor.due)
          return;
        const int destination = logical.source_rank[point];
        const int slot = offsets[destination] +
                         amrex::Gpu::Atomic::Add(cursors + destination, 1);
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

  std::vector<int> receive_counts(ranks, 0);
  check_mpi(MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                         receive_counts.data(), 1, MPI_INT,
                         amrex::ParallelDescriptor::Communicator()),
            "update count exchange");
  const auto receive_offsets = exclusive_offsets(receive_counts);
  const int receive_total = total_count(receive_counts, receive_offsets);
  std::vector<int> send_byte_counts;
  std::vector<int> send_byte_offsets;
  std::vector<int> receive_byte_counts;
  std::vector<int> receive_byte_offsets;
  make_byte_layout<MeshUpdateRecord>(send_counts, send_offsets,
                                     send_byte_counts, send_byte_offsets);
  make_byte_layout<MeshUpdateRecord>(receive_counts, receive_offsets,
                                     receive_byte_counts,
                                     receive_byte_offsets);
  amrex::Gpu::DeviceVector<MeshUpdateRecord> receive_records(
      receive_total > 0 ? receive_total : 1);
  if (ranks == 1) {
    amrex::Gpu::copy(amrex::Gpu::deviceToDevice, send_records.begin(),
                     send_records.begin() + send_total,
                     receive_records.begin());
  } else {
    amrex::Gpu::streamSynchronize();
    check_mpi(MPI_Alltoallv(
                  send_records.dataPtr(), send_byte_counts.data(),
                  send_byte_offsets.data(), MPI_BYTE,
                  receive_records.dataPtr(), receive_byte_counts.data(),
                  receive_byte_offsets.data(), MPI_BYTE,
                  amrex::ParallelDescriptor::Communicator()),
              "update payload exchange");
  }

  amrex::Long local_population = 0;
  for (const auto &container : containers)
    local_population += container->local_particle_count();
  amrex::Gpu::DeviceVector<int> update_arrival(local_population, 0);
  amrex::Gpu::DeviceVector<int> update_expected(local_population, 0);
  amrex::Gpu::DeviceVector<MeshUpdateRecord> ordered_updates(local_population);
  auto *const arrival = update_arrival.dataPtr();
  auto *const expected = update_expected.dataPtr();
  auto *const ordered = ordered_updates.dataPtr();
  const auto *const received = receive_records.dataPtr();
  amrex::Gpu::DeviceVector<int> scatter_error(1, 0);
  int *const invalid_record = scatter_error.dataPtr();
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
  for (const auto &container : containers)
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
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, scatter_error.begin(),
                   scatter_error.end(), &host_error);
  return {send_total, receive_total, amrex::get<0>(reduced),
          amrex::get<1>(reduced), host_error};
}

} // namespace ParticleAHFinderX
