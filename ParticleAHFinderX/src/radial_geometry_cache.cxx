/**
 * \file radial_geometry_cache.cxx
 * \brief Portable construction and lookup for frozen radial geometry.
 */
#include "radial_geometry_cache.hxx"

#include "angular_geometry.hxx"
#include "interpolation.hxx"
#include "surface_particle_container.hxx"

#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <mpi.h>

#include <climits>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ParticleAHFinderX {

namespace {

enum class RadialProbeStatus : int {
  valid = 0,
  outside_active_band = 1,
  stencil_unavailable = 2,
  nonfinite = 3,
  invalid_owner = 4,
};

struct RadialProbePosition {
  amrex::Real value[3];

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
  pos(const int d) const noexcept {
    return value[d];
  }
};

struct RadialProbeRequest {
  amrex::Long cache_value_index = -1;
  amrex::Real position[3]{};
  int level = -1;
  int grid = -1;
  int status = static_cast<int>(RadialProbeStatus::invalid_owner);
};

struct RadialProbeResponse {
  amrex::Long cache_value_index = -1;
  amrex::Real adm[30]{};
  int sampled_level = -1;
  int status = static_cast<int>(RadialProbeStatus::invalid_owner);
};

std::vector<int> exclusive_offsets(const std::vector<int> &counts) {
  std::vector<int> offsets(counts.size(), 0);
  for (std::size_t rank = 1; rank < counts.size(); ++rank) {
    if (offsets[rank - 1] > INT_MAX - counts[rank - 1])
      throw std::runtime_error(
          "ParticleAHFinderX radial-probe count overflowed");
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
        "ParticleAHFinderX radial-probe count overflowed");
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

template <typename Record> int mpi_record_bytes(const int count) {
  if (count < 0 || sizeof(Record) > static_cast<std::size_t>(INT_MAX) ||
      static_cast<std::size_t>(count) >
          static_cast<std::size_t>(INT_MAX) / sizeof(Record))
    throw std::runtime_error(
        "ParticleAHFinderX radial-probe payload exceeds the MPI byte-count "
        "limit");
  return count * static_cast<int>(sizeof(Record));
}

template <typename Record>
void exchange_device_records(
    amrex::Gpu::DeviceVector<Record> &send_records,
    amrex::Gpu::DeviceVector<Record> &receive_records,
    const std::vector<int> &send_counts,
    const std::vector<int> &send_offsets,
    const std::vector<int> &receive_counts,
    const std::vector<int> &receive_offsets, const bool payload_uses_mpi,
    std::vector<MPI_Request> &requests, const char *const operation) {
  const int rank = amrex::ParallelDescriptor::MyProc();
  const int ranks = amrex::ParallelDescriptor::NProcs();
  if (send_counts.size() != static_cast<std::size_t>(ranks) ||
      send_offsets.size() != static_cast<std::size_t>(ranks) ||
      receive_counts.size() != static_cast<std::size_t>(ranks) ||
      receive_offsets.size() != static_cast<std::size_t>(ranks))
    throw std::runtime_error(
        "ParticleAHFinderX radial-probe exchange has an invalid rank layout");
  if (send_counts[rank] != receive_counts[rank])
    throw std::runtime_error(
        "ParticleAHFinderX process-local radial-probe route is not its own "
        "transpose");
  if (send_counts[rank] > 0)
    amrex::Gpu::copy(
        amrex::Gpu::deviceToDevice,
        send_records.begin() + send_offsets[rank],
        send_records.begin() + send_offsets[rank] + send_counts[rank],
        receive_records.begin() + receive_offsets[rank]);
  if (!payload_uses_mpi)
    return;

  amrex::Gpu::streamSynchronize();
  const int tag = amrex::ParallelDescriptor::SeqNum();
  requests.clear();
  requests.reserve(2 * static_cast<std::size_t>(ranks - 1));
  auto *const send_data =
      reinterpret_cast<unsigned char *>(send_records.dataPtr());
  auto *const receive_data =
      reinterpret_cast<unsigned char *>(receive_records.dataPtr());
  for (int peer = 0; peer < ranks; ++peer) {
    if (peer == rank || receive_counts[peer] == 0)
      continue;
    requests.push_back(MPI_REQUEST_NULL);
    check_mpi(
        MPI_Irecv(receive_data +
                      static_cast<std::size_t>(receive_offsets[peer]) *
                          sizeof(Record),
                  mpi_record_bytes<Record>(receive_counts[peer]), MPI_BYTE,
                  peer, tag, amrex::ParallelDescriptor::Communicator(),
                  &requests.back()),
        operation);
  }
  for (int peer = 0; peer < ranks; ++peer) {
    if (peer == rank || send_counts[peer] == 0)
      continue;
    requests.push_back(MPI_REQUEST_NULL);
    check_mpi(
        MPI_Isend(send_data +
                      static_cast<std::size_t>(send_offsets[peer]) *
                          sizeof(Record),
                  mpi_record_bytes<Record>(send_counts[peer]), MPI_BYTE, peer,
                  tag, amrex::ParallelDescriptor::Communicator(),
                  &requests.back()),
        operation);
  }
  if (!requests.empty())
    check_mpi(MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                          MPI_STATUSES_IGNORE),
              operation);
}

template <typename Record>
amrex::Long record_bytes(const int count) {
  return static_cast<amrex::Long>(count) *
         static_cast<amrex::Long>(sizeof(Record));
}

} // namespace

struct RadialProbeWorkspace {
  amrex::Gpu::DeviceVector<int> level_grid_offsets;
  amrex::Gpu::DeviceVector<int> grid_owner;
  amrex::Gpu::DeviceVector<int> grid_local_index;
  amrex::Gpu::DeviceVector<amrex::Box> metric_available;
  amrex::Gpu::DeviceVector<amrex::Box> curv_available;
  amrex::Gpu::DeviceVector<int> destination;
  amrex::Gpu::DeviceVector<int> rank_counts;
  amrex::Gpu::DeviceVector<int> rank_offsets;
  amrex::Gpu::DeviceVector<int> rank_cursors;
  amrex::Gpu::DeviceVector<int> arrival;
  amrex::Gpu::DeviceVector<int> response_status;
  amrex::Gpu::DeviceVector<RadialProbeRequest> request_send;
  amrex::Gpu::DeviceVector<RadialProbeRequest> request_receive;
  amrex::Gpu::DeviceVector<RadialProbeResponse> response_send;
  amrex::Gpu::DeviceVector<RadialProbeResponse> response_receive;
  std::vector<MPI_Request> mpi_requests;

  std::size_t allocated_bytes() const {
    return (level_grid_offsets.capacity() + grid_owner.capacity() +
            grid_local_index.capacity() + destination.capacity() +
            rank_counts.capacity() + rank_offsets.capacity() +
            rank_cursors.capacity() + arrival.capacity() +
            response_status.capacity()) *
               sizeof(int) +
           (metric_available.capacity() + curv_available.capacity()) *
               sizeof(amrex::Box) +
           (request_send.capacity() + request_receive.capacity()) *
               sizeof(RadialProbeRequest) +
           (response_send.capacity() + response_receive.capacity()) *
               sizeof(RadialProbeResponse);
  }
};

RadialGeometryCache::RadialGeometryCache()
    : probe_workspace_(std::make_unique<RadialProbeWorkspace>()) {}

RadialGeometryCache::~RadialGeometryCache() = default;

void RadialGeometryCache::prepare(
    const LogicalSurfaceDeviceView &logical,
    const std::vector<RadialCacheLayer> &layers, const int radial_nodes,
    const amrex::Real half_width_factor,
    const amrex::Real level_width_factor,
    const amrex::Real minimum_radius_factor,
    const amrex::Real maximum_radius_factor) {
  valid_ = false;
  if (radial_nodes < 4 || !(half_width_factor > 0) ||
      !(level_width_factor > 0) || !(level_width_factor <= 1) ||
      !(minimum_radius_factor > 0) ||
      !(maximum_radius_factor > minimum_radius_factor))
    amrex::Abort("ParticleAHFinderX received invalid radial-cache bounds");

  point_count_ = 0;
  for (std::size_t index = 0; index < layers.size(); ++index) {
    const auto &layer = layers[index];
    if (layer.stable_id <= 0 || layer.logical_offset < 0 ||
        layer.point_count <= 0 || layer.cache_offset != point_count_ ||
        layer.angular_level < 0 ||
        layer.logical_offset > logical.point_count - layer.point_count)
      amrex::Abort("ParticleAHFinderX received an invalid radial-cache "
                   "logical layer");
    if (index > 0 &&
        (layers[index - 1].stable_id > layer.stable_id ||
         (layers[index - 1].stable_id == layer.stable_id &&
          layers[index - 1].angular_level >= layer.angular_level)))
      amrex::Abort("ParticleAHFinderX radial-cache layers must be unique "
                   "and ordered");
    if (point_count_ > std::numeric_limits<amrex::Long>::max() -
                           layer.point_count)
      amrex::Abort("ParticleAHFinderX radial-cache point count overflowed");
    point_count_ += layer.point_count;
  }

  radial_nodes_ = radial_nodes;
  layers_.resize(layers.size());
  if (!layers.empty())
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, layers.begin(), layers.end(),
                     layers_.begin());
  logical_point_.resize(point_count_);
  lower_radius_.resize(point_count_);
  upper_radius_.resize(point_count_);
  point_valid_.resize(point_count_);
  const auto values_per_point = static_cast<std::size_t>(30) * radial_nodes_;
  if (static_cast<std::uintmax_t>(point_count_) >
      std::numeric_limits<std::size_t>::max() / values_per_point)
    amrex::Abort("ParticleAHFinderX radial-cache allocation overflowed");
  const auto geometry_values =
      values_per_point * static_cast<std::size_t>(point_count_);
  adm_.resize(geometry_values);

  if (point_count_ == 0) {
    valid_ = true;
    return;
  }

  const auto *const layer = layers_.dataPtr();
  const int layer_count = static_cast<int>(layers.size());
  auto *const logical_point = logical_point_.dataPtr();
  auto *const lower_radius = lower_radius_.dataPtr();
  auto *const upper_radius = upper_radius_.dataPtr();
  auto *const point_valid = point_valid_.dataPtr();
  amrex::ParallelFor(
      point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long cache_point) noexcept {
        int lower = 0;
        int upper = layer_count;
        while (lower + 1 < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (layer[middle].cache_offset <= cache_point)
            lower = middle;
          else
            upper = middle;
        }
        const auto &selected = layer[lower];
        const amrex::Long point =
            selected.logical_offset + cache_point - selected.cache_offset;
        logical_point[cache_point] = point;
        const auto &surface =
            logical.descriptors[logical.surface_ordinal[point]];
        amrex::Real layer_half_width_factor = half_width_factor;
        for (int level = 0; level < selected.angular_level; ++level)
          layer_half_width_factor *= level_width_factor;
        const amrex::Real width =
            layer_half_width_factor * surface.radius;
        const amrex::Real minimum_radius =
            minimum_radius_factor * surface.radius;
        const amrex::Real maximum_radius =
            maximum_radius_factor * surface.radius;
        lower_radius[cache_point] =
            amrex::max(minimum_radius, logical.height[point] - width);
        upper_radius[cache_point] =
            amrex::min(maximum_radius, logical.height[point] + width);
        point_valid[cache_point] =
            amrex::Math::isfinite(lower_radius[cache_point]) &&
                    amrex::Math::isfinite(upper_radius[cache_point]) &&
                    lower_radius[cache_point] <= logical.height[point] &&
                    logical.height[point] <= upper_radius[cache_point] &&
                    upper_radius[cache_point] > lower_radius[cache_point]
                ? 1
                : 0;
      });
  valid_ = true;
}

SurfaceRoutingStats RadialGeometryCache::sample_batched(
    const LogicalSurfaceDeviceView &logical,
    SurfaceParticleContainer &container,
    const std::vector<RadialCacheMeshLevel> &levels,
    const int minimum_active_level, const int maximum_active_level,
    const int interpolation_order) {
  if (!valid_ || radial_nodes_ < 4 || point_count_ < 0)
    throw std::runtime_error(
        "ParticleAHFinderX attempted to sample an invalid radial cache");
  if (minimum_active_level < 0 ||
      maximum_active_level <= minimum_active_level ||
      maximum_active_level > container.finestLevel() + 1 ||
      levels.size() != static_cast<std::size_t>(maximum_active_level -
                                                minimum_active_level) ||
      (interpolation_order != 1 && interpolation_order != 3))
    throw std::runtime_error(
        "ParticleAHFinderX radial-probe batch has an invalid active level "
        "layout");
  for (std::size_t index = 0; index < levels.size(); ++index) {
    const auto &level = levels[index];
    const int expected_level =
        minimum_active_level + static_cast<int>(index);
    if (level.level != expected_level || !level.metric || !level.curv ||
        level.metric_component < 0 || level.curv_component < 0 ||
        level.metric->nComp() < level.metric_component + 6 ||
        level.curv->nComp() < level.curv_component + 6 ||
        level.metric->boxArray().size() !=
            container.ParticleBoxArray(level.level).size() ||
        level.curv->boxArray().size() !=
            container.ParticleBoxArray(level.level).size() ||
        level.metric->DistributionMap() !=
            container.ParticleDistributionMap(level.level) ||
        level.curv->DistributionMap() !=
            container.ParticleDistributionMap(level.level))
      throw std::runtime_error(
          "ParticleAHFinderX radial-probe ADM and particle layouts differ");
  }

  SurfaceRoutingStats stats;
  // This is a communicator-wide transpose. A logical owner with no local
  // cache points can still own mesh boxes needed by another rank, and must
  // participate in the count and payload exchanges in the same order.
  if (point_count_ > std::numeric_limits<amrex::Long>::max() /
                         radial_nodes_)
    throw std::runtime_error(
        "ParticleAHFinderX radial-probe request count overflowed");
  const amrex::Long request_count_long = point_count_ * radial_nodes_;
  if (request_count_long > INT_MAX)
    throw std::runtime_error(
        "ParticleAHFinderX radial-probe batch exceeds INT_MAX requests");
  const int request_count = static_cast<int>(request_count_long);
  const int rank = amrex::ParallelDescriptor::MyProc();
  const int ranks = amrex::ParallelDescriptor::NProcs();
  auto &workspace = *probe_workspace_;

  if (!container.m_particle_locator.isValid(
          container.GetParGDB(), container.do_tiling, container.tile_size))
    container.m_particle_locator.build(container.GetParGDB(),
                                       container.do_tiling,
                                       container.tile_size);
  container.m_particle_locator.setGeometry(container.GetParGDB());
  const auto assign_grid = container.m_particle_locator.getGridAssignor();

  const int hierarchy_levels = container.finestLevel() + 1;
  std::vector<int> host_level_grid_offsets(
      static_cast<std::size_t>(hierarchy_levels + 1), 0);
  for (int lev = 0; lev < hierarchy_levels; ++lev) {
    const auto grids = container.ParticleBoxArray(lev).size();
    if (grids > static_cast<std::size_t>(INT_MAX) ||
        host_level_grid_offsets[lev] >
            INT_MAX - static_cast<int>(grids))
      throw std::runtime_error(
          "ParticleAHFinderX radial-probe mesh metadata overflowed");
    host_level_grid_offsets[lev + 1] =
        host_level_grid_offsets[lev] + static_cast<int>(grids);
  }
  const int grid_count = host_level_grid_offsets.back();
  std::vector<int> host_grid_owner(static_cast<std::size_t>(grid_count), -1);
  std::vector<int> host_grid_local_index(
      static_cast<std::size_t>(grid_count), -1);
  std::vector<amrex::Box> host_metric_available(
      static_cast<std::size_t>(grid_count));
  std::vector<amrex::Box> host_curv_available(
      static_cast<std::size_t>(grid_count));
  for (int lev = 0; lev < hierarchy_levels; ++lev) {
    const auto &distribution = container.ParticleDistributionMap(lev);
    const int offset = host_level_grid_offsets[lev];
    const int grids = static_cast<int>(container.ParticleBoxArray(lev).size());
    for (int grid = 0; grid < grids; ++grid)
      host_grid_owner[offset + grid] = distribution[grid];
  }
  for (const auto &level : levels) {
    const int offset = host_level_grid_offsets[level.level];
    const int grids = static_cast<int>(level.metric->boxArray().size());
    for (int grid = 0; grid < grids; ++grid) {
      const int local_index = level.metric->localindex(grid);
      const int curv_local_index = level.curv->localindex(grid);
      if (local_index != curv_local_index)
        throw std::runtime_error(
            "ParticleAHFinderX radial-probe ADM local indices differ");
      host_grid_local_index[offset + grid] = local_index;
      if (local_index >= 0) {
        host_metric_available[offset + grid] = level.metric->fabbox(grid);
        host_curv_available[offset + grid] = level.curv->fabbox(grid);
      }
    }
  }

  workspace.level_grid_offsets.resize(host_level_grid_offsets.size());
  workspace.grid_owner.resize(host_grid_owner.size());
  workspace.grid_local_index.resize(host_grid_local_index.size());
  workspace.metric_available.resize(host_metric_available.size());
  workspace.curv_available.resize(host_curv_available.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                   host_level_grid_offsets.begin(),
                   host_level_grid_offsets.end(),
                   workspace.level_grid_offsets.begin());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_grid_owner.begin(),
                   host_grid_owner.end(), workspace.grid_owner.begin());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                   host_grid_local_index.begin(),
                   host_grid_local_index.end(),
                   workspace.grid_local_index.begin());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                   host_metric_available.begin(),
                   host_metric_available.end(),
                   workspace.metric_available.begin());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_curv_available.begin(),
                   host_curv_available.end(),
                   workspace.curv_available.begin());

  workspace.destination.resize(request_count);
  workspace.rank_counts.resize(ranks);
  workspace.rank_offsets.resize(ranks);
  workspace.rank_cursors.resize(ranks);
  int *const destination = workspace.destination.dataPtr();
  int *const count_by_rank = workspace.rank_counts.dataPtr();
  const int *const level_grid_offsets =
      workspace.level_grid_offsets.dataPtr();
  const int *const grid_owner = workspace.grid_owner.dataPtr();
  const auto *const logical_point = logical_point_.dataPtr();
  const auto *const lower_radius = lower_radius_.dataPtr();
  const auto *const upper_radius = upper_radius_.dataPtr();
  const amrex::Long cache_points = point_count_;
  const int radial_nodes = radial_nodes_;
  amrex::ParallelFor(ranks,
                     [=] AMREX_GPU_DEVICE(const int peer) noexcept {
                       count_by_rank[peer] = 0;
                     });
  amrex::ParallelFor(
      request_count,
      [=] AMREX_GPU_DEVICE(const int request_index) noexcept {
        const int radial_node =
            static_cast<int>(request_index / cache_points);
        const amrex::Long cache_point =
            request_index - amrex::Long(radial_node) * cache_points;
        const amrex::Long point = logical_point[cache_point];
        const auto &surface =
            logical.descriptors[logical.surface_ordinal[point]];
        const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                                  (logical.itheta[point] + amrex::Real(0.5)) /
                                  surface.ntheta;
        const amrex::Real phi = 2 * amrex::Math::pi<amrex::Real>() *
                                (logical.iphi[point] + amrex::Real(0.5)) /
                                surface.nphi;
        const AngularGeometry basis = angular_geometry(theta, phi);
        const amrex::Real radius =
            lower_radius[cache_point] +
            (upper_radius[cache_point] - lower_radius[cache_point]) *
                radial_node / (radial_nodes - 1);
        RadialProbePosition probe{};
        for (int d = 0; d < 3; ++d)
          probe.value[d] = surface.center[d] + radius * basis.radial[d];
        const auto location = assign_grid(
            probe, minimum_active_level, maximum_active_level - 1, 0);
        const int grid = amrex::get<0>(location);
        const int level = amrex::get<2>(location);
        int owner = rank;
        if (grid >= 0 && level >= minimum_active_level &&
            level < maximum_active_level)
          owner = grid_owner[level_grid_offsets[level] + grid];
        if (owner < 0 || owner >= ranks)
          owner = rank;
        destination[request_index] = owner;
        amrex::Gpu::Atomic::AddNoRet(count_by_rank + owner, 1);
      });

  std::vector<int> send_counts(static_cast<std::size_t>(ranks), 0);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   workspace.rank_counts.begin(),
                   workspace.rank_counts.end(), send_counts.begin());
  const auto send_offsets = exclusive_offsets(send_counts);
  const int send_total = total_count(send_counts, send_offsets);
  if (send_total != request_count)
    throw std::runtime_error(
        "ParticleAHFinderX radial-probe packing lost a request");
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, send_offsets.begin(),
                   send_offsets.end(), workspace.rank_offsets.begin());
  int *const cursors = workspace.rank_cursors.dataPtr();
  const int *const offsets = workspace.rank_offsets.dataPtr();
  amrex::ParallelFor(ranks,
                     [=] AMREX_GPU_DEVICE(const int peer) noexcept {
                       cursors[peer] = 0;
                     });

  workspace.request_send.resize(send_total > 0 ? send_total : 1);
  auto *const request_records = workspace.request_send.dataPtr();
  amrex::ParallelFor(
      request_count,
      [=] AMREX_GPU_DEVICE(const int request_index) noexcept {
        const int radial_node =
            static_cast<int>(request_index / cache_points);
        const amrex::Long cache_point =
            request_index - amrex::Long(radial_node) * cache_points;
        const amrex::Long point = logical_point[cache_point];
        const auto &surface =
            logical.descriptors[logical.surface_ordinal[point]];
        const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                                  (logical.itheta[point] + amrex::Real(0.5)) /
                                  surface.ntheta;
        const amrex::Real phi = 2 * amrex::Math::pi<amrex::Real>() *
                                (logical.iphi[point] + amrex::Real(0.5)) /
                                surface.nphi;
        const AngularGeometry basis = angular_geometry(theta, phi);
        const amrex::Real radius =
            lower_radius[cache_point] +
            (upper_radius[cache_point] - lower_radius[cache_point]) *
                radial_node / (radial_nodes - 1);
        RadialProbePosition probe{};
        for (int d = 0; d < 3; ++d)
          probe.value[d] = surface.center[d] + radius * basis.radial[d];
        const auto location = assign_grid(
            probe, minimum_active_level, maximum_active_level - 1, 0);
        const int grid = amrex::get<0>(location);
        const int level = amrex::get<2>(location);
        const int peer = destination[request_index];
        const int slot =
            offsets[peer] + amrex::Gpu::Atomic::Add(cursors + peer, 1);
        auto &record = request_records[slot];
        record.cache_value_index = request_index;
        for (int d = 0; d < 3; ++d)
          record.position[d] = probe.value[d];
        record.level = level;
        record.grid = grid;
        record.status =
            grid >= 0 && level >= minimum_active_level &&
                    level < maximum_active_level
                ? static_cast<int>(RadialProbeStatus::valid)
                : static_cast<int>(RadialProbeStatus::outside_active_band);
      });

  std::vector<int> receive_counts(static_cast<std::size_t>(ranks), 0);
  check_mpi(MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                         receive_counts.data(), 1, MPI_INT,
                         amrex::ParallelDescriptor::Communicator()),
            "radial-probe count exchange");
  const auto receive_offsets = exclusive_offsets(receive_counts);
  const int receive_total = total_count(receive_counts, receive_offsets);
  int local_remote_payload = 0;
  for (int peer = 0; peer < ranks; ++peer)
    if (peer != rank &&
        (send_counts[peer] != 0 || receive_counts[peer] != 0)) {
      local_remote_payload = 1;
      break;
    }
  int any_remote_payload = local_remote_payload;
  if (ranks > 1)
    check_mpi(MPI_Allreduce(&local_remote_payload, &any_remote_payload, 1,
                            MPI_INT, MPI_MAX,
                            amrex::ParallelDescriptor::Communicator()),
              "radial-probe payload locality reduction");
  const bool payload_uses_mpi = any_remote_payload != 0;

  workspace.request_receive.resize(receive_total > 0 ? receive_total : 1);
  exchange_device_records(
      workspace.request_send, workspace.request_receive, send_counts,
      send_offsets, receive_counts, receive_offsets, payload_uses_mpi,
      workspace.mpi_requests, "radial-probe request exchange");

  workspace.response_send.resize(receive_total > 0 ? receive_total : 1);
  const auto *const received_requests = workspace.request_receive.dataPtr();
  auto *const response_records = workspace.response_send.dataPtr();
  amrex::ParallelFor(
      receive_total, [=] AMREX_GPU_DEVICE(const int index) noexcept {
        const auto &request = received_requests[index];
        auto &response = response_records[index];
        response.cache_value_index = request.cache_value_index;
        response.sampled_level = -1;
        response.status = request.status;
        for (int component = 0; component < 30; ++component)
          response.adm[component] = 0;
      });

  const int *const grid_local_index =
      workspace.grid_local_index.dataPtr();
  const auto *const metric_available =
      workspace.metric_available.dataPtr();
  const auto *const curv_available = workspace.curv_available.dataPtr();
  for (const auto &level : levels) {
    const int lev = level.level;
    const auto prob_lo = container.Geom(lev).ProbLoArray();
    const auto inv_dx = container.Geom(lev).InvCellSizeArray();
    const auto metric_arrays = level.metric->const_arrays();
    const auto curv_arrays = level.curv->const_arrays();
    const int metric_component = level.metric_component;
    const int curv_component = level.curv_component;
    const FieldCentering centering = level.centering;
    amrex::ParallelFor(
        receive_total, [=] AMREX_GPU_DEVICE(const int index) noexcept {
          const auto &request = received_requests[index];
          if (request.status != static_cast<int>(RadialProbeStatus::valid) ||
              request.level != lev)
            return;
          auto &response = response_records[index];
          const int flat_grid = level_grid_offsets[lev] + request.grid;
          const int local_index = grid_local_index[flat_grid];
          if (local_index < 0) {
            response.status =
                static_cast<int>(RadialProbeStatus::invalid_owner);
            return;
          }
          const amrex::GpuArray<amrex::Real, 3> position{{
              request.position[0], request.position[1], request.position[2]}};
          AdmInterpolationData data;
          const InterpolationStatus interpolation_status =
              interpolation_order == 3
                  ? tensor_cubic_interpolate_adm(
                        metric_arrays[local_index], metric_component,
                        curv_arrays[local_index], curv_component, position,
                        prob_lo, inv_dx, centering,
                        metric_available[flat_grid],
                        curv_available[flat_grid], data)
                  : trilinear_interpolate_adm(
                        metric_arrays[local_index], metric_component,
                        curv_arrays[local_index], curv_component, position,
                        prob_lo, inv_dx, centering,
                        metric_available[flat_grid],
                        curv_available[flat_grid], data);
          if (interpolation_status ==
              InterpolationStatus::stencil_unavailable) {
            response.status =
                static_cast<int>(RadialProbeStatus::stencil_unavailable);
            return;
          }
          if (interpolation_status == InterpolationStatus::nonfinite) {
            response.status =
                static_cast<int>(RadialProbeStatus::nonfinite);
            return;
          }
          for (int component = 0; component < 6; ++component) {
            response.adm[component] = data.gamma[component];
            response.adm[24 + component] = data.curv[component];
            for (int d = 0; d < 3; ++d)
              response.adm[6 + 6 * d + component] =
                  data.d_gamma[d][component];
          }
          response.sampled_level = lev;
          response.status = static_cast<int>(RadialProbeStatus::valid);
        });
  }

  workspace.response_receive.resize(send_total > 0 ? send_total : 1);
  exchange_device_records(
      workspace.response_send, workspace.response_receive, receive_counts,
      receive_offsets, send_counts, send_offsets, payload_uses_mpi,
      workspace.mpi_requests, "radial-probe response exchange");

  workspace.arrival.resize(request_count);
  workspace.response_status.resize(request_count);
  int *const arrival = workspace.arrival.dataPtr();
  int *const response_status = workspace.response_status.dataPtr();
  amrex::ParallelFor(
      request_count, [=] AMREX_GPU_DEVICE(const int index) noexcept {
        arrival[index] = 0;
        response_status[index] =
            static_cast<int>(RadialProbeStatus::invalid_owner);
      });
  const auto *const responses = workspace.response_receive.dataPtr();
  auto *const geometry = adm_.dataPtr();
  auto *const point_valid = point_valid_.dataPtr();
  amrex::ParallelFor(
      send_total, [=] AMREX_GPU_DEVICE(const int index) noexcept {
        const auto &response = responses[index];
        const amrex::Long value_index = response.cache_value_index;
        if (value_index < 0 || value_index >= request_count)
          return;
        amrex::Gpu::Atomic::AddNoRet(arrival + value_index, 1);
        amrex::Gpu::Atomic::Exch(response_status + value_index,
                                response.status);
        const int radial_node =
            static_cast<int>(value_index / cache_points);
        const amrex::Long cache_point =
            value_index - amrex::Long(radial_node) * cache_points;
        if (response.status != static_cast<int>(RadialProbeStatus::valid)) {
          amrex::Gpu::Atomic::Exch(point_valid + cache_point, 0);
          return;
        }
        for (int component = 0; component < 30; ++component)
          geometry[(amrex::Long(component) * radial_nodes + radial_node) *
                       cache_points +
                   cache_point] = response.adm[component];
      });

  amrex::Long missing = 0;
  amrex::Long duplicate = 0;
  amrex::Long invalid = 0;
  amrex::Long outside_active_band = 0;
  if (request_count > 0) {
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum>
        reduce_ops;
    amrex::ReduceData<amrex::Long, amrex::Long, amrex::Long, amrex::Long>
        reduce_data(reduce_ops);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    reduce_ops.eval(
        request_count, reduce_data,
        [=] AMREX_GPU_DEVICE(const int index) noexcept -> ReduceTuple {
          const int count = arrival[index];
          if (count != 1)
            return {count == 0, count > 1, amrex::Long(0), amrex::Long(0)};
          const auto status =
              static_cast<RadialProbeStatus>(response_status[index]);
          return {amrex::Long(0), amrex::Long(0),
                  status != RadialProbeStatus::valid,
                  status == RadialProbeStatus::outside_active_band};
        });
    const auto reduced = reduce_data.value();
    missing = amrex::get<0>(reduced);
    duplicate = amrex::get<1>(reduced);
    invalid = amrex::get<2>(reduced);
    outside_active_band = amrex::get<3>(reduced);
  }
  stats.sent = send_total;
  stats.received = send_total;
  stats.missing = missing;
  stats.duplicate = duplicate;
  stats.invalid = invalid;
  stats.outside_active_band = outside_active_band;

  const int local_forward = send_counts[rank];
  const int local_reverse = receive_counts[rank];
  stats.device_local_copy_bytes =
      record_bytes<RadialProbeRequest>(local_forward) +
      record_bytes<RadialProbeResponse>(local_reverse);
  if (payload_uses_mpi) {
    stats.device_payload_bytes_sent =
        record_bytes<RadialProbeRequest>(send_total - local_forward) +
        record_bytes<RadialProbeResponse>(receive_total - local_reverse);
    stats.device_payload_bytes_received =
        record_bytes<RadialProbeRequest>(receive_total - local_reverse) +
        record_bytes<RadialProbeResponse>(send_total - local_forward);
    stats.mpi_payload_exchange_rounds = 2;
    stats.mpi_payload_messages_sent =
        remote_peer_count(send_counts, rank) +
        remote_peer_count(receive_counts, rank);
    stats.mpi_payload_messages_received =
        stats.mpi_payload_messages_sent;
    stats.explicit_device_synchronizations = 2;
  }
#ifdef AMREX_USE_GPU
  stats.control_device_to_host_bytes =
      static_cast<amrex::Long>(ranks) * sizeof(int) +
      4 * sizeof(amrex::Long);
  stats.device_to_host_control_copies = 2;
  stats.control_host_to_device_bytes =
      static_cast<amrex::Long>(
          host_level_grid_offsets.size() + host_grid_owner.size() +
          host_grid_local_index.size() + send_offsets.size()) *
          sizeof(int) +
      static_cast<amrex::Long>(host_metric_available.size() +
                               host_curv_available.size()) *
          sizeof(amrex::Box);
  stats.host_to_device_control_copies = 6;
#endif
  stats.mpi_control_bytes_sent =
      (static_cast<amrex::Long>(ranks) + (ranks > 1)) * sizeof(int);
  stats.mpi_control_bytes_received = stats.mpi_control_bytes_sent;
  stats.mpi_count_collectives = 1 + (ranks > 1);
  return stats;
}

RadialCacheComparison RadialGeometryCache::compare_direct(
    const LogicalSurfaceDeviceView &logical) const {
  if (!valid_)
    amrex::Abort("ParticleAHFinderX attempted to compare an invalid radial "
                 "geometry cache");
  if (point_count_ == 0)
    return {};
  const auto *const logical_point = logical_point_.dataPtr();
  const auto *const lower_radius = lower_radius_.dataPtr();
  const auto *const upper_radius = upper_radius_.dataPtr();
  const auto *const geometry = adm_.dataPtr();
  const auto *const point_valid = point_valid_.dataPtr();
  const amrex::Long cache_points = point_count_;
  const amrex::Long logical_points = logical.point_count;
  const int radial_nodes = radial_nodes_;
  amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                   amrex::ReduceOpSum>
      reduce_ops;
  amrex::ReduceData<amrex::Real, amrex::Real, amrex::Long> reduce_data(
      reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  reduce_ops.eval(
      point_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const amrex::Long cache_point) noexcept
          -> ReduceTuple {
        const amrex::Long point = logical_point[cache_point];
        if (point_valid[cache_point] == 0)
          return {amrex::Real(0), amrex::Real(0), amrex::Long(1)};
        amrex::Real maximum_absolute_error = 0;
        amrex::Real maximum_scaled_error = 0;
        amrex::Long invalid = 0;
        for (int component = 0; component < 30; ++component) {
          amrex::Real cached = 0;
          const bool component_valid = radial_cubic_interpolate(
              geometry + amrex::Long(component) * radial_nodes *
                             cache_points +
                  cache_point,
              cache_points, radial_nodes, lower_radius[cache_point],
              upper_radius[cache_point], logical.height[point], cached);
          const amrex::Real direct =
              logical.adm[amrex::Long(component) * logical_points + point];
          if (!component_valid || !amrex::Math::isfinite(direct)) {
            ++invalid;
            continue;
          }
          const amrex::Real difference = cached - direct;
          const amrex::Real absolute_error =
              difference < 0 ? -difference : difference;
          const amrex::Real absolute_direct = direct < 0 ? -direct : direct;
          const amrex::Real scale =
              amrex::max(amrex::Real(1), absolute_direct);
          maximum_absolute_error =
              amrex::max(maximum_absolute_error, absolute_error);
          maximum_scaled_error =
              amrex::max(maximum_scaled_error, absolute_error / scale);
        }
        return {maximum_absolute_error, maximum_scaled_error, invalid};
      });
  const auto result = reduce_data.value();
  return {amrex::get<0>(result), amrex::get<1>(result),
          amrex::get<2>(result)};
}

void RadialGeometryCache::interpolate(
    const LogicalSurfaceDeviceView &logical) const {
  if (!valid_)
    amrex::Abort("ParticleAHFinderX attempted to use an invalid radial "
                 "geometry cache");
  const auto *const logical_point = logical_point_.dataPtr();
  const auto *const lower_radius = lower_radius_.dataPtr();
  const auto *const upper_radius = upper_radius_.dataPtr();
  const auto *const geometry = adm_.dataPtr();
  const auto *const point_valid = point_valid_.dataPtr();
  const amrex::Long cache_points = point_count_;
  const amrex::Long logical_points = logical.point_count;
  const int radial_nodes = radial_nodes_;
  amrex::ParallelFor(
      point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long cache_point) noexcept {
        const amrex::Long point = logical_point[cache_point];
        const auto &surface =
            logical.descriptors[logical.surface_ordinal[point]];
        if (surface.relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return;
        bool valid = point_valid[cache_point] != 0;
        for (int component = 0; component < 30 && valid; ++component) {
          amrex::Real value = 0;
          valid = radial_cubic_interpolate(
              geometry + amrex::Long(component) * radial_nodes *
                             cache_points +
                  cache_point,
              cache_points, radial_nodes, lower_radius[cache_point],
              upper_radius[cache_point], logical.height[point], value);
          if (valid)
            logical.adm[amrex::Long(component) * logical_points + point] =
                value;
        }
        logical.geometry_status[point] = valid ? 0 : 1;
      });
}

std::size_t RadialGeometryCache::allocated_bytes() const {
  return layers_.capacity() * sizeof(RadialCacheLayer) +
         logical_point_.capacity() * sizeof(amrex::Long) +
         (lower_radius_.capacity() + upper_radius_.capacity() +
          adm_.capacity()) *
             sizeof(amrex::Real) +
         point_valid_.capacity() * sizeof(int) +
         (probe_workspace_ ? probe_workspace_->allocated_bytes() : 0);
}

} // namespace ParticleAHFinderX
