/**
 * \file surface_manager.cxx
 * \brief Surface ownership and device-seed construction utilities.
 */
#include "surface_manager.hxx"

#include "candidate_events.hxx"

#include <cctk.h>

#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace ParticleAHFinderX {
namespace {

bool finite_center(const std::array<amrex::Real, 3> &center) {
  return amrex::Math::isfinite(center[0]) &&
         amrex::Math::isfinite(center[1]) &&
         amrex::Math::isfinite(center[2]);
}

bool external_center(const cGH *const cctkGH, const Runtime &state,
                     const SurfaceRecord &surface,
                     std::array<amrex::Real, 3> &center) {
  if (surface.configured_slot < 0 ||
      surface.configured_slot >= max_configured_surfaces)
    return false;
  const auto &provider = state.center_providers[surface.configured_slot];
  for (int d = 0; d < 3; ++d) {
    if (provider.variable_indices[d] < 0 || provider.array_indices[d] < 0)
      return false;
    const auto *const values = static_cast<const CCTK_REAL *>(
        CCTK_VarDataPtrI(cctkGH, 0, provider.variable_indices[d]));
    if (!values)
      return false;
    center[d] = values[provider.array_indices[d]];
  }
  return finite_center(center);
}

bool history_center(const SurfaceRecord &surface, const amrex::Real time,
                    const amrex::Real maximum_extrapolation_intervals,
                    std::array<amrex::Real, 3> &center) {
  const auto &history = surface.center_history;
  if (history.count <= 0)
    return false;
  const int latest = history.count - 1;
  center = history.centers[latest];
  if (!surface.predict_center || history.count < 2 ||
      maximum_extrapolation_intervals <= 0)
    return finite_center(center);

  const int previous = latest - 1;
  const amrex::Real interval =
      history.times[latest] - history.times[previous];
  if (!(interval > 0) || !amrex::Math::isfinite(interval))
    return finite_center(center);
  const amrex::Real target =
      amrex::min(time, history.times[latest] +
                           maximum_extrapolation_intervals * interval);

  if (history.count == 3) {
    const amrex::Real t0 = history.times[0];
    const amrex::Real t1 = history.times[1];
    const amrex::Real t2 = history.times[2];
    const amrex::Real d0 = (t0 - t1) * (t0 - t2);
    const amrex::Real d1 = (t1 - t0) * (t1 - t2);
    const amrex::Real d2 = (t2 - t0) * (t2 - t1);
    if (d0 != 0 && d1 != 0 && d2 != 0) {
      const amrex::Real w0 = (target - t1) * (target - t2) / d0;
      const amrex::Real w1 = (target - t0) * (target - t2) / d1;
      const amrex::Real w2 = (target - t0) * (target - t1) / d2;
      for (int d = 0; d < 3; ++d)
        center[d] = w0 * history.centers[0][d] +
                    w1 * history.centers[1][d] +
                    w2 * history.centers[2][d];
      if (finite_center(center))
        return true;
      center = history.centers[latest];
    }
  }

  const amrex::Real fraction =
      (target - history.times[latest]) / interval;
  for (int d = 0; d < 3; ++d)
    center[d] = history.centers[latest][d] +
                fraction * (history.centers[latest][d] -
                            history.centers[previous][d]);
  return finite_center(center);
}

bool parent_center(const std::vector<SurfaceRecord> &surfaces,
                   const SurfaceRecord &surface,
                   std::array<amrex::Real, 3> &center) {
  const auto find_parent = [&](const amrex::Long stable_id) {
    return std::lower_bound(
        surfaces.begin(), surfaces.end(), stable_id,
        [](const SurfaceRecord &record, const amrex::Long id) {
          return record.stable_id < id;
        });
  };
  const auto first = find_parent(surface.parent_ids[0]);
  const auto second = find_parent(surface.parent_ids[1]);
  if (first == surfaces.end() || second == surfaces.end() ||
      first->stable_id != surface.parent_ids[0] ||
      second->stable_id != surface.parent_ids[1] ||
      first->lifecycle == ParticleLifecycle::retired ||
      second->lifecycle == ParticleLifecycle::retired)
    return false;
  const amrex::Real first_weight = first->mass_scale;
  const amrex::Real second_weight = second->mass_scale;
  const amrex::Real total_weight = first_weight + second_weight;
  if (!(total_weight > 0))
    return false;
  for (int d = 0; d < 3; ++d)
    center[d] = (first_weight * first->centroid[d] +
                 second_weight * second->centroid[d]) /
                total_weight;
  return finite_center(center);
}

} // namespace

void initialize_single_angular_layer(SurfaceRecord &surface) {
  initialize_angular_layers(surface, false, surface.ntheta, 1);
}

void initialize_angular_layers(SurfaceRecord &surface, const bool enable,
                               const int minimum_ntheta,
                               const int maximum_levels) {
  if (surface.first_particle_id <= 0 || surface.particle_count <= 0 ||
      surface.ntheta <= 0 || surface.nphi <= 0)
    amrex::Abort("ParticleAHFinderX cannot initialize an invalid angular "
                 "surface layer");
  if (minimum_ntheta < 8 || maximum_levels <= 0 ||
      maximum_levels > max_angular_layers)
    amrex::Abort("ParticleAHFinderX received invalid angular continuation "
                 "limits");

  std::array<std::array<int, 2>, max_angular_layers> reversed{};
  int levels = 1;
  reversed[0] = {{surface.ntheta, surface.nphi}};
  while (enable && levels < maximum_levels) {
    const int previous_ntheta = reversed[levels - 1][0];
    const int previous_nphi = reversed[levels - 1][1];
    if (previous_ntheta % 2 != 0 || previous_nphi % 2 != 0 ||
        previous_ntheta / 2 < minimum_ntheta)
      break;
    reversed[levels] = {{previous_ntheta / 2, previous_nphi / 2}};
    ++levels;
  }

  surface.angular_levels = levels;
  surface.active_angular_level = 0;
  const amrex::Long first_particle_id = surface.first_particle_id;
  amrex::Long next_particle_id = first_particle_id;
  for (int level = 0; level < levels; ++level) {
    const auto resolution = reversed[levels - level - 1];
    auto &layer = surface.angular_layer[level];
    layer = AngularLayerRecord{};
    layer.first_particle_id = next_particle_id;
    layer.ntheta = resolution[0];
    layer.nphi = resolution[1];
    layer.particle_count = amrex::Long(layer.ntheta) * layer.nphi;
    layer.relaxation_state = SurfaceRelaxationState::idle;
    if (next_particle_id > std::numeric_limits<amrex::Long>::max() -
                               layer.particle_count)
      amrex::Abort("ParticleAHFinderX angular layer particle ID overflowed");
    next_particle_id += layer.particle_count;
  }
  for (int level = levels; level < max_angular_layers; ++level)
    surface.angular_layer[level] = AngularLayerRecord{};

  const auto &target = surface.angular_layer[levels - 1];
  surface.first_particle_id = target.first_particle_id;
  surface.particle_count = target.particle_count;
  surface.ntheta = target.ntheta;
  surface.nphi = target.nphi;
}

amrex::Long total_angular_particle_count(const SurfaceRecord &surface) {
  if (surface.angular_levels <= 0 ||
      surface.angular_levels > max_angular_layers)
    amrex::Abort("ParticleAHFinderX surface has an invalid angular layer "
                 "count");
  amrex::Long total = 0;
  for (int level = 0; level < surface.angular_levels; ++level) {
    const auto count = surface.angular_layer[level].particle_count;
    if (count <= 0 ||
        total > std::numeric_limits<amrex::Long>::max() - count)
      amrex::Abort("ParticleAHFinderX angular layer population overflowed");
    total += count;
  }
  return total;
}

std::vector<SurfaceSeed>
particle_seeds(const std::vector<SurfaceRecord> &surfaces) {
  std::vector<SurfaceSeed> seeds;
  for (const auto &surface : surfaces) {
    if (surface.lifecycle == ParticleLifecycle::retired)
      continue;
    for (int level = 0; level < surface.angular_levels; ++level) {
      const auto &layer = surface.angular_layer[level];
      SurfaceSeed seed;
      seed.surface_id = surface.stable_id;
      seed.first_particle_id = layer.first_particle_id;
      seed.logical_owner_offset = layer.logical_owner_offset;
      seed.generation = surface.generation;
      seed.angular_level = level;
      seed.angular_levels = surface.angular_levels;
      seed.logical_owner_rank = surface.logical_owner_rank;
      seed.ntheta = layer.ntheta;
      seed.nphi = layer.nphi;
      seed.center = surface.center;
      seed.radius = surface.radius;
      seed.lifecycle = surface.lifecycle;
      seed.role = surface.role;
      if (seed.particle_count() != layer.particle_count)
        amrex::Abort("ParticleAHFinderX angular layer metadata are "
                     "inconsistent");
      seeds.push_back(seed);
    }
  }
  return seeds;
}

bool surface_fits_physical_domain(
    const SurfaceRecord &surface,
    const std::array<amrex::Real, 3> &center,
    const PhysicalDomainBounds &domain) {
  if (!finite_center(center))
    return false;
  const amrex::Real radial_extent =
      amrex::max(surface.radius, surface.maximum_radius);
  if (!(radial_extent > 0) || !amrex::Math::isfinite(radial_extent))
    return false;
  for (int d = 0; d < 3; ++d) {
    if (domain.periodic[d])
      continue;
    if (!amrex::Math::isfinite(domain.lower[d]) ||
        !amrex::Math::isfinite(domain.upper[d]) ||
        !(domain.lower[d] < domain.upper[d]) ||
        center[d] - radial_extent < domain.lower[d] ||
        center[d] + radial_extent > domain.upper[d])
      return false;
  }
  return true;
}

namespace {

struct SpatialSurfacePopulations {
  std::vector<amrex::Long> rank_surface_counts;
  std::vector<amrex::Long> surface_counts;
};

SpatialSurfacePopulations gather_spatial_surface_populations(
    Runtime &state, const std::vector<std::size_t> &surface_indices) {
  const int rank_count = amrex::ParallelDescriptor::NProcs();
  if (surface_indices.empty())
    return {};
  if (surface_indices.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX live surface count exceeds the MPI count "
                 "range");
  if (static_cast<std::size_t>(rank_count) >
      std::numeric_limits<std::size_t>::max() / surface_indices.size())
    amrex::Abort("ParticleAHFinderX spatial ownership table size overflowed");

  std::vector<SurfaceLevelRequest> requests;
  requests.reserve(surface_indices.size());
  for (const auto index : surface_indices) {
    if (index >= state.surfaces.size() ||
        state.surfaces[index].lifecycle == ParticleLifecycle::retired ||
        state.surfaces[index].stable_id <= 0 ||
        (!requests.empty() &&
         requests.back().stable_id >= state.surfaces[index].stable_id))
      amrex::Abort("ParticleAHFinderX spatial ownership surfaces must be "
                   "live and ordered by stable ID");
    requests.push_back({state.surfaces[index].stable_id, -1});
  }

  std::vector<amrex::Long> local_counts(surface_indices.size(), 0);
  for (auto &container : state.containers) {
    const auto local = container->local_surface_level_ranges(requests);
    if (local.size() != local_counts.size())
      amrex::Abort("ParticleAHFinderX received an invalid spatial ownership "
                   "count table");
    for (std::size_t surface = 0; surface < local.size(); ++surface) {
      if (local[surface].stable_id != requests[surface].stable_id ||
          local[surface].particle_count < 0 ||
          local_counts[surface] >
              std::numeric_limits<amrex::Long>::max() -
                  local[surface].particle_count)
        amrex::Abort("ParticleAHFinderX spatial ownership counts are "
                     "inconsistent");
      local_counts[surface] += local[surface].particle_count;
    }
  }

  SpatialSurfacePopulations populations;
  populations.rank_surface_counts.resize(
      static_cast<std::size_t>(rank_count) * surface_indices.size(), 0);
  const int surface_count = static_cast<int>(surface_indices.size());
  const int mpi_error = MPI_Allgather(
      local_counts.data(), surface_count,
      amrex::ParallelDescriptor::Mpi_typemap<amrex::Long>::type(),
      populations.rank_surface_counts.data(), surface_count,
      amrex::ParallelDescriptor::Mpi_typemap<amrex::Long>::type(),
      amrex::ParallelDescriptor::Communicator());
  if (mpi_error != MPI_SUCCESS)
    amrex::Abort("ParticleAHFinderX spatial ownership MPI_Allgather failed");

  populations.surface_counts.assign(surface_indices.size(), 0);
  for (std::size_t surface = 0; surface < surface_indices.size(); ++surface) {
    for (int owner = 0; owner < rank_count; ++owner) {
      const amrex::Long count =
          populations.rank_surface_counts[
              static_cast<std::size_t>(owner) * surface_indices.size() +
              surface];
      if (count < 0 ||
          populations.surface_counts[surface] >
              std::numeric_limits<amrex::Long>::max() - count)
        amrex::Abort("ParticleAHFinderX spatial ownership particle count "
                     "overflowed");
      populations.surface_counts[surface] += count;
    }
    if (populations.surface_counts[surface] !=
        total_angular_particle_count(state.surfaces[surface_indices[surface]]))
      amrex::Abort("ParticleAHFinderX spatial ownership did not find every "
                   "persistent surface particle");
  }
  return populations;
}

void remap_and_validate_logical_ownership(Runtime &state) {
  const auto ownership = logical_ownership(state.surfaces);
  std::vector<amrex::Long> layer_counts(ownership.size(), 0);
  for (auto &container : state.containers) {
    const auto local = container->remap_logical_ownership(ownership);
    if (local.size() != layer_counts.size())
      amrex::Abort("ParticleAHFinderX logical ownership remap returned an "
                   "invalid count table");
    for (std::size_t layer = 0; layer < local.size(); ++layer) {
      if (local[layer] < 0 ||
          layer_counts[layer] >
              std::numeric_limits<amrex::Long>::max() - local[layer])
        amrex::Abort("ParticleAHFinderX logical ownership layer count "
                     "overflowed");
      layer_counts[layer] += local[layer];
    }
  }
  if (layer_counts.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX logical ownership layer count exceeds "
                 "the MPI count range");
  if (!layer_counts.empty())
    amrex::ParallelDescriptor::ReduceLongSum(
        layer_counts.data(), static_cast<int>(layer_counts.size()));
  for (std::size_t layer = 0; layer < ownership.size(); ++layer) {
    const auto found = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(),
        ownership[layer].surface_id,
        [](const SurfaceRecord &candidate, const amrex::Long stable_id) {
          return candidate.stable_id < stable_id;
        });
    if (found == state.surfaces.end() ||
        found->stable_id != ownership[layer].surface_id ||
        ownership[layer].angular_level < 0 ||
        ownership[layer].angular_level >= found->angular_levels ||
        layer_counts[layer] !=
            found->angular_layer[ownership[layer].angular_level]
                .particle_count)
      amrex::Abort("ParticleAHFinderX logical ownership remap corrupted a "
                   "persistent angular layer");
  }
}

} // namespace

void assign_logical_owners(std::vector<SurfaceRecord> &surfaces) {
  const int num_owners = amrex::ParallelDescriptor::NProcs();
  if (num_owners <= 0)
    amrex::Abort("ParticleAHFinderX found no logical surface owners");
  std::vector<amrex::Long> load(num_owners, 0);
  for (auto &surface : surfaces) {
    if (surface.lifecycle == ParticleLifecycle::retired) {
      surface.logical_owner_rank = -1;
      surface.logical_owner_offset = -1;
      continue;
    }
    int owner = 0;
    for (int candidate = 1; candidate < num_owners; ++candidate)
      if (load[candidate] < load[owner])
        owner = candidate;
    surface.logical_owner_rank = owner;
    const amrex::Long count = total_angular_particle_count(surface);
    if (load[owner] > std::numeric_limits<amrex::Long>::max() - count)
      amrex::Abort("ParticleAHFinderX logical owner load overflowed");
    amrex::Long offset = load[owner];
    for (int level = 0; level < surface.angular_levels; ++level) {
      surface.angular_layer[level].logical_owner_offset = offset;
      offset += surface.angular_layer[level].particle_count;
    }
    surface.logical_owner_offset =
        surface.angular_layer[surface.angular_levels - 1]
            .logical_owner_offset;
    load[owner] += count;
  }
}

LogicalOwnerAssignmentStats assign_spatial_logical_owners(Runtime &state) {
  const int rank_count = amrex::ParallelDescriptor::NProcs();
  if (rank_count <= 0)
    amrex::Abort("ParticleAHFinderX found no logical surface owners");

  std::vector<std::size_t> live_surfaces;
  live_surfaces.reserve(state.surfaces.size());
  for (std::size_t index = 0; index < state.surfaces.size(); ++index) {
    auto &surface = state.surfaces[index];
    if (surface.lifecycle == ParticleLifecycle::retired) {
      surface.logical_owner_rank = -1;
      surface.logical_owner_offset = -1;
      continue;
    }
    if (surface.stable_id <= 0 ||
        (!live_surfaces.empty() &&
         state.surfaces[live_surfaces.back()].stable_id >= surface.stable_id))
      amrex::Abort("ParticleAHFinderX live surfaces are not ordered by stable "
                   "ID during logical ownership assignment");
    live_surfaces.push_back(index);
  }

  LogicalOwnerAssignmentStats stats;
  if (live_surfaces.empty()) {
    rebuild_logical_offsets(state.surfaces);
    return stats;
  }
  const auto populations =
      gather_spatial_surface_populations(state, live_surfaces);
  const auto &surface_particles = populations.surface_counts;
  const auto &rank_surface_counts = populations.rank_surface_counts;
  for (std::size_t surface = 0; surface < live_surfaces.size(); ++surface) {
    if (stats.particle_count >
        std::numeric_limits<amrex::Long>::max() -
            surface_particles[surface])
      amrex::Abort("ParticleAHFinderX spatial ownership total overflowed");
    stats.particle_count += surface_particles[surface];
  }

  std::vector<std::size_t> assignment_order(live_surfaces.size());
  std::iota(assignment_order.begin(), assignment_order.end(), 0);
  std::sort(assignment_order.begin(), assignment_order.end(),
            [&](const std::size_t first, const std::size_t second) {
              if (surface_particles[first] != surface_particles[second])
                return surface_particles[first] > surface_particles[second];
              return state.surfaces[live_surfaces[first]].stable_id <
                     state.surfaces[live_surfaces[second]].stable_id;
            });

  std::vector<amrex::Long> owner_load(rank_count, 0);
  for (const std::size_t surface : assignment_order) {
    const amrex::Long count = surface_particles[surface];
    int selected_owner = -1;
    amrex::Long selected_score = 0;
    for (int owner = 0; owner < rank_count; ++owner) {
      const amrex::Long local =
          rank_surface_counts[static_cast<std::size_t>(owner) *
                                  live_surfaces.size() +
                              surface];
      if (local < 0 || local > count)
        amrex::Abort("ParticleAHFinderX spatial ownership found an invalid "
                     "rank-local population");
      const amrex::Long remote = count - local;
      if (remote >
          (std::numeric_limits<amrex::Long>::max() - owner_load[owner]) /
              rank_count)
        amrex::Abort("ParticleAHFinderX spatial ownership score overflowed");
      const amrex::Long score =
          remote * static_cast<amrex::Long>(rank_count) + owner_load[owner];
      if (selected_owner < 0 || score < selected_score ||
          (score == selected_score &&
           (owner_load[owner] < owner_load[selected_owner] ||
            (owner_load[owner] == owner_load[selected_owner] &&
             owner < selected_owner)))) {
        selected_owner = owner;
        selected_score = score;
      }
    }
    if (owner_load[selected_owner] >
        std::numeric_limits<amrex::Long>::max() - count)
      amrex::Abort("ParticleAHFinderX spatial logical-owner load overflowed");
    state.surfaces[live_surfaces[surface]].logical_owner_rank =
        selected_owner;
    owner_load[selected_owner] += count;
    stats.colocated_particles +=
        rank_surface_counts[static_cast<std::size_t>(selected_owner) *
                                live_surfaces.size() +
                            surface];
  }
  stats.maximum_owner_load =
      *std::max_element(owner_load.begin(), owner_load.end());
  rebuild_logical_offsets(state.surfaces);
  remap_and_validate_logical_ownership(state);
  return stats;
}

LogicalOwnerAssignmentStats assign_new_spatial_logical_owners(
    Runtime &state, const std::vector<amrex::Long> &new_surface_ids) {
  const int rank_count = amrex::ParallelDescriptor::NProcs();
  if (rank_count <= 0)
    amrex::Abort("ParticleAHFinderX found no logical surface owners");
  LogicalOwnerAssignmentStats stats;
  if (new_surface_ids.empty())
    return stats;

  std::vector<std::size_t> new_surfaces;
  new_surfaces.reserve(new_surface_ids.size());
  for (std::size_t index = 0; index < new_surface_ids.size(); ++index) {
    if (new_surface_ids[index] <= 0 ||
        (index > 0 &&
         new_surface_ids[index - 1] >= new_surface_ids[index]))
      amrex::Abort("ParticleAHFinderX new spatial owner IDs must be unique "
                   "and ordered");
    const auto found = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), new_surface_ids[index],
        [](const SurfaceRecord &surface, const amrex::Long stable_id) {
          return surface.stable_id < stable_id;
        });
    if (found == state.surfaces.end() ||
        found->stable_id != new_surface_ids[index] ||
        found->lifecycle == ParticleLifecycle::retired)
      amrex::Abort("ParticleAHFinderX new spatial owner ID is not a live "
                   "surface");
    new_surfaces.push_back(
        static_cast<std::size_t>(found - state.surfaces.begin()));
  }

  struct PreservedOwnerLayout {
    std::size_t surface_index = 0;
    int owner_rank = -1;
    std::array<amrex::Long, max_angular_layers> layer_offsets{};
  };
  std::vector<PreservedOwnerLayout> preserved;
  std::vector<amrex::Long> owner_load(rank_count, 0);
  for (std::size_t index = 0; index < state.surfaces.size(); ++index) {
    const auto &surface = state.surfaces[index];
    if (surface.lifecycle == ParticleLifecycle::retired ||
        std::binary_search(new_surface_ids.begin(), new_surface_ids.end(),
                           surface.stable_id))
      continue;
    if (surface.logical_owner_rank < 0 ||
        surface.logical_owner_rank >= rank_count)
      amrex::Abort("ParticleAHFinderX existing logical owner is invalid at "
                   "dynamic birth");
    PreservedOwnerLayout layout;
    layout.surface_index = index;
    layout.owner_rank = surface.logical_owner_rank;
    for (int level = 0; level < surface.angular_levels; ++level)
      layout.layer_offsets[level] =
          surface.angular_layer[level].logical_owner_offset;
    preserved.push_back(layout);

    const amrex::Long begin =
        surface.angular_layer[0].logical_owner_offset;
    const amrex::Long count = total_angular_particle_count(surface);
    if (begin < 0 ||
        begin > std::numeric_limits<amrex::Long>::max() - count)
      amrex::Abort("ParticleAHFinderX existing logical owner load overflowed "
                   "at dynamic birth");
    owner_load[surface.logical_owner_rank] =
        amrex::max(owner_load[surface.logical_owner_rank], begin + count);
  }

  const auto populations =
      gather_spatial_surface_populations(state, new_surfaces);
  std::vector<std::size_t> assignment_order(new_surfaces.size());
  std::iota(assignment_order.begin(), assignment_order.end(), 0);
  std::sort(assignment_order.begin(), assignment_order.end(),
            [&](const std::size_t first, const std::size_t second) {
              if (populations.surface_counts[first] !=
                  populations.surface_counts[second])
                return populations.surface_counts[first] >
                       populations.surface_counts[second];
              return state.surfaces[new_surfaces[first]].stable_id <
                     state.surfaces[new_surfaces[second]].stable_id;
            });

  for (const std::size_t surface : assignment_order) {
    const amrex::Long count = populations.surface_counts[surface];
    if (stats.particle_count >
        std::numeric_limits<amrex::Long>::max() - count)
      amrex::Abort("ParticleAHFinderX dynamic spatial ownership total "
                   "overflowed");
    stats.particle_count += count;
    int selected_owner = -1;
    amrex::Long selected_score = 0;
    for (int owner = 0; owner < rank_count; ++owner) {
      const amrex::Long local =
          populations.rank_surface_counts[
              static_cast<std::size_t>(owner) * new_surfaces.size() +
              surface];
      if (local < 0 || local > count)
        amrex::Abort("ParticleAHFinderX dynamic spatial ownership found an "
                     "invalid rank-local population");
      const amrex::Long remote = count - local;
      if (remote >
          (std::numeric_limits<amrex::Long>::max() - owner_load[owner]) /
              rank_count)
        amrex::Abort("ParticleAHFinderX dynamic spatial ownership score "
                     "overflowed");
      const amrex::Long score =
          remote * static_cast<amrex::Long>(rank_count) + owner_load[owner];
      if (selected_owner < 0 || score < selected_score ||
          (score == selected_score &&
           (owner_load[owner] < owner_load[selected_owner] ||
            (owner_load[owner] == owner_load[selected_owner] &&
             owner < selected_owner)))) {
        selected_owner = owner;
        selected_score = score;
      }
    }
    if (owner_load[selected_owner] >
        std::numeric_limits<amrex::Long>::max() - count)
      amrex::Abort("ParticleAHFinderX dynamic spatial owner load overflowed");
    state.surfaces[new_surfaces[surface]].logical_owner_rank = selected_owner;
    owner_load[selected_owner] += count;
    stats.colocated_particles +=
        populations.rank_surface_counts[
            static_cast<std::size_t>(selected_owner) * new_surfaces.size() +
            surface];
  }
  stats.maximum_owner_load =
      *std::max_element(owner_load.begin(), owner_load.end());

  rebuild_logical_offsets(state.surfaces);
  for (const auto &layout : preserved) {
    const auto &surface = state.surfaces[layout.surface_index];
    if (surface.logical_owner_rank != layout.owner_rank)
      amrex::Abort("ParticleAHFinderX dynamic birth migrated an existing "
                   "logical owner");
    for (int level = 0; level < surface.angular_levels; ++level)
      if (surface.angular_layer[level].logical_owner_offset !=
          layout.layer_offsets[level])
        amrex::Abort("ParticleAHFinderX dynamic birth changed an existing "
                     "logical owner offset");
  }
  remap_and_validate_logical_ownership(state);
  return stats;
}

void assign_new_logical_owner(
    SurfaceRecord &surface, const std::vector<SurfaceRecord> &surfaces) {
  const int num_owners = amrex::ParallelDescriptor::NProcs();
  if (num_owners <= 0)
    amrex::Abort("ParticleAHFinderX found no logical surface owners");
  std::vector<amrex::Long> load(num_owners, 0);
  for (const auto &existing : surfaces) {
    if (existing.lifecycle == ParticleLifecycle::retired)
      continue;
    if (existing.logical_owner_rank < 0 ||
        existing.logical_owner_rank >= num_owners ||
        existing.logical_owner_offset < 0)
      amrex::Abort("ParticleAHFinderX existing logical ownership is invalid "
                   "at dynamic birth");
    const amrex::Long count = total_angular_particle_count(existing);
    const amrex::Long begin =
        existing.angular_layer[0].logical_owner_offset;
    if (begin < 0 ||
        begin > std::numeric_limits<amrex::Long>::max() - count)
      amrex::Abort("ParticleAHFinderX existing logical owner load "
                   "overflowed at dynamic birth");
    load[existing.logical_owner_rank] = amrex::max(
        load[existing.logical_owner_rank],
        begin + count);
  }
  int owner = 0;
  for (int candidate = 1; candidate < num_owners; ++candidate)
    if (load[candidate] < load[owner])
      owner = candidate;
  const amrex::Long count = total_angular_particle_count(surface);
  if (load[owner] > std::numeric_limits<amrex::Long>::max() - count)
    amrex::Abort("ParticleAHFinderX dynamic logical owner load overflowed");
  surface.logical_owner_rank = owner;
  amrex::Long offset = load[owner];
  for (int level = 0; level < surface.angular_levels; ++level) {
    surface.angular_layer[level].logical_owner_offset = offset;
    offset += surface.angular_layer[level].particle_count;
  }
  surface.logical_owner_offset =
      surface.angular_layer[surface.angular_levels - 1].logical_owner_offset;
}

void rebuild_logical_offsets(std::vector<SurfaceRecord> &surfaces) {
  const int num_owners = amrex::ParallelDescriptor::NProcs();
  std::vector<amrex::Long> offsets(num_owners, 0);
  for (auto &surface : surfaces) {
    if (surface.lifecycle == ParticleLifecycle::retired) {
      surface.logical_owner_rank = -1;
      surface.logical_owner_offset = -1;
      continue;
    }
    const amrex::Long count = total_angular_particle_count(surface);
    if (surface.logical_owner_rank < 0 ||
        surface.logical_owner_rank >= num_owners ||
        offsets[surface.logical_owner_rank] >
            std::numeric_limits<amrex::Long>::max() - count)
      amrex::Abort("ParticleAHFinderX cannot rebuild logical owner offsets");
    amrex::Long offset = offsets[surface.logical_owner_rank];
    for (int level = 0; level < surface.angular_levels; ++level) {
      surface.angular_layer[level].logical_owner_offset = offset;
      offset += surface.angular_layer[level].particle_count;
    }
    surface.logical_owner_offset =
        surface.angular_layer[surface.angular_levels - 1]
            .logical_owner_offset;
    offsets[surface.logical_owner_rank] += count;
  }
}

void retire_surface_populations(
    Runtime &state, const std::vector<amrex::Long> &stable_ids) {
  if (stable_ids.empty())
    return;
  amrex::Long expected = 0;
  for (std::size_t index = 0; index < stable_ids.size(); ++index) {
    if (stable_ids[index] <= 0 ||
        (index > 0 && stable_ids[index - 1] >= stable_ids[index]))
      amrex::Abort("ParticleAHFinderX retirement IDs must be unique and "
                   "increasing");
    const auto surface = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), stable_ids[index],
        [](const SurfaceRecord &record, const amrex::Long stable_id) {
          return record.stable_id < stable_id;
        });
    if (surface == state.surfaces.end() ||
        surface->stable_id != stable_ids[index] ||
        surface->lifecycle == ParticleLifecycle::retired)
      amrex::Abort("ParticleAHFinderX attempted to retire an unknown or "
                   "already retired surface");
    const amrex::Long count = total_angular_particle_count(*surface);
    if (expected > std::numeric_limits<amrex::Long>::max() - count)
      amrex::Abort("ParticleAHFinderX retirement particle count overflowed");
    expected += count;
    surface->lifecycle = ParticleLifecycle::retired;
    surface->search_every = 0;
  }

  amrex::Long retired = 0;
  for (auto &container : state.containers)
    retired += container->retire_surfaces(stable_ids);
  amrex::ParallelDescriptor::ReduceLongSum(retired);
  if (retired != expected || state.global_particle_count < retired)
    amrex::Abort("ParticleAHFinderX retirement did not remove the expected "
                 "persistent particle population");
  state.global_particle_count -= retired;

  rebuild_logical_offsets(state.surfaces);
  state.logical_surfaces->compact(local_logical_seeds(state.surfaces));
  const auto ownership = logical_ownership(state.surfaces);
  std::vector<amrex::Long> counts(ownership.size(), 0);
  for (auto &container : state.containers) {
    const auto local = container->remap_logical_ownership(ownership);
    for (std::size_t surface = 0; surface < counts.size(); ++surface)
      counts[surface] += local[surface];
  }
  if (!counts.empty()) {
    if (counts.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
      amrex::Abort("ParticleAHFinderX retirement validation exceeds the MPI "
                   "count range");
    amrex::ParallelDescriptor::ReduceLongSum(
        counts.data(), static_cast<int>(counts.size()));
  }
  for (std::size_t surface = 0; surface < ownership.size(); ++surface) {
    const auto record = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(),
        ownership[surface].surface_id,
        [](const SurfaceRecord &candidate, const amrex::Long stable_id) {
          return candidate.stable_id < stable_id;
        });
    if (record == state.surfaces.end() ||
        record->stable_id != ownership[surface].surface_id ||
        ownership[surface].angular_level < 0 ||
        ownership[surface].angular_level >= record->angular_levels ||
        counts[surface] !=
            record->angular_layer[ownership[surface].angular_level]
                .particle_count)
      amrex::Abort("ParticleAHFinderX retirement corrupted a live surface "
                   "population");
  }
  amrex::Long logical_count = state.logical_surfaces->local_point_count();
  amrex::ParallelDescriptor::ReduceLongSum(logical_count);
  if (logical_count != state.global_particle_count)
    amrex::Abort("ParticleAHFinderX retirement left inconsistent logical and "
                 "particle population counts");
}

void update_surface_states(Runtime &state,
                           const std::vector<SurfaceStateUpdate> &updates) {
  if (updates.empty())
    return;
  for (std::size_t index = 0; index < updates.size(); ++index) {
    const auto &update = updates[index];
    if (update.surface_id <= 0 ||
        update.lifecycle == ParticleLifecycle::retired ||
        (index > 0 && updates[index - 1].surface_id >= update.surface_id))
      amrex::Abort("ParticleAHFinderX device state updates must be unique, "
                   "increasing, and live");
    const auto surface = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), update.surface_id,
        [](const SurfaceRecord &record, const amrex::Long stable_id) {
          return record.stable_id < stable_id;
        });
    if (surface == state.surfaces.end() ||
        surface->stable_id != update.surface_id ||
        surface->lifecycle == ParticleLifecycle::retired)
      amrex::Abort("ParticleAHFinderX state update refers to an unknown "
                   "surface");
    surface->lifecycle = update.lifecycle;
    surface->role = update.role;
  }
  state.logical_surfaces->update_surface_states(updates);
  for (auto &container : state.containers)
    container->update_surface_states(updates);
}

void reset_surface_spheres(Runtime &state,
                           const std::vector<SurfaceSphereReset> &resets) {
  if (resets.empty())
    return;
  for (std::size_t index = 0; index < resets.size(); ++index) {
    const auto &reset = resets[index];
    if (reset.surface_id <= 0 || !(reset.radius > 0) ||
        !finite_center(reset.center) ||
        (index > 0 && resets[index - 1].surface_id >= reset.surface_id))
      amrex::Abort("ParticleAHFinderX sphere resets must be finite, "
                   "positive, unique, and increasing");
    const auto surface = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), reset.surface_id,
        [](const SurfaceRecord &record, const amrex::Long stable_id) {
          return record.stable_id < stable_id;
        });
    if (surface == state.surfaces.end() ||
        surface->stable_id != reset.surface_id ||
        surface->lifecycle == ParticleLifecycle::retired)
      amrex::Abort("ParticleAHFinderX sphere reset refers to an unknown "
                   "surface");
    surface->radius = reset.radius;
    surface->center = reset.center;
  }
  state.logical_surfaces->reset_surface_spheres(resets);
  for (auto &container : state.containers) {
    container->reset_surface_spheres(resets);
    container->redistribute_global();
  }
}

std::vector<LogicalSurfaceSeed>
local_logical_seeds(const std::vector<SurfaceRecord> &surfaces) {
  const int rank = amrex::ParallelDescriptor::MyProc();
  std::vector<LogicalSurfaceSeed> seeds;
  for (const auto &surface : surfaces) {
    if (surface.lifecycle == ParticleLifecycle::retired ||
        surface.logical_owner_rank != rank)
      continue;
    for (int level = 0; level < surface.angular_levels; ++level) {
      const auto &layer = surface.angular_layer[level];
      LogicalSurfaceSeed seed;
      seed.stable_id = surface.stable_id;
      seed.first_particle_id = layer.first_particle_id;
      seed.owner_offset = layer.logical_owner_offset;
      seed.particle_count = layer.particle_count;
      seed.generation = surface.generation;
      seed.angular_level = level;
      seed.angular_levels = surface.angular_levels;
      seed.ntheta = layer.ntheta;
      seed.nphi = layer.nphi;
      seed.lifecycle = static_cast<int>(surface.lifecycle);
      seed.role = static_cast<int>(surface.role);
      seed.center = surface.center;
      seed.radius = surface.radius;
      seed.mass_scale = surface.mass_scale;
      seed.theta_l2_tolerance = surface.tolerances.l2_times_mass;
      seed.theta_linf_tolerance = surface.tolerances.linf_times_mass;
      seeds.push_back(seed);
    }
  }
  return seeds;
}

std::vector<LogicalOwnership>
logical_ownership(const std::vector<SurfaceRecord> &surfaces) {
  std::vector<LogicalOwnership> ownership;
  std::size_t layer_count = 0;
  for (const auto &surface : surfaces)
    if (surface.lifecycle != ParticleLifecycle::retired)
      layer_count += static_cast<std::size_t>(surface.angular_levels);
  ownership.reserve(layer_count);
  for (const auto &surface : surfaces)
    if (surface.lifecycle != ParticleLifecycle::retired)
      for (int level = 0; level < surface.angular_levels; ++level) {
        const auto &layer = surface.angular_layer[level];
        ownership.push_back(
            {surface.stable_id, layer.logical_owner_offset,
             surface.logical_owner_rank, surface.generation, level,
             surface.angular_levels, layer.ntheta, layer.nphi});
      }
  std::sort(ownership.begin(), ownership.end(),
            [](const auto &a, const auto &b) {
              return a.surface_id < b.surface_id ||
                     (a.surface_id == b.surface_id &&
                      a.angular_level < b.angular_level);
            });
  return ownership;
}

void update_surface_diagnostics(LogicalSurfaceStorage &logical_surfaces,
                                std::vector<SurfaceRecord> &surfaces,
                                const int iteration, const amrex::Real time,
                                const int relaxation_steps) {
  constexpr int real_components = 25;
  constexpr int long_components = 7;
  const auto local = logical_surfaces.local_surface_diagnostics();
  std::vector<amrex::Real> real_values(real_components * surfaces.size(), 0);
  std::vector<amrex::Long> long_values(long_components * surfaces.size(), 0);
  for (const auto &diagnostics : local) {
    const auto found = std::lower_bound(
        surfaces.begin(), surfaces.end(), diagnostics.stable_id,
        [](const SurfaceRecord &surface, const amrex::Long stable_id) {
          return surface.stable_id < stable_id;
        });
    if (found == surfaces.end() || found->stable_id != diagnostics.stable_id)
      amrex::Abort("ParticleAHFinderX logical diagnostics have an unknown "
                   "surface ID");
    if (diagnostics.angular_level < 0 ||
        diagnostics.angular_level >= found->angular_levels ||
        diagnostics.angular_levels != found->angular_levels)
      amrex::Abort("ParticleAHFinderX logical diagnostics have invalid "
                   "angular-layer metadata");
    if (diagnostics.angular_level + 1 != diagnostics.angular_levels)
      continue;
    const std::size_t surface =
        static_cast<std::size_t>(found - surfaces.begin());
    real_values[real_components * surface + 0] = diagnostics.area;
    real_values[real_components * surface + 1] = diagnostics.mean_radius;
    real_values[real_components * surface + 2] = diagnostics.minimum_radius;
    real_values[real_components * surface + 3] = diagnostics.maximum_radius;
    real_values[real_components * surface + 4] = diagnostics.expansion_l2;
    real_values[real_components * surface + 5] = diagnostics.expansion_linf;
    for (int d = 0; d < 3; ++d)
      real_values[real_components * surface + 6 + d] =
          diagnostics.centroid[d];
    real_values[real_components * surface + 9] =
        diagnostics.mean_expansion;
    for (int component = 0; component < 6; ++component)
      real_values[real_components * surface + 10 + component] =
          diagnostics.coordinate_quadrupole[component];
    for (int d = 0; d < 3; ++d) {
      real_values[real_components * surface + 16 + d] =
          diagnostics.position_minimum[d];
      real_values[real_components * surface + 19 + d] =
          diagnostics.position_maximum[d];
      real_values[real_components * surface + 22 + d] =
          diagnostics.proper_circumference[d];
    }
    long_values[long_components * surface + 0] = 1;
    long_values[long_components * surface + 1] =
        static_cast<int>(diagnostics.relaxation_state);
    long_values[long_components * surface + 2] = diagnostics.invalid_points;
    long_values[long_components * surface + 3] = diagnostics.due;
    long_values[long_components * surface + 4] =
        diagnostics.minimum_sampled_level;
    long_values[long_components * surface + 5] =
        diagnostics.maximum_sampled_level;
    long_values[long_components * surface + 6] =
        diagnostics.sampled_level_changes;
  }
  if (!real_values.empty()) {
    if (real_values.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        long_values.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
      amrex::Abort("ParticleAHFinderX bounded diagnostics table exceeds the "
                   "MPI reduction count range");
    amrex::ParallelDescriptor::ReduceRealSum(
        real_values.data(), static_cast<int>(real_values.size()));
    amrex::ParallelDescriptor::ReduceLongSum(
        long_values.data(), static_cast<int>(long_values.size()));
  }

  for (std::size_t surface = 0; surface < surfaces.size(); ++surface) {
    auto &record = surfaces[surface];
    if (record.lifecycle == ParticleLifecycle::retired) {
      if (long_values[long_components * surface] != 0)
        amrex::Abort("ParticleAHFinderX retired surface retained a logical "
                     "diagnostic owner");
      continue;
    }
    if (long_values[long_components * surface] != 1)
      amrex::Abort("ParticleAHFinderX expected exactly one logical diagnostic "
                   "owner per surface");
    if (long_values[long_components * surface + 3] == 0)
      continue;
    record.area = real_values[real_components * surface + 0];
    record.mean_radius = real_values[real_components * surface + 1];
    record.minimum_radius = real_values[real_components * surface + 2];
    record.maximum_radius = real_values[real_components * surface + 3];
    record.expansion_l2 = real_values[real_components * surface + 4];
    record.expansion_linf = real_values[real_components * surface + 5];
    for (int d = 0; d < 3; ++d)
      record.centroid[d] = real_values[real_components * surface + 6 + d];
    record.mean_expansion = real_values[real_components * surface + 9];
    for (int component = 0; component < 6; ++component)
      record.coordinate_quadrupole[component] =
          real_values[real_components * surface + 10 + component];
    for (int d = 0; d < 3; ++d) {
      record.position_minimum[d] =
          real_values[real_components * surface + 16 + d];
      record.position_maximum[d] =
          real_values[real_components * surface + 19 + d];
      record.proper_circumference[d] =
          real_values[real_components * surface + 22 + d];
    }
    record.invalid_points = long_values[long_components * surface + 2];
    record.minimum_sampled_level = static_cast<int>(
        long_values[long_components * surface + 4]);
    record.maximum_sampled_level = static_cast<int>(
        long_values[long_components * surface + 5]);
    record.sampled_level_changes =
        long_values[long_components * surface + 6];
    if (record.cumulative_sampled_level_changes >
        std::numeric_limits<amrex::Long>::max() -
            record.sampled_level_changes)
      amrex::Abort("ParticleAHFinderX cumulative AMR level-change counter "
                   "overflowed");
    record.cumulative_sampled_level_changes +=
        record.sampled_level_changes;
    record.relaxation_state = static_cast<SurfaceRelaxationState>(
        long_values[long_components * surface + 1]);
    record.last_search_iteration = iteration;
    record.last_search_time = time;
    record.last_search_steps = relaxation_steps;
    if (record.relaxation_state == SurfaceRelaxationState::converged)
      record.last_success_iteration = iteration;
  }
}

void predict_surface_centers(
    const cGH *const cctkGH, Runtime &state,
    std::vector<amrex::Long> &due_surface_ids, const int iteration,
    const amrex::Real time,
    const amrex::Real maximum_extrapolation_intervals,
    const int minimum_active_level, const int maximum_active_level,
    const bool complete_hierarchy) {
  DECLARE_CCTK_PARAMETERS;
  if (due_surface_ids.empty())
    return;
  if (state.containers.empty())
    CCTK_ERROR("ParticleAHFinderX cannot preflight domain exit without a "
               "persistent AMReX container");
  const PhysicalDomainBounds domain =
      state.containers.front()->physical_domain_bounds();
  std::vector<SurfaceCenterUpdate> updates;
  std::vector<amrex::Long> ready_surface_ids;
  std::vector<amrex::Long> retire_ids;
  std::vector<amrex::Long> candidate_domain_retire_ids;
  updates.reserve(due_surface_ids.size());
  ready_surface_ids.reserve(due_surface_ids.size());
  for (auto &surface : state.surfaces) {
    if (!std::binary_search(due_surface_ids.begin(), due_surface_ids.end(),
                            surface.stable_id))
      continue;

    std::array<amrex::Real, 3> external{{0, 0, 0}};
    std::array<amrex::Real, 3> history{{0, 0, 0}};
    std::array<amrex::Real, 3> target = surface.manual_center;
    const bool have_external =
        external_center(cctkGH, state, surface, external);
    const bool have_history =
        history_center(surface, time, maximum_extrapolation_intervals,
                       history);

    switch (surface.center_policy) {
    case CenterPolicy::manual:
      target = surface.manual_center;
      break;
    case CenterPolicy::external:
      target = have_external ? external : surface.manual_center;
      break;
    case CenterPolicy::centroid:
      target = have_history ? history : surface.manual_center;
      break;
    case CenterPolicy::external_then_centroid:
      target = have_history ? history
                            : (have_external ? external : surface.manual_center);
      break;
    case CenterPolicy::centroid_then_external:
      target = have_history ? history
                            : (have_external ? external : surface.manual_center);
      break;
    case CenterPolicy::blend_external_centroid:
      if (have_external && have_history)
        for (int d = 0; d < 3; ++d)
          target[d] = surface.external_center_weight * external[d] +
                      (1 - surface.external_center_weight) * history[d];
      else if (have_external)
        target = external;
      else if (have_history)
        target = history;
      break;
    case CenterPolicy::puncture_tracker_displacement: {
      const auto &center_history = surface.center_history;
      const int latest = center_history.count - 1;
      if (have_external && latest >= 0 &&
          center_history.external_valid[latest] != 0) {
        // C_guess(t) = C_AH(last) + P(t) - P(last).
        for (int d = 0; d < 3; ++d)
          target[d] = center_history.centers[latest][d] + external[d] -
                      center_history.external_centers[latest][d];
      } else if (have_external) {
        target = external;
      } else if (have_history) {
        target = history;
      } else {
        target = surface.manual_center;
      }
      break;
    }
    case CenterPolicy::parent_midpoint:
      if (!parent_center(state.surfaces, surface, target))
        target = surface.center;
      break;
    }

    if (!finite_center(target))
      CCTK_VERROR("ParticleAHFinderX predicted a non-finite center for "
                  "surface %lld",
                  static_cast<long long>(surface.stable_id));

    if (retire_on_domain_exit &&
        !surface_fits_physical_domain(surface, target, domain)) {
      if (!complete_hierarchy) {
        if (verbose)
          CCTK_VINFO("Withholding surface %lld because its proposed radial "
                     "bound does not fit the partial active level band; "
                     "domain-exit lifecycle checks wait for complete "
                     "hierarchy synchronization",
                     static_cast<long long>(surface.stable_id));
        continue;
      }
      if (surface.last_domain_exit_check_iteration != iteration) {
        if (surface.consecutive_domain_exit_checks ==
            std::numeric_limits<int>::max())
          CCTK_ERROR("ParticleAHFinderX domain-exit counter overflowed");
        ++surface.consecutive_domain_exit_checks;
        surface.last_domain_exit_check_iteration = iteration;
        surface.last_state_change_iteration = iteration;
      }
      if (verbose)
        CCTK_VINFO("Withholding surface %lld because its proposed radial "
                   "bound does not fit the non-periodic physical domain "
                   "(domain-exit check %d/%d)",
                   static_cast<long long>(surface.stable_id),
                   surface.consecutive_domain_exit_checks,
                   domain_exit_grace_searches);
      if (surface.consecutive_domain_exit_checks >=
          domain_exit_grace_searches) {
        surface.retired_outside_domain = true;
        retire_ids.push_back(surface.stable_id);
        if (surface.lifecycle == ParticleLifecycle::candidate &&
            surface.role == SurfaceRole::common)
          candidate_domain_retire_ids.push_back(surface.stable_id);
      }
      continue;
    }

    surface.consecutive_domain_exit_checks = 0;
    surface.last_domain_exit_check_iteration = -1;
    ready_surface_ids.push_back(surface.stable_id);
    amrex::Real displacement_squared = 0;
    amrex::Real scale = 1;
    for (int d = 0; d < 3; ++d) {
      const amrex::Real displacement = target[d] - surface.center[d];
      displacement_squared += displacement * displacement;
      scale = amrex::max(scale, amrex::max(std::abs(target[d]),
                                           std::abs(surface.center[d])));
    }
    const amrex::Real threshold =
        16 * std::numeric_limits<amrex::Real>::epsilon() * scale;
    if (displacement_squared <= threshold * threshold)
      continue;
    updates.push_back({surface.stable_id, surface.center, target});
  }

  if (!complete_hierarchy && !updates.empty()) {
    std::vector<amrex::Long> translation_stats(2 * updates.size(), 0);
    for (const auto &container : state.containers) {
      const auto local = container->local_active_band_translation_stats(
          updates, minimum_active_level, maximum_active_level);
      if (local.size() != updates.size())
        CCTK_ERROR("ParticleAHFinderX active-band translation preflight "
                   "returned an inconsistent result size");
      for (std::size_t index = 0; index < updates.size(); ++index) {
        if (local[index].stable_id != updates[index].surface_id)
          CCTK_ERROR("ParticleAHFinderX active-band translation preflight "
                     "returned an inconsistent surface ID");
        translation_stats[2 * index] += local[index].particle_count;
        translation_stats[2 * index + 1] +=
            local[index].outside_active_band;
      }
    }
    amrex::ParallelDescriptor::ReduceLongSum(
        translation_stats.data(), static_cast<int>(translation_stats.size()));

    std::vector<SurfaceCenterUpdate> accepted_updates;
    std::vector<amrex::Long> deferred_ids;
    accepted_updates.reserve(updates.size());
    deferred_ids.reserve(updates.size());
    for (std::size_t index = 0; index < updates.size(); ++index) {
      const auto found = std::lower_bound(
          state.surfaces.begin(), state.surfaces.end(),
          updates[index].surface_id,
          [](const SurfaceRecord &surface, const amrex::Long stable_id) {
            return surface.stable_id < stable_id;
          });
      if (found == state.surfaces.end() ||
          found->stable_id != updates[index].surface_id)
        CCTK_ERROR("ParticleAHFinderX active-band translation preflight lost "
                   "surface metadata");
      const amrex::Long expected = total_angular_particle_count(*found);
      const amrex::Long in_band = translation_stats[2 * index];
      const amrex::Long outside_after_translation =
          translation_stats[2 * index + 1];
      if (in_band > expected || outside_after_translation > in_band)
        CCTK_VERROR(
            "ParticleAHFinderX active-band translation preflight counted "
            "an invalid population for surface %lld: in_band=%lld "
            "outside_after_translation=%lld expected=%lld",
            static_cast<long long>(updates[index].surface_id),
            static_cast<long long>(in_band),
            static_cast<long long>(outside_after_translation),
            static_cast<long long>(expected));
      if (in_band != expected || outside_after_translation != 0) {
        deferred_ids.push_back(updates[index].surface_id);
        if (verbose)
          CCTK_VINFO(
              "Withholding surface %lld because translating its persistent "
              "population requires %lld inactive point(s) and would move "
              "%lld active point(s) outside level band=[%d,%d)",
              static_cast<long long>(updates[index].surface_id),
              static_cast<long long>(expected - in_band),
              static_cast<long long>(outside_after_translation),
              minimum_active_level, maximum_active_level);
      } else {
        accepted_updates.push_back(updates[index]);
      }
    }
    if (!deferred_ids.empty())
      ready_surface_ids.erase(
          std::remove_if(ready_surface_ids.begin(), ready_surface_ids.end(),
                         [&](const amrex::Long stable_id) {
                           return std::binary_search(deferred_ids.begin(),
                                                     deferred_ids.end(),
                                                     stable_id);
                         }),
          ready_surface_ids.end());
    updates.swap(accepted_updates);
  }

  if (!updates.empty()) {
    for (const auto &update : updates) {
      const auto found = std::lower_bound(
          state.surfaces.begin(), state.surfaces.end(), update.surface_id,
          [](const SurfaceRecord &surface, const amrex::Long stable_id) {
            return surface.stable_id < stable_id;
          });
      if (found == state.surfaces.end() ||
          found->stable_id != update.surface_id)
        CCTK_ERROR("ParticleAHFinderX center update lost surface metadata");
      found->center = update.new_center;
    }
    state.logical_surfaces->update_surface_centers(updates);
    for (auto &container : state.containers) {
      container->update_surface_centers(updates);
      if (complete_hierarchy)
        container->redistribute_global();
      else
        container->redistribute_active_band(minimum_active_level,
                                            maximum_active_level, 0);
    }
  }
  if (!retire_ids.empty()) {
    retire_surface_populations(state, retire_ids);
    for (const auto stable_id : candidate_domain_retire_ids) {
      const auto found = std::lower_bound(
          state.surfaces.begin(), state.surfaces.end(), stable_id,
          [](const SurfaceRecord &surface, const amrex::Long id) {
            return surface.stable_id < id;
          });
      if (found == state.surfaces.end() || found->stable_id != stable_id)
        CCTK_ERROR("ParticleAHFinderX lost domain-retired candidate "
                   "metadata");
      append_candidate_event(state, *found,
                             CandidateEventType::retirement,
                             CandidateEventReason::outside_domain,
                             iteration, time);
    }
  }
  due_surface_ids.swap(ready_surface_ids);
}

void update_tracking_history(const cGH *const cctkGH, Runtime &state,
                             const int iteration, const amrex::Real time) {
  for (auto &surface : state.surfaces) {
    if (surface.last_search_iteration != iteration)
      continue;
    if (surface.relaxation_state != SurfaceRelaxationState::converged ||
        surface.invalid_points != 0 || !(surface.area > 0) ||
        !finite_center(surface.centroid)) {
      if (surface.relaxation_state == SurfaceRelaxationState::failed)
        ++surface.consecutive_failures;
      continue;
    }
    surface.consecutive_failures = 0;
    if (surface.last_history_iteration == iteration)
      continue;
    auto &history = surface.center_history;
    if (history.count == 3) {
      for (int sample = 0; sample < 2; ++sample) {
        history.times[sample] = history.times[sample + 1];
        history.centers[sample] = history.centers[sample + 1];
        history.mean_radii[sample] = history.mean_radii[sample + 1];
        history.external_centers[sample] =
            history.external_centers[sample + 1];
        history.external_valid[sample] = history.external_valid[sample + 1];
      }
    } else {
      ++history.count;
    }
    const int sample = history.count - 1;
    history.times[sample] = time;
    history.centers[sample] = surface.centroid;
    history.mean_radii[sample] = surface.mean_radius;
    std::array<amrex::Real, 3> external{{0, 0, 0}};
    history.external_valid[sample] =
        external_center(cctkGH, state, surface, external) ? 1 : 0;
    history.external_centers[sample] = external;
    surface.last_history_iteration = iteration;
  }
}

} // namespace ParticleAHFinderX
