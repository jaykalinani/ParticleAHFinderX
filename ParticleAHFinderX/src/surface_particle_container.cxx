/**
 * \file surface_particle_container.cxx
 * \brief One-time GPU creation and persistent AMReX ownership of surfaces.
 */
#include "surface_particle_container.hxx"
#include "angular_geometry.hxx"

#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleReduce.H>

#include <mpi.h>

#include <limits>

namespace ParticleAHFinderX {
namespace {

struct ParticleBirthDescriptor {
  amrex::Long begin;
  amrex::Long count;
  amrex::Long surface_id;
  amrex::Long first_particle_id;
  amrex::Long logical_owner_offset;
  int logical_owner_rank;
  int generation;
  int angular_level;
  int angular_levels;
  int ntheta;
  int nphi;
  int lifecycle;
  int role;
  amrex::Real center[3];
  amrex::Real radius;
};

amrex::Long proportional_offset(const amrex::Long count, const int part,
                                const int parts) {
  const amrex::Long quotient = count / parts;
  const amrex::Long remainder = count % parts;
  return quotient * part + remainder * part / parts;
}

} // namespace

amrex::Long SurfaceSeed::particle_count() const {
  if (ntheta <= 0 || nphi <= 0 ||
      amrex::Long(ntheta) >
          std::numeric_limits<amrex::Long>::max() / amrex::Long(nphi))
    amrex::Abort("ParticleAHFinderX invalid or overflowing angular surface "
                 "size");
  return amrex::Long(ntheta) * amrex::Long(nphi);
}

SurfaceParticleContainer::SurfaceParticleContainer(amrex::AmrCore *const core)
    : BaseContainer(core) {
  SetSoACompileTimeNames(
      {"x",
       "y",
       "z",
       "height",
       "relaxation_velocity",
       "expansion",
       "normal_x",
       "normal_y",
       "normal_z",
       "area_weight",
       "gamma_xx",
       "gamma_xy",
       "gamma_xz",
       "gamma_yy",
       "gamma_yz",
       "gamma_zz",
       "d_x_gamma_xx",
       "d_x_gamma_xy",
       "d_x_gamma_xz",
       "d_x_gamma_yy",
       "d_x_gamma_yz",
       "d_x_gamma_zz",
       "d_y_gamma_xx",
       "d_y_gamma_xy",
       "d_y_gamma_xz",
       "d_y_gamma_yy",
       "d_y_gamma_yz",
       "d_y_gamma_zz",
       "d_z_gamma_xx",
       "d_z_gamma_xy",
       "d_z_gamma_xz",
       "d_z_gamma_yy",
       "d_z_gamma_yz",
       "d_z_gamma_zz",
       "curv_xx",
       "curv_xy",
       "curv_xz",
       "curv_yy",
       "curv_yz",
       "curv_zz"},
      {"surface_id_low", "surface_id_middle", "surface_id_high",
       "generation", "angular_level", "angular_levels", "itheta", "iphi",
       "lifecycle", "role", "solver_active", "ntheta", "nphi", "validity",
       "sampled_level", "sampled_level_changed", "logical_owner_rank",
       "logical_index_low", "logical_index_middle", "logical_index_high"});
}

bool SurfaceParticleContainer::has_level_zero_tile() const {
  for (amrex::MFIter mfi = MakeMFIter(0); mfi.isValid(); ++mfi)
    return true;
  return false;
}

void SurfaceParticleContainer::create_surface(const SurfaceSeed &seed) {
  create_surfaces({seed});
}

void SurfaceParticleContainer::create_surfaces(
    const std::vector<SurfaceSeed> &seeds) {
  if (seeds.empty())
    return;
  if (seeds.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX surface birth descriptor count exceeds "
                 "the portable device-kernel index range");
  std::vector<ParticleBirthDescriptor> host_descriptors;
  host_descriptors.reserve(seeds.size());
  amrex::Long total_count = 0;
  for (const auto &seed : seeds) {
    const amrex::Long count = seed.particle_count();
    if (seed.surface_id <= 0 || seed.first_particle_id <= 0 ||
        seed.radius <= 0 || seed.logical_owner_rank < 0 ||
        seed.logical_owner_offset < 0 || seed.angular_level < 0 ||
        seed.angular_levels <= 0 ||
        seed.angular_level >= seed.angular_levels ||
        total_count > std::numeric_limits<amrex::Long>::max() - count)
      amrex::Abort("ParticleAHFinderX received an invalid or overflowing "
                   "batched surface birth");
    ParticleBirthDescriptor descriptor{};
    descriptor.begin = total_count;
    descriptor.count = count;
    descriptor.surface_id = seed.surface_id;
    descriptor.first_particle_id = seed.first_particle_id;
    descriptor.logical_owner_offset = seed.logical_owner_offset;
    descriptor.logical_owner_rank = seed.logical_owner_rank;
    descriptor.generation = seed.generation;
    descriptor.angular_level = seed.angular_level;
    descriptor.angular_levels = seed.angular_levels;
    descriptor.ntheta = seed.ntheta;
    descriptor.nphi = seed.nphi;
    descriptor.lifecycle = static_cast<int>(seed.lifecycle);
    descriptor.role = static_cast<int>(seed.role);
    for (int d = 0; d < 3; ++d)
      descriptor.center[d] = seed.center[d];
    descriptor.radius = seed.radius;
    host_descriptors.push_back(descriptor);
    total_count += count;
  }

  const int rank = amrex::ParallelDescriptor::MyProc();
  const int has_tile = has_level_zero_tile() ? 1 : 0;
  int active_ranks = 0;
  int active_rank = 0;
  MPI_Allreduce(&has_tile, &active_ranks, 1, MPI_INT, MPI_SUM,
                amrex::ParallelDescriptor::Communicator());
  MPI_Exscan(&has_tile, &active_rank, 1, MPI_INT, MPI_SUM,
             amrex::ParallelDescriptor::Communicator());
  if (rank == 0)
    active_rank = 0;
  if (active_ranks == 0)
    amrex::Abort("ParticleAHFinderX found no level-zero AMReX particle tiles");
  if (!has_tile)
    return;

  const amrex::Long rank_first =
      proportional_offset(total_count, active_rank, active_ranks);
  const amrex::Long rank_last =
      proportional_offset(total_count, active_rank + 1, active_ranks);
  const amrex::Long rank_count = rank_last - rank_first;
  if (rank_count == 0)
    return;

  int num_tiles = 0;
  for (amrex::MFIter mfi = MakeMFIter(0); mfi.isValid(); ++mfi)
    ++num_tiles;

  amrex::Gpu::DeviceVector<ParticleBirthDescriptor> device_descriptors(
      host_descriptors.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_descriptors.begin(),
                   host_descriptors.end(), device_descriptors.begin());
  const auto *const descriptors = device_descriptors.dataPtr();
  const int num_surfaces = static_cast<int>(host_descriptors.size());

  int tile_ordinal = 0;
  for (amrex::MFIter mfi = MakeMFIter(0); mfi.isValid();
       ++mfi, ++tile_ordinal) {
    const amrex::Long tile_begin =
        proportional_offset(rank_count, tile_ordinal, num_tiles);
    const amrex::Long tile_end =
        proportional_offset(rank_count, tile_ordinal + 1, num_tiles);
    const amrex::Long tile_count = tile_end - tile_begin;
    if (tile_count == 0)
      continue;

    auto &tile = DefineAndReturnParticleTile(0, mfi);
    const amrex::Long old_size = tile.numParticles();
    tile.resize(old_size + tile_count);
    auto particles = tile.getParticleTileData();

    amrex::ParallelFor(
        tile_count,
        [=] AMREX_GPU_DEVICE(const amrex::Long tile_index) noexcept {
          const amrex::Long flattened_index =
              rank_first + tile_begin + tile_index;
          int lower = 0;
          int upper = num_surfaces;
          while (lower + 1 < upper) {
            const int middle = lower + (upper - lower) / 2;
            if (descriptors[middle].begin <= flattened_index)
              lower = middle;
            else
              upper = middle;
          }
          const auto &surface = descriptors[lower];
          const amrex::Long surface_index =
              flattened_index - surface.begin;
          const int itheta = int(surface_index / surface.nphi);
          const int iphi = int(surface_index -
                               amrex::Long(itheta) * surface.nphi);
          const amrex::Real theta =
              amrex::Math::pi<amrex::Real>() *
              (amrex::Real(itheta) + amrex::Real(0.5)) / surface.ntheta;
          const amrex::Real phi =
              amrex::Real(2) * amrex::Math::pi<amrex::Real>() *
              (amrex::Real(iphi) + amrex::Real(0.5)) / surface.nphi;
          const AngularGeometry basis = angular_geometry(theta, phi);

          const amrex::Long i = old_size + tile_index;
          particles.id(i) = surface.first_particle_id + surface_index;
          particles.cpu(i) = rank;
          particles.pos(0, i) =
              surface.center[0] + surface.radius * basis.radial[0];
          particles.pos(1, i) =
              surface.center[1] + surface.radius * basis.radial[1];
          particles.pos(2, i) =
              surface.center[2] + surface.radius * basis.radial[2];
          particles.rdata(RealIdx::height)[i] = surface.radius;
          particles.rdata(RealIdx::relaxation_velocity)[i] = 0;
          particles.rdata(RealIdx::expansion)[i] = 0;
          for (int d = 0; d < 3; ++d)
            particles.rdata(RealIdx::normal_component(d))[i] = 0;
          particles.rdata(RealIdx::area_weight)[i] = 0;
          for (int component = RealIdx::gamma; component < RealIdx::count;
               ++component)
            particles.rdata(component)[i] = 0;
          particles.idata(IntIdx::surface_id_low)[i] =
              int(surface.surface_id & id_limb_mask);
          particles.idata(IntIdx::surface_id_middle)[i] =
              int((surface.surface_id >> id_limb_bits) & id_limb_mask);
          particles.idata(IntIdx::surface_id_high)[i] =
              int(surface.surface_id >> (2 * id_limb_bits));
          particles.idata(IntIdx::generation)[i] = surface.generation;
          particles.idata(IntIdx::angular_level)[i] = surface.angular_level;
          particles.idata(IntIdx::angular_levels)[i] = surface.angular_levels;
          particles.idata(IntIdx::itheta)[i] = itheta;
          particles.idata(IntIdx::iphi)[i] = iphi;
          particles.idata(IntIdx::lifecycle)[i] = surface.lifecycle;
          particles.idata(IntIdx::role)[i] = surface.role;
          particles.idata(IntIdx::solver_active)[i] = 0;
          particles.idata(IntIdx::ntheta)[i] = surface.ntheta;
          particles.idata(IntIdx::nphi)[i] = surface.nphi;
          particles.idata(IntIdx::validity)[i] =
              static_cast<int>(ParticleValidity::not_sampled);
          particles.idata(IntIdx::sampled_level)[i] = -1;
          particles.idata(IntIdx::sampled_level_changed)[i] = 0;
          particles.idata(IntIdx::logical_owner_rank)[i] =
              surface.logical_owner_rank;
          int logical_index_low;
          int logical_index_middle;
          int logical_index_high;
          long_to_limbs(surface.logical_owner_offset + surface_index,
                        logical_index_low, logical_index_middle,
                        logical_index_high);
          particles.idata(IntIdx::logical_index_low)[i] = logical_index_low;
          particles.idata(IntIdx::logical_index_middle)[i] =
              logical_index_middle;
          particles.idata(IntIdx::logical_index_high)[i] =
              logical_index_high;
        });
  }

  // Surface birth is a quiescent lifecycle boundary. Synchronizing here makes
  // the following AMReX ownership discovery explicit; there is no analogous
  // synchronization or allocation in an ordinary finder iteration.
  amrex::Gpu::streamSynchronize();
}

void SurfaceParticleContainer::update_surface_centers(
    const std::vector<SurfaceCenterUpdate> &updates) {
  if (updates.empty())
    return;
  if (updates.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX center-update table exceeds the portable "
                 "device-kernel index range");
  for (std::size_t update = 0; update < updates.size(); ++update) {
    if (updates[update].surface_id <= 0 ||
        (update > 0 && updates[update - 1].surface_id >=
                           updates[update].surface_id))
      amrex::Abort("ParticleAHFinderX center updates must have unique, "
                   "increasing surface IDs");
    for (int d = 0; d < 3; ++d)
      if (!amrex::Math::isfinite(updates[update].old_center[d]) ||
          !amrex::Math::isfinite(updates[update].new_center[d]))
        amrex::Abort("ParticleAHFinderX received a non-finite center update");
  }

  amrex::Gpu::DeviceVector<SurfaceCenterUpdate> device_updates(updates.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, updates.begin(), updates.end(),
                   device_updates.begin());
  const auto *const center_updates = device_updates.dataPtr();
  const int update_count = static_cast<int>(updates.size());
  for (int lev = 0; lev <= finestLevel(); ++lev)
    for (ParIter iterator(*this, lev); iterator.isValid(); ++iterator) {
      const int count = iterator.numParticles();
      if (count == 0)
        continue;
      auto particles = iterator.GetParticleTile().getParticleTileData();
      amrex::ParallelFor(count,
                         [=] AMREX_GPU_DEVICE(const int i) noexcept {
        if (!particles.id(i).is_valid())
          return;
        const amrex::Long surface_id = surface_id_from_limbs(
            particles.idata(IntIdx::surface_id_low)[i],
            particles.idata(IntIdx::surface_id_middle)[i],
            particles.idata(IntIdx::surface_id_high)[i]);
        int lower = 0;
        int upper = update_count;
        while (lower < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (center_updates[middle].surface_id < surface_id)
            lower = middle + 1;
          else
            upper = middle;
        }
        if (lower >= update_count ||
            center_updates[lower].surface_id != surface_id)
          return;
        for (int d = 0; d < 3; ++d)
          particles.pos(d, i) += center_updates[lower].new_center[d] -
                                 center_updates[lower].old_center[d];
      });
    }
  amrex::Gpu::streamSynchronize();
}

void SurfaceParticleContainer::update_surface_states(
    const std::vector<SurfaceStateUpdate> &updates) {
  if (updates.empty())
    return;
  if (updates.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX state-update table exceeds the portable "
                 "device-kernel index range");
  amrex::Gpu::DeviceVector<SurfaceStateUpdate> device_updates(updates.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, updates.begin(), updates.end(),
                   device_updates.begin());
  const auto *const state_updates = device_updates.dataPtr();
  const int update_count = static_cast<int>(updates.size());
  for (int lev = 0; lev <= finestLevel(); ++lev)
    for (ParIter iterator(*this, lev); iterator.isValid(); ++iterator) {
      const int count = iterator.numParticles();
      if (count == 0)
        continue;
      auto particles = iterator.GetParticleTile().getParticleTileData();
      amrex::ParallelFor(count,
                         [=] AMREX_GPU_DEVICE(const int i) noexcept {
        if (!particles.id(i).is_valid())
          return;
        const amrex::Long surface_id = surface_id_from_limbs(
            particles.idata(IntIdx::surface_id_low)[i],
            particles.idata(IntIdx::surface_id_middle)[i],
            particles.idata(IntIdx::surface_id_high)[i]);
        int lower = 0;
        int upper = update_count;
        while (lower < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (state_updates[middle].surface_id < surface_id)
            lower = middle + 1;
          else
            upper = middle;
        }
        if (lower >= update_count ||
            state_updates[lower].surface_id != surface_id)
          return;
        particles.idata(IntIdx::lifecycle)[i] =
            static_cast<int>(state_updates[lower].lifecycle);
        particles.idata(IntIdx::role)[i] =
            static_cast<int>(state_updates[lower].role);
      });
    }
  amrex::Gpu::streamSynchronize();
}

void SurfaceParticleContainer::reset_surface_spheres(
    const std::vector<SurfaceSphereReset> &resets) {
  if (resets.empty())
    return;
  if (resets.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX sphere-reset table exceeds the portable "
                 "device-kernel index range");
  amrex::Gpu::DeviceVector<SurfaceSphereReset> device_resets(resets.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, resets.begin(), resets.end(),
                   device_resets.begin());
  const auto *const sphere_resets = device_resets.dataPtr();
  const int reset_count = static_cast<int>(resets.size());
  for (int lev = 0; lev <= finestLevel(); ++lev)
    for (ParIter iterator(*this, lev); iterator.isValid(); ++iterator) {
      const int count = iterator.numParticles();
      if (count == 0)
        continue;
      auto particles = iterator.GetParticleTile().getParticleTileData();
      amrex::ParallelFor(count,
                         [=] AMREX_GPU_DEVICE(const int i) noexcept {
        if (!particles.id(i).is_valid())
          return;
        const amrex::Long surface_id = surface_id_from_limbs(
            particles.idata(IntIdx::surface_id_low)[i],
            particles.idata(IntIdx::surface_id_middle)[i],
            particles.idata(IntIdx::surface_id_high)[i]);
        int lower = 0;
        int upper = reset_count;
        while (lower < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (sphere_resets[middle].surface_id < surface_id)
            lower = middle + 1;
          else
            upper = middle;
        }
        if (lower >= reset_count ||
            sphere_resets[lower].surface_id != surface_id)
          return;
        const int itheta = particles.idata(IntIdx::itheta)[i];
        const int iphi = particles.idata(IntIdx::iphi)[i];
        const int ntheta = particles.idata(IntIdx::ntheta)[i];
        const int nphi = particles.idata(IntIdx::nphi)[i];
        const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                                  (itheta + amrex::Real(0.5)) / ntheta;
        const amrex::Real phi = 2 * amrex::Math::pi<amrex::Real>() *
                                (iphi + amrex::Real(0.5)) / nphi;
        const AngularGeometry basis = angular_geometry(theta, phi);
        const auto &reset = sphere_resets[lower];
        particles.pos(0, i) =
            reset.center[0] + reset.radius * basis.radial[0];
        particles.pos(1, i) =
            reset.center[1] + reset.radius * basis.radial[1];
        particles.pos(2, i) = reset.center[2] +
                              reset.radius * basis.radial[2];
        particles.rdata(RealIdx::height)[i] = reset.radius;
        particles.rdata(RealIdx::relaxation_velocity)[i] = 0;
        particles.rdata(RealIdx::expansion)[i] = 0;
        particles.idata(IntIdx::validity)[i] =
            static_cast<int>(ParticleValidity::not_sampled);
        particles.idata(IntIdx::sampled_level)[i] = -1;
        particles.idata(IntIdx::sampled_level_changed)[i] = 0;
      });
    }
  amrex::Gpu::streamSynchronize();
}

amrex::Long SurfaceParticleContainer::retire_surfaces(
    const std::vector<amrex::Long> &stable_ids) {
  if (stable_ids.empty())
    return 0;
  if (stable_ids.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX retirement table exceeds the portable "
                 "device-kernel index range");
  for (std::size_t surface = 0; surface < stable_ids.size(); ++surface)
    if (stable_ids[surface] <= 0 ||
        (surface > 0 && stable_ids[surface - 1] >= stable_ids[surface]))
      amrex::Abort("ParticleAHFinderX retirement IDs must be unique and "
                   "increasing");

  amrex::Gpu::DeviceVector<amrex::Long> device_ids(stable_ids.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, stable_ids.begin(),
                   stable_ids.end(), device_ids.begin());
  amrex::Gpu::DeviceVector<amrex::Long> device_retired(1, 0);
  const auto *const ids = device_ids.dataPtr();
  amrex::Long *const retired = device_retired.dataPtr();
  const int surface_count = static_cast<int>(stable_ids.size());
  for (int lev = 0; lev <= finestLevel(); ++lev)
    for (ParIter iterator(*this, lev); iterator.isValid(); ++iterator) {
      const int count = iterator.numParticles();
      if (count == 0)
        continue;
      auto particles = iterator.GetParticleTile().getParticleTileData();
      amrex::ParallelFor(count,
                         [=] AMREX_GPU_DEVICE(const int i) noexcept {
        if (!particles.id(i).is_valid())
          return;
        const amrex::Long surface_id = surface_id_from_limbs(
            particles.idata(IntIdx::surface_id_low)[i],
            particles.idata(IntIdx::surface_id_middle)[i],
            particles.idata(IntIdx::surface_id_high)[i]);
        int lower = 0;
        int upper = surface_count;
        while (lower < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (ids[middle] < surface_id)
            lower = middle + 1;
          else
            upper = middle;
        }
        if (lower < surface_count && ids[lower] == surface_id) {
          particles.id(i) = -1;
          amrex::Gpu::Atomic::AddNoRet(retired, amrex::Long(1));
        }
      });
    }
  amrex::Long local_retired = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_retired.begin(),
                   device_retired.end(), &local_retired);
  Redistribute(0, -1, 0, 0, true);
  return local_retired;
}

void SurfaceParticleContainer::resize_after_regrid() {
  resizeData();
  fine_masks_.clear();
}

void SurfaceParticleContainer::rebuild_amr_masks() {
  fine_masks_.clear();
  fine_masks_.resize(static_cast<std::size_t>(finestLevel() + 1));
  for (int lev = 0; lev < finestLevel(); ++lev) {
    fine_masks_[lev] = std::make_unique<amrex::iMultiFab>(amrex::makeFineMask(
        ParticleBoxArray(lev), ParticleDistributionMap(lev),
        ParticleBoxArray(lev + 1), GetParGDB()->refRatio(lev), 1, 0));
  }
}

void SurfaceParticleContainer::redistribute_global() {
  Redistribute(0, -1, 0, 0, true);
}

void SurfaceParticleContainer::redistribute_local_hierarchy(
    const int max_num_cells_moved) {
  Redistribute(0, -1, 0, max_num_cells_moved, true);
}

std::vector<amrex::Long> SurfaceParticleContainer::remap_logical_ownership(
    const std::vector<LogicalOwnership> &ownership) {
  if (ownership.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX logical ownership table exceeds the "
                 "portable device-kernel index range");
  amrex::Gpu::DeviceVector<LogicalOwnership> device_ownership(
      ownership.size());
  if (!ownership.empty())
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, ownership.begin(),
                     ownership.end(), device_ownership.begin());
  amrex::Gpu::DeviceVector<int> device_error(1, 0);
  amrex::Gpu::DeviceVector<amrex::Long> device_counts(ownership.size(), 0);
  const auto *const owners = device_ownership.dataPtr();
  const int num_surfaces = static_cast<int>(ownership.size());
  int *const error = device_error.dataPtr();
  amrex::Long *const counts = device_counts.dataPtr();

  for (int lev = 0; lev <= finestLevel(); ++lev)
    for (ParIter iterator(*this, lev); iterator.isValid(); ++iterator) {
      const int count = iterator.numParticles();
      if (count == 0)
        continue;
      auto particles =
          iterator.GetParticleTile().getParticleTileData();
      amrex::ParallelFor(count,
                         [=] AMREX_GPU_DEVICE(const int i) noexcept {
        if (!particles.id(i).is_valid())
          return;
        const amrex::Long surface_id = surface_id_from_limbs(
            particles.idata(IntIdx::surface_id_low)[i],
            particles.idata(IntIdx::surface_id_middle)[i],
            particles.idata(IntIdx::surface_id_high)[i]);
        const int angular_level =
            particles.idata(IntIdx::angular_level)[i];
        int lower = 0;
        int upper = num_surfaces;
        while (lower < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (owners[middle].surface_id < surface_id ||
              (owners[middle].surface_id == surface_id &&
               owners[middle].angular_level < angular_level))
            lower = middle + 1;
          else
            upper = middle;
        }
        if (lower >= num_surfaces ||
            owners[lower].surface_id != surface_id ||
            owners[lower].generation !=
                particles.idata(IntIdx::generation)[i] ||
            owners[lower].angular_level != angular_level ||
            owners[lower].angular_levels !=
                particles.idata(IntIdx::angular_levels)[i] ||
            owners[lower].ntheta != particles.idata(IntIdx::ntheta)[i] ||
            owners[lower].nphi != particles.idata(IntIdx::nphi)[i]) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }
        const int itheta = particles.idata(IntIdx::itheta)[i];
        const int iphi = particles.idata(IntIdx::iphi)[i];
        if (itheta < 0 || itheta >= owners[lower].ntheta || iphi < 0 ||
            iphi >= owners[lower].nphi) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }
        particles.idata(IntIdx::logical_owner_rank)[i] =
            owners[lower].owner_rank;
        const amrex::Long logical_index =
            owners[lower].owner_offset +
            amrex::Long(itheta) * owners[lower].nphi + iphi;
        int low;
        int middle;
        int high;
        long_to_limbs(logical_index, low, middle, high);
        particles.idata(IntIdx::logical_index_low)[i] = low;
        particles.idata(IntIdx::logical_index_middle)[i] = middle;
        particles.idata(IntIdx::logical_index_high)[i] = high;
        amrex::Gpu::Atomic::AddNoRet(counts + lower, amrex::Long(1));
      });
    }

  int host_error = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_error.begin(),
                   device_error.end(), &host_error);
  if (host_error)
    amrex::Abort("ParticleAHFinderX checkpoint particles do not match the "
                 "restored logical surface metadata");
  std::vector<amrex::Long> local_counts(ownership.size(), 0);
  if (!ownership.empty())
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_counts.begin(),
                     device_counts.end(), local_counts.begin());
  return local_counts;
}

void SurfaceParticleContainer::set_active_surface_layers(
    const std::vector<SurfaceLayerSelection> &selections) {
  if (selections.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX active angular-layer table exceeds the "
                 "portable device-kernel index range");
  for (std::size_t index = 0; index < selections.size(); ++index) {
    if (selections[index].stable_id <= 0 ||
        selections[index].angular_level < 0 ||
        (index > 0 &&
         (selections[index - 1].stable_id > selections[index].stable_id ||
          (selections[index - 1].stable_id == selections[index].stable_id &&
           selections[index - 1].angular_level >=
               selections[index].angular_level))))
      amrex::Abort("ParticleAHFinderX active angular layers must be unique "
                   "and ordered");
  }
  amrex::Gpu::DeviceVector<SurfaceLayerSelection> device_selections(
      selections.size());
  if (!selections.empty())
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, selections.begin(),
                     selections.end(), device_selections.begin());
  const auto *const selected = device_selections.dataPtr();
  const int selected_count = static_cast<int>(selections.size());

  for (int lev = 0; lev <= finestLevel(); ++lev)
    for (ParIter pti(*this, lev); pti.isValid(); ++pti) {
      const int count = pti.numParticles();
      if (count == 0)
        continue;
      auto particles = pti.GetParticleTile().getParticleTileData();
      amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(const int i) noexcept {
        if (!particles.id(i).is_valid())
          return;
        const amrex::Long stable_id = surface_id_from_limbs(
            particles.idata(IntIdx::surface_id_low)[i],
            particles.idata(IntIdx::surface_id_middle)[i],
            particles.idata(IntIdx::surface_id_high)[i]);
        const int angular_level =
            particles.idata(IntIdx::angular_level)[i];
        int lower = 0;
        int upper = selected_count;
        while (lower < upper) {
          const int middle = lower + (upper - lower) / 2;
          const bool before = selected[middle].stable_id < stable_id ||
                              (selected[middle].stable_id == stable_id &&
                               selected[middle].angular_level < angular_level);
          if (before)
            lower = middle + 1;
          else
            upper = middle;
        }
        particles.idata(IntIdx::solver_active)[i] =
            lower < selected_count &&
                    selected[lower].stable_id == stable_id &&
                    selected[lower].angular_level == angular_level
                ? 1
                : 0;
      });
    }
  amrex::Gpu::streamSynchronize();
}

void SurfaceParticleContainer::begin_adm_gather() {
  for (int lev = 0; lev <= finestLevel(); ++lev)
    for (ParIter pti(*this, lev); pti.isValid(); ++pti) {
      const int count = pti.numParticles();
      if (count == 0)
        continue;
      auto particles = pti.GetParticleTile().getParticleTileData();
      amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(const int i) noexcept {
        if (particles.id(i).is_valid() &&
            particles.idata(IntIdx::solver_active)[i] != 0)
          particles.idata(IntIdx::validity)[i] =
              static_cast<int>(ParticleValidity::not_sampled);
      });
    }
}

void SurfaceParticleContainer::begin_sampled_level_tracking() {
  for (int lev = 0; lev <= finestLevel(); ++lev)
    for (ParIter pti(*this, lev); pti.isValid(); ++pti) {
      const int count = pti.numParticles();
      if (count == 0)
        continue;
      auto particles = pti.GetParticleTile().getParticleTileData();
      amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(const int i) noexcept {
        if (particles.id(i).is_valid())
          particles.idata(IntIdx::sampled_level_changed)[i] = 0;
      });
    }
}

void SurfaceParticleContainer::gather_adm_level(
    const int lev, const amrex::MultiFab &metric, const int metric_component,
    const amrex::MultiFab &curv, const int curv_component,
    const FieldCentering centering, const int interpolation_order,
    const AdmGatherPass pass) {
  if (lev < 0 || lev > finestLevel())
    amrex::Abort("ParticleAHFinderX ADM gather received an invalid AMR level");
  if (interpolation_order != 1 && interpolation_order != 3)
    amrex::Abort(
        "ParticleAHFinderX ADM gather supports interpolation order 1 or 3");

  const auto prob_lo = Geom(lev).ProbLoArray();
  const auto inv_dx = Geom(lev).InvCellSizeArray();
  const auto domain = Geom(lev).Domain();
  const bool has_fine_mask = lev < finestLevel();
  if (has_fine_mask &&
      (lev >= static_cast<int>(fine_masks_.size()) || !fine_masks_[lev]))
    amrex::Abort("ParticleAHFinderX ADM gather has no AMR fine mask");
  for (ParIter pti(*this, lev); pti.isValid(); ++pti) {
    const int count = pti.numParticles();
    if (count == 0)
      continue;

    auto particles = pti.GetParticleTile().getParticleTileData();
    const auto metric_array = metric.const_array(pti);
    const auto curv_array = curv.const_array(pti);
    const amrex::Box metric_available = metric[pti].box();
    const amrex::Box curv_available = curv[pti].box();
    amrex::Array4<const int> fine_mask;
    amrex::Box fine_mask_available;
    if (has_fine_mask) {
      fine_mask = fine_masks_[lev]->const_array(pti);
      fine_mask_available = (*fine_masks_[lev])[pti].box();
    }
    amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(const int i) noexcept {
      if (!particles.id(i).is_valid() ||
          particles.idata(IntIdx::solver_active)[i] == 0)
        return;

      const amrex::GpuArray<amrex::Real, 3> position{{
          particles.pos(0, i), particles.pos(1, i), particles.pos(2, i)}};
      amrex::IntVect cell;
      for (int d = 0; d < 3; ++d)
        cell[d] = static_cast<int>(amrex::Math::floor(
            (position[d] - prob_lo[d]) * inv_dx[d]));
      if (!domain.contains(cell)) {
        particles.idata(IntIdx::validity)[i] =
            static_cast<int>(ParticleValidity::outside_supported_domain);
        return;
      }
      if (has_fine_mask &&
          (!fine_mask_available.contains(cell) || fine_mask(cell) == 0)) {
        particles.idata(IntIdx::validity)[i] = static_cast<int>(
            pass == AdmGatherPass::final
                ? ParticleValidity::invalid_ghost_data
                : ParticleValidity::needs_hierarchy_redistribute);
        return;
      }

      AdmInterpolationData data;
      const InterpolationStatus status =
          interpolation_order == 3
              ? tensor_cubic_interpolate_adm(
                    metric_array, metric_component, curv_array,
                    curv_component, position, prob_lo, inv_dx, centering,
                    metric_available, curv_available, data)
              : trilinear_interpolate_adm(
                    metric_array, metric_component, curv_array,
                    curv_component, position, prob_lo, inv_dx, centering,
                    metric_available, curv_available, data);

      if (status == InterpolationStatus::stencil_unavailable) {
        const ParticleValidity validity =
            pass == AdmGatherPass::initial
                ? ParticleValidity::needs_local_redistribute
                : pass == AdmGatherPass::local_retry
                      ? ParticleValidity::needs_hierarchy_redistribute
                      : ParticleValidity::invalid_ghost_data;
        particles.idata(IntIdx::validity)[i] = static_cast<int>(validity);
        return;
      }
      if (status == InterpolationStatus::nonfinite) {
        particles.idata(IntIdx::validity)[i] =
            static_cast<int>(ParticleValidity::nonfinite);
        return;
      }

      for (int component = 0; component < 6; ++component) {
        particles.rdata(RealIdx::gamma_component(component))[i] =
            data.gamma[component];
        particles.rdata(RealIdx::curv_component(component))[i] =
            data.curv[component];
        for (int d = 0; d < 3; ++d)
          particles.rdata(RealIdx::d_gamma_component(d, component))[i] =
              data.d_gamma[d][component];
      }
      particles.idata(IntIdx::validity)[i] =
          static_cast<int>(ParticleValidity::valid);
      const int previous_level =
          particles.idata(IntIdx::sampled_level)[i];
      if (previous_level >= 0 && previous_level != lev)
        particles.idata(IntIdx::sampled_level_changed)[i] = 1;
      particles.idata(IntIdx::sampled_level)[i] = lev;
    });
  }
}

AdmGatherStats SurfaceParticleContainer::local_adm_gather_stats() const {
  amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                   amrex::ReduceOpSum, amrex::ReduceOpSum,
                   amrex::ReduceOpSum, amrex::ReduceOpSum,
                   amrex::ReduceOpSum>
      reduce_ops;
  using ReduceData = amrex::ReduceData<amrex::Long, amrex::Long, amrex::Long,
                                       amrex::Long, amrex::Long, amrex::Long,
                                       amrex::Long>;
  using ConstPTD = BaseContainer::ConstPTDType;
  const auto reduced = amrex::ParticleReduce<ReduceData>(
      *this,
      [=] AMREX_GPU_DEVICE(const ConstPTD &particles,
                           const int i) noexcept
          -> amrex::GpuTuple<amrex::Long, amrex::Long, amrex::Long,
                             amrex::Long, amrex::Long, amrex::Long,
                             amrex::Long> {
        if (!particles.id(i).is_valid() ||
            particles.idata(IntIdx::solver_active)[i] == 0)
          return {0, 0, 0, 0, 0, 0, 0};
        const auto status = static_cast<ParticleValidity>(
            particles.idata(IntIdx::validity)[i]);
        return {status == ParticleValidity::valid,
                status == ParticleValidity::not_sampled,
                status == ParticleValidity::needs_local_redistribute,
                status == ParticleValidity::needs_hierarchy_redistribute,
                status == ParticleValidity::outside_supported_domain,
                status == ParticleValidity::invalid_ghost_data,
                status == ParticleValidity::nonfinite};
      },
      reduce_ops);
  return {amrex::get<0>(reduced), amrex::get<1>(reduced),
          amrex::get<2>(reduced), amrex::get<3>(reduced),
          amrex::get<4>(reduced), amrex::get<5>(reduced),
          amrex::get<6>(reduced)};
}

amrex::Real SurfaceParticleContainer::local_minkowski_adm_error() const {
  amrex::ReduceOps<amrex::ReduceOpMax> reduce_ops;
  using ReduceData = amrex::ReduceData<amrex::Real>;
  using ConstPTD = BaseContainer::ConstPTDType;
  const auto reduced = amrex::ParticleReduce<ReduceData>(
      *this,
      [=] AMREX_GPU_DEVICE(const ConstPTD &particles,
                           const int i) noexcept
          -> amrex::GpuTuple<amrex::Real> {
        if (!particles.id(i).is_valid() ||
            particles.idata(IntIdx::solver_active)[i] == 0 ||
            static_cast<ParticleValidity>(
                particles.idata(IntIdx::validity)[i]) !=
                ParticleValidity::valid)
          return {0};
        amrex::Real error = 0;
        for (int component = 0; component < 6; ++component) {
          const amrex::Real expected_gamma =
              component == 0 || component == 3 || component == 5 ? 1 : 0;
          const amrex::Real gamma_error =
              particles.rdata(RealIdx::gamma_component(component))[i] -
              expected_gamma;
          error = amrex::max(error,
                             gamma_error < 0 ? -gamma_error : gamma_error);
          const amrex::Real curv_error =
              particles.rdata(RealIdx::curv_component(component))[i];
          error = amrex::max(error,
                             curv_error < 0 ? -curv_error : curv_error);
          for (int d = 0; d < 3; ++d) {
            const amrex::Real derivative_error = particles.rdata(
                RealIdx::d_gamma_component(d, component))[i];
            error = amrex::max(
                error, derivative_error < 0 ? -derivative_error
                                            : derivative_error);
          }
        }
        return {error};
      },
      reduce_ops);
  return amrex::get<0>(reduced);
}

amrex::Long SurfaceParticleContainer::local_particle_count() const {
  amrex::Long count = 0;
  for (int level = 0; level <= finestLevel(); ++level)
    for (const auto &entry : GetParticles(level))
      count += entry.second.numParticles();
  return count;
}

amrex::Long SurfaceParticleContainer::local_target_particle_count() const {
  amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
  using ReduceData = amrex::ReduceData<amrex::Long>;
  using ConstPTD = BaseContainer::ConstPTDType;
  const auto reduced = amrex::ParticleReduce<ReduceData>(
      *this,
      [=] AMREX_GPU_DEVICE(const ConstPTD &particles,
                           const int i) noexcept
          -> amrex::GpuTuple<amrex::Long> {
        return {particles.id(i).is_valid() &&
                particles.idata(IntIdx::angular_level)[i] + 1 ==
                    particles.idata(IntIdx::angular_levels)[i]};
      },
      reduce_ops);
  return amrex::get<0>(reduced);
}

PhysicalDomainBounds
SurfaceParticleContainer::physical_domain_bounds() const {
  PhysicalDomainBounds bounds;
  const auto lower = Geom(0).ProbLoArray();
  const auto upper = Geom(0).ProbHiArray();
  for (int d = 0; d < 3; ++d) {
    bounds.lower[d] = lower[d];
    bounds.upper[d] = upper[d];
    bounds.periodic[d] = Geom(0).isPeriodic(d);
  }
  return bounds;
}

} // namespace ParticleAHFinderX
