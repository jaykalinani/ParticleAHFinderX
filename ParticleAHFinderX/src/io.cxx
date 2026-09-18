/**
 * \file io.cxx
 * \brief Parallel particle visualization output and bounded TSV diagnostics.
 *
 * Parallel chunking and pinned-host transport are confined to output cadence.
 * The solver defines its own surface schema and lifecycle rules.
 */
#include "runtime.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <driver.hxx>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleUtil.H>
#include <AMReX_Scan.H>

#include <mpi.h>

#ifdef HAVE_CAPABILITY_openPMD_api
#include <openPMD/openPMD.hpp>
#endif

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ParticleAHFinderX {
namespace {

constexpr int output_schema = 6;
constexpr const char *particle_dataset_name = "particleahfinderx_particles";

using HostParticleTile =
    amrex::ParticleTile<ParticleType, RealIdx::count, IntIdx::count,
                        amrex::PolymorphicArenaAllocator>;

std::string iteration_string(const int iteration) {
  std::ostringstream stream;
  stream << std::setw(8) << std::setfill('0') << iteration;
  return stream.str();
}

std::string patch_name(const std::size_t patch) {
  std::ostringstream stream;
  stream << "patch" << std::setw(2) << std::setfill('0') << patch;
  return stream.str();
}

bool output_time_is_valid(const cGH *const cctkGH, const int cadence) {
  return cadence > 0 && cctkGH->cctk_iteration % cadence == 0 &&
         CarpetX::all_levels_synchronized();
}

void write_debug_tsv(const cGH *const cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  if (!output_time_is_valid(cctkGH, debug_tsv_every))
    return;

  auto &state = runtime();
  amrex::Long global = 0;
  for (const auto &container : state.containers)
    global += container->local_particle_count();
  amrex::ParallelDescriptor::ReduceLongSum(global);
  if (global > debug_max_particles)
    CCTK_VERROR("ParticleAHFinderX debug TSV is limited to %d particles, "
                "but the current population is %lld",
                int(debug_max_particles), static_cast<long long>(global));

  if (CCTK_CreateDirectory(0755, particle_out_dir) < 0)
    CCTK_VERROR("Could not create ParticleAHFinderX output directory '%s'",
                particle_out_dir);
  std::ostringstream filename;
  filename << particle_out_dir << '/' << particle_dataset_name << ".it"
           << iteration_string(cctkGH->cctk_iteration) << ".rank"
           << std::setw(6) << std::setfill('0')
           << amrex::ParallelDescriptor::MyProc() << ".tsv";
  std::ofstream output(filename.str());
  if (!output)
    CCTK_VERROR("Could not create ParticleAHFinderX debug file '%s'",
                filename.str().c_str());
  output << "# x y z particle_id cpu surface_id generation angular_level "
            "angular_levels itheta iphi ntheta nphi lifecycle role "
            "validity sampled_level h v "
            "sampled_level_changed expansion normal_x normal_y normal_z "
            "area_weight\n";
  output << std::setprecision(17);

  // This intentionally copies only a hard-bounded debug population to pinned
  // host memory. It is never the production visualization path.
  for (const auto &container : state.containers)
    for (int level = 0; level <= container->finestLevel(); ++level)
      for (const auto &entry : container->GetParticles(level)) {
        HostParticleTile host_tile;
        host_tile.define(container->NumRuntimeRealComps(),
                         container->NumRuntimeIntComps(), nullptr, nullptr,
                         amrex::The_Pinned_Arena());
        host_tile.resize(entry.second.numParticles());
        amrex::copyParticles(host_tile, entry.second);
        const auto particles = host_tile.getConstParticleTileData();
        for (int i = 0; i < host_tile.numParticles(); ++i) {
          if (!particles.id(i).is_valid())
            continue;
          const amrex::Long surface_id = surface_id_from_limbs(
              particles.idata(IntIdx::surface_id_low)[i],
              particles.idata(IntIdx::surface_id_middle)[i],
              particles.idata(IntIdx::surface_id_high)[i]);
          output << particles.pos(0, i) << ' ' << particles.pos(1, i) << ' '
                 << particles.pos(2, i) << ' '
                 << static_cast<long long>(particles.id(i)) << ' '
                 << static_cast<int>(particles.cpu(i)) << ' '
                 << static_cast<long long>(surface_id) << ' '
                 << particles.idata(IntIdx::generation)[i] << ' '
                 << particles.idata(IntIdx::angular_level)[i] << ' '
                 << particles.idata(IntIdx::angular_levels)[i] << ' '
                 << particles.idata(IntIdx::itheta)[i] << ' '
                 << particles.idata(IntIdx::iphi)[i] << ' '
                 << particles.idata(IntIdx::ntheta)[i] << ' '
                 << particles.idata(IntIdx::nphi)[i] << ' '
                 << particles.idata(IntIdx::lifecycle)[i] << ' '
                 << particles.idata(IntIdx::role)[i] << ' '
                 << particles.idata(IntIdx::validity)[i] << ' '
                 << particles.idata(IntIdx::sampled_level)[i] << ' '
                 << particles.rdata(RealIdx::height)[i] << ' '
                 << particles.rdata(RealIdx::relaxation_velocity)[i] << ' '
                 << particles.idata(IntIdx::sampled_level_changed)[i] << ' '
                 << particles.rdata(RealIdx::expansion)[i] << ' '
                 << particles.rdata(RealIdx::normal_component(0))[i] << ' '
                 << particles.rdata(RealIdx::normal_component(1))[i] << ' '
                 << particles.rdata(RealIdx::normal_component(2))[i] << ' '
                 << particles.rdata(RealIdx::area_weight)[i] << '\n';
        }
      }
}

#ifdef HAVE_CAPABILITY_openPMD_api

template <typename T>
std::shared_ptr<T> host_array(const std::size_t size) {
  auto *const arena = amrex::The_Pinned_Arena();
  auto *const data = static_cast<T *>(arena->alloc(size * sizeof(T)));
  return std::shared_ptr<T>(data, [arena](T *const ptr) { arena->free(ptr); });
}

template <typename T>
void define_component(openPMD::RecordComponent &component,
                      const std::uint64_t count) {
  component.resetDataset(openPMD::Dataset(openPMD::determineDatatype<T>(),
                                          openPMD::Extent{count}));
}

void write_openpmd(const cGH *const cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  if (!output_time_is_valid(cctkGH, particle_out_every))
    return;

  auto &state = runtime();
  if (state.last_particle_output_iteration == cctkGH->cctk_iteration)
    return;
  if (CCTK_CreateDirectory(0755, particle_out_dir) < 0)
    throw std::runtime_error(
        "Could not create the ParticleAHFinderX output directory");

  const std::string filename =
      std::string(particle_out_dir) + "/" + particle_dataset_name + ".it" +
      iteration_string(cctkGH->cctk_iteration) + "." + particle_out_backend;
  openPMD::Series series(filename, openPMD::Access::CREATE,
                         amrex::ParallelDescriptor::Communicator());
  series.setSoftware("ParticleAHFinderX", "0.1.0");
  auto &iteration = series.iterations[cctkGH->cctk_iteration];
  iteration.setAttribute("schemaVersion", output_schema);
  iteration.setAttribute("cactusTime", static_cast<double>(cctkGH->cctk_time));
  iteration.setAttribute("coordinateSystem", std::string("Cartesian"));
  iteration.setAttribute("angularTopology",
                         std::string("cell_centered_latitude_longitude"));
  iteration.setAttribute("coordinateUnits", std::string("Cactus code units"));

  for (std::size_t patch = 0; patch < state.containers.size(); ++patch) {
    auto &container = *state.containers[patch];
    const long long local = container.local_target_particle_count();
    long long offset = 0;
    long long global = 0;
    const auto communicator = amrex::ParallelDescriptor::Communicator();
    MPI_Exscan(&local, &offset, 1, MPI_LONG_LONG, MPI_SUM, communicator);
    MPI_Allreduce(&local, &global, 1, MPI_LONG_LONG, MPI_SUM, communicator);
    if (amrex::ParallelDescriptor::MyProc() == 0)
      offset = 0;
    if (global == 0)
      continue;

    auto &species =
        iteration.particles[std::string(particle_dataset_name) + "_" +
                            patch_name(patch)];
    auto &position = species["position"];
    auto &position_offset = species["positionOffset"];
    for (const char *axis : {"x", "y", "z"}) {
      define_component<amrex::Real>(position[axis], global);
      define_component<amrex::Real>(position_offset[axis], global);
      position_offset[axis].makeConstant(amrex::Real(0));
    }

    auto &particle_id =
        species["id"][openPMD::RecordComponent::SCALAR];
    auto &cpu = species["cpu"][openPMD::RecordComponent::SCALAR];
    auto &surface_id =
        species["surfaceId"][openPMD::RecordComponent::SCALAR];
    auto &generation =
        species["generation"][openPMD::RecordComponent::SCALAR];
    auto &angular_level =
        species["angularLevel"][openPMD::RecordComponent::SCALAR];
    auto &angular_levels =
        species["angularLevels"][openPMD::RecordComponent::SCALAR];
    auto &itheta = species["itheta"][openPMD::RecordComponent::SCALAR];
    auto &iphi = species["iphi"][openPMD::RecordComponent::SCALAR];
    auto &ntheta = species["ntheta"][openPMD::RecordComponent::SCALAR];
    auto &nphi = species["nphi"][openPMD::RecordComponent::SCALAR];
    auto &lifecycle =
        species["lifecycle"][openPMD::RecordComponent::SCALAR];
    auto &role = species["role"][openPMD::RecordComponent::SCALAR];
    auto &validity =
        species["validity"][openPMD::RecordComponent::SCALAR];
    auto &sampled_level =
        species["sampledLevel"][openPMD::RecordComponent::SCALAR];
    auto &sampled_level_changed =
        species["sampledLevelChanged"][openPMD::RecordComponent::SCALAR];
    auto &height = species["height"][openPMD::RecordComponent::SCALAR];
    auto &velocity = species["relaxationVelocity"]
                            [openPMD::RecordComponent::SCALAR];
    auto &expansion =
        species["expansion"][openPMD::RecordComponent::SCALAR];
    auto &normal = species["normal"];
    auto &area_weight =
        species["areaWeight"][openPMD::RecordComponent::SCALAR];

    define_component<std::int64_t>(particle_id, global);
    define_component<std::int32_t>(cpu, global);
    define_component<std::int64_t>(surface_id, global);
    for (auto *component : {&generation, &angular_level, &angular_levels,
                            &itheta, &iphi, &ntheta, &nphi, &lifecycle,
                            &role, &validity, &sampled_level,
                            &sampled_level_changed})
      define_component<std::int32_t>(*component, global);
    define_component<amrex::Real>(height, global);
    define_component<amrex::Real>(velocity, global);
    define_component<amrex::Real>(expansion, global);
    for (const char *axis : {"x", "y", "z"})
      define_component<amrex::Real>(normal[axis], global);
    define_component<amrex::Real>(area_weight, global);

    std::uint64_t current = static_cast<std::uint64_t>(offset);
    for (int level = 0; level <= container.finestLevel(); ++level) {
      for (ParIter iterator(container, level); iterator.isValid(); ++iterator) {
        const int count = iterator.numParticles();
        if (count == 0)
          continue;
        amrex::Gpu::DeviceVector<int> output_flags(count);
        amrex::Gpu::DeviceVector<int> output_offsets(count);
        auto *const flags = output_flags.dataPtr();
        auto *const compact_offsets = output_offsets.dataPtr();
        const auto particles =
            iterator.GetParticleTile().getConstParticleTileData();
        amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(const int i) noexcept {
          flags[i] = particles.id(i).is_valid() &&
                             particles.idata(IntIdx::angular_level)[i] + 1 ==
                                 particles.idata(IntIdx::angular_levels)[i]
                         ? 1
                         : 0;
        });
        const int output_count = amrex::Scan::ExclusiveSum(
            count, flags, compact_offsets, amrex::Scan::retSum);
        if (output_count == 0)
          continue;
        const openPMD::Offset chunk_offset{current};
        const openPMD::Extent chunk_extent{
            static_cast<std::uint64_t>(output_count)};

        enum RealOutputComponent {
          output_x,
          output_y,
          output_z,
          output_height,
          output_velocity,
          output_expansion,
          output_normal_x,
          output_normal_y,
          output_normal_z,
          output_area_weight,
          num_real_output_components
        };
        enum LongOutputComponent {
          output_particle_id,
          output_surface_id,
          num_long_output_components
        };
        enum IntOutputComponent {
          output_cpu,
          output_generation,
          output_angular_level,
          output_angular_levels,
          output_itheta,
          output_iphi,
          output_ntheta,
          output_nphi,
          output_lifecycle,
          output_role,
          output_validity,
          output_sampled_level,
          output_sampled_level_changed,
          num_int_output_components
        };

        amrex::Gpu::DeviceVector<amrex::Real> device_reals(
            num_real_output_components * output_count);
        amrex::Gpu::DeviceVector<std::int64_t> device_longs(
            num_long_output_components * output_count);
        amrex::Gpu::DeviceVector<std::int32_t> device_ints(
            num_int_output_components * output_count);
        auto *const packed_reals = device_reals.dataPtr();
        auto *const packed_longs = device_longs.dataPtr();
        auto *const packed_ints = device_ints.dataPtr();
        amrex::ParallelFor(count,
                           [=] AMREX_GPU_DEVICE(const int i) noexcept {
          if (!flags[i])
            return;
          const int slot = compact_offsets[i];
          packed_reals[output_x * output_count + slot] = particles.pos(0, i);
          packed_reals[output_y * output_count + slot] = particles.pos(1, i);
          packed_reals[output_z * output_count + slot] = particles.pos(2, i);
          packed_reals[output_height * output_count + slot] =
              particles.rdata(RealIdx::height)[i];
          packed_reals[output_velocity * output_count + slot] =
              particles.rdata(RealIdx::relaxation_velocity)[i];
          packed_reals[output_expansion * output_count + slot] =
              particles.rdata(RealIdx::expansion)[i];
          packed_reals[output_normal_x * output_count + slot] =
              particles.rdata(RealIdx::normal_component(0))[i];
          packed_reals[output_normal_y * output_count + slot] =
              particles.rdata(RealIdx::normal_component(1))[i];
          packed_reals[output_normal_z * output_count + slot] =
              particles.rdata(RealIdx::normal_component(2))[i];
          packed_reals[output_area_weight * output_count + slot] =
              particles.rdata(RealIdx::area_weight)[i];

          packed_longs[output_particle_id * output_count + slot] =
              static_cast<std::int64_t>(particles.id(i));
          packed_longs[output_surface_id * output_count + slot] =
              static_cast<std::int64_t>(surface_id_from_limbs(
                  particles.idata(IntIdx::surface_id_low)[i],
                  particles.idata(IntIdx::surface_id_middle)[i],
                  particles.idata(IntIdx::surface_id_high)[i]));

          packed_ints[output_cpu * output_count + slot] =
              static_cast<std::int32_t>(particles.cpu(i));
          packed_ints[output_generation * output_count + slot] =
              particles.idata(IntIdx::generation)[i];
          packed_ints[output_angular_level * output_count + slot] =
              particles.idata(IntIdx::angular_level)[i];
          packed_ints[output_angular_levels * output_count + slot] =
              particles.idata(IntIdx::angular_levels)[i];
          packed_ints[output_itheta * output_count + slot] =
              particles.idata(IntIdx::itheta)[i];
          packed_ints[output_iphi * output_count + slot] =
              particles.idata(IntIdx::iphi)[i];
          packed_ints[output_ntheta * output_count + slot] =
              particles.idata(IntIdx::ntheta)[i];
          packed_ints[output_nphi * output_count + slot] =
              particles.idata(IntIdx::nphi)[i];
          packed_ints[output_lifecycle * output_count + slot] =
              particles.idata(IntIdx::lifecycle)[i];
          packed_ints[output_role * output_count + slot] =
              particles.idata(IntIdx::role)[i];
          packed_ints[output_validity * output_count + slot] =
              particles.idata(IntIdx::validity)[i];
          packed_ints[output_sampled_level * output_count + slot] =
              particles.idata(IntIdx::sampled_level)[i];
          packed_ints[output_sampled_level_changed * output_count + slot] =
              particles.idata(IntIdx::sampled_level_changed)[i];
        });

        const auto copy_real = [&](const int component) {
          auto values = host_array<amrex::Real>(output_count);
          amrex::Gpu::copy(
              amrex::Gpu::deviceToHost,
              device_reals.begin() + component * output_count,
              device_reals.begin() + (component + 1) * output_count,
              values.get());
          return values;
        };
        const auto copy_long = [&](const int component) {
          auto values = host_array<std::int64_t>(output_count);
          amrex::Gpu::copy(
              amrex::Gpu::deviceToHost,
              device_longs.begin() + component * output_count,
              device_longs.begin() + (component + 1) * output_count,
              values.get());
          return values;
        };
        const auto copy_int = [&](const int component) {
          auto values = host_array<std::int32_t>(output_count);
          amrex::Gpu::copy(
              amrex::Gpu::deviceToHost,
              device_ints.begin() + component * output_count,
              device_ints.begin() + (component + 1) * output_count,
              values.get());
          return values;
        };

        auto px = copy_real(output_x);
        auto py = copy_real(output_y);
        auto pz = copy_real(output_z);
        auto heights = copy_real(output_height);
        auto velocities = copy_real(output_velocity);
        auto expansions = copy_real(output_expansion);
        auto normal_x = copy_real(output_normal_x);
        auto normal_y = copy_real(output_normal_y);
        auto normal_z = copy_real(output_normal_z);
        auto area_weights = copy_real(output_area_weight);
        auto ids = copy_long(output_particle_id);
        auto surface_ids = copy_long(output_surface_id);
        auto cpus = copy_int(output_cpu);
        auto generations = copy_int(output_generation);
        auto angular_level_values = copy_int(output_angular_level);
        auto angular_levels_values = copy_int(output_angular_levels);
        auto ithetas = copy_int(output_itheta);
        auto iphis = copy_int(output_iphi);
        auto nthetas = copy_int(output_ntheta);
        auto nphis = copy_int(output_nphi);
        auto lifecycles = copy_int(output_lifecycle);
        auto roles = copy_int(output_role);
        auto validities = copy_int(output_validity);
        auto sampled_levels = copy_int(output_sampled_level);
        auto sampled_levels_changed =
            copy_int(output_sampled_level_changed);

        position["x"].storeChunk(px, chunk_offset, chunk_extent);
        position["y"].storeChunk(py, chunk_offset, chunk_extent);
        position["z"].storeChunk(pz, chunk_offset, chunk_extent);
        particle_id.storeChunk(ids, chunk_offset, chunk_extent);
        cpu.storeChunk(cpus, chunk_offset, chunk_extent);
        surface_id.storeChunk(surface_ids, chunk_offset, chunk_extent);
        generation.storeChunk(generations, chunk_offset, chunk_extent);
        angular_level.storeChunk(angular_level_values, chunk_offset,
                                 chunk_extent);
        angular_levels.storeChunk(angular_levels_values, chunk_offset,
                                  chunk_extent);
        itheta.storeChunk(ithetas, chunk_offset, chunk_extent);
        iphi.storeChunk(iphis, chunk_offset, chunk_extent);
        ntheta.storeChunk(nthetas, chunk_offset, chunk_extent);
        nphi.storeChunk(nphis, chunk_offset, chunk_extent);
        lifecycle.storeChunk(lifecycles, chunk_offset, chunk_extent);
        role.storeChunk(roles, chunk_offset, chunk_extent);
        validity.storeChunk(validities, chunk_offset, chunk_extent);
        sampled_level.storeChunk(sampled_levels, chunk_offset, chunk_extent);
        sampled_level_changed.storeChunk(sampled_levels_changed, chunk_offset,
                                         chunk_extent);
        height.storeChunk(heights, chunk_offset, chunk_extent);
        velocity.storeChunk(velocities, chunk_offset, chunk_extent);
        expansion.storeChunk(expansions, chunk_offset, chunk_extent);
        normal["x"].storeChunk(normal_x, chunk_offset, chunk_extent);
        normal["y"].storeChunk(normal_y, chunk_offset, chunk_extent);
        normal["z"].storeChunk(normal_z, chunk_offset, chunk_extent);
        area_weight.storeChunk(area_weights, chunk_offset, chunk_extent);
        current += output_count;
      }
      // Bound deferred pinned storage without assuming identical tile counts.
      series.flush();
    }
  }

  iteration.close();
  series.flush();
  series.close();
  state.last_particle_output_iteration = cctkGH->cctk_iteration;
}

#endif // HAVE_CAPABILITY_openPMD_api

} // namespace

extern "C" void ParticleAHFinderX_Output(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  try {
    write_debug_tsv(cctkGH);
#ifdef HAVE_CAPABILITY_openPMD_api
    write_openpmd(cctkGH);
#endif
  } catch (const std::exception &error) {
    CCTK_VERROR("ParticleAHFinderX output failed: %s", error.what());
  }
}

} // namespace ParticleAHFinderX
