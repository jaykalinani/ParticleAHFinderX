/**
 * \file analysis_point_io.cxx
 * \brief Bounded GPU-packed point output for dynamic horizon analysis.
 */
#include "analysis_api.hxx"
#include "runtime.hxx"

#include <cctk.h>

#include <AMReX_Gpu.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Scan.H>
#include <AMReX_Utility.H>

#include <mpi.h>

#ifdef HAVE_CAPABILITY_openPMD_api
#include <openPMD/openPMD.hpp>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace ParticleAHFinderX {
namespace {

constexpr const char *analysis_point_dataset_name = "ah_analysis_points";

std::uint64_t stable_id_hash(
    const std::vector<std::int64_t> &stable_ids) {
  // FNV-1a over an explicit little-endian representation is deterministic
  // across hosts and distinguishes disjoint active-band transactions.
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (const std::int64_t stable_id : stable_ids) {
    const std::uint64_t value = static_cast<std::uint64_t>(stable_id);
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (8 * byte)) & UINT64_C(0xff);
      hash *= UINT64_C(1099511628211);
    }
  }
  return hash;
}

std::string point_frame_name(const int iteration,
                             const std::vector<std::int64_t> &stable_ids,
                             const char *const backend,
                             const bool temporary) {
  std::ostringstream name;
  name << analysis_point_dataset_name << ".it" << std::setw(8)
       << std::setfill('0') << iteration << ".n" << std::setw(4)
       << stable_ids.size() << ".h" << std::hex << std::setw(16)
       << stable_id_hash(stable_ids) << std::dec;
  if (temporary)
    name << ".tmp";
  name << '.' << backend;
  return name.str();
}

bool all_characters(const std::string &value, const std::size_t begin,
                    const std::size_t end,
                    const bool hexadecimal = false) {
  if (begin >= end)
    return false;
  for (std::size_t index = begin; index < end; ++index) {
    const unsigned char character =
        static_cast<unsigned char>(value[index]);
    if (hexadecimal ? !std::isxdigit(character) : !std::isdigit(character))
      return false;
  }
  return true;
}

bool recognized_point_frame(const std::string &name, const char *const backend,
                            int &iteration, bool &temporary) {
  const std::string prefix =
      std::string(analysis_point_dataset_name) + ".it";
  if (!name.starts_with(prefix))
    return false;
  const std::size_t count_marker = name.find(".n", prefix.size());
  const std::size_t hash_marker =
      count_marker == std::string::npos
          ? std::string::npos
          : name.find(".h", count_marker + 2);
  if (count_marker == std::string::npos || hash_marker == std::string::npos ||
      !all_characters(name, prefix.size(), count_marker) ||
      !all_characters(name, count_marker + 2, hash_marker))
    return false;

  const std::string final_suffix = std::string(".") + backend;
  const std::string temporary_suffix = std::string(".tmp.") + backend;
  temporary = name.ends_with(temporary_suffix);
  const std::string &suffix = temporary ? temporary_suffix : final_suffix;
  if (!name.ends_with(suffix))
    return false;
  const std::size_t hash_begin = hash_marker + 2;
  const std::size_t hash_end = name.size() - suffix.size();
  if (hash_end - hash_begin != 16 ||
      !all_characters(name, hash_begin, hash_end, true))
    return false;

  try {
    const long long parsed = std::stoll(
        name.substr(prefix.size(), count_marker - prefix.size()));
    if (parsed < 0 || parsed > std::numeric_limits<int>::max())
      return false;
    iteration = static_cast<int>(parsed);
  } catch (const std::exception &) {
    return false;
  }
  return true;
}

void validate_point_options(
    const SurfaceAnalysisPointOutputOptions &options) {
  if (!options.directory || options.directory[0] == '\0' ||
      !options.backend || std::string(options.backend) != "bp5" ||
      options.chunk_points <= 0)
    amrex::Abort("ParticleAHFinderX received invalid analysis point-output "
                 "options");
}

void validate_stable_ids(
    const std::vector<std::int64_t> &stable_ids) {
  if (stable_ids.empty())
    amrex::Abort("ParticleAHFinderX point output requires at least one stable "
                 "surface ID");
  if (stable_ids.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX point-output stable-ID table exceeds the "
                 "portable MPI range");
  std::int64_t previous = 0;
  for (const std::int64_t stable_id : stable_ids) {
    if (stable_id <= previous)
      amrex::Abort("ParticleAHFinderX point-output stable IDs must be positive, "
                   "sorted, and unique");
    previous = stable_id;
  }
}

void validate_collective_stable_ids(
    const std::vector<std::int64_t> &stable_ids) {
  const auto communicator = amrex::ParallelDescriptor::Communicator();
  int root_count = static_cast<int>(stable_ids.size());
  MPI_Bcast(&root_count, 1, MPI_INT, 0, communicator);
  if (root_count != static_cast<int>(stable_ids.size()))
    amrex::Abort("ParticleAHFinderX point-output stable-ID count differs "
                 "between MPI ranks");
  std::vector<std::int64_t> root_ids(stable_ids.size());
  if (amrex::ParallelDescriptor::MyProc() == 0)
    root_ids = stable_ids;
  MPI_Bcast(root_ids.data(), root_count, MPI_INT64_T, 0, communicator);
  if (root_ids != stable_ids)
    amrex::Abort("ParticleAHFinderX point-output stable IDs differ between "
                 "MPI ranks");
}

#ifdef HAVE_CAPABILITY_openPMD_api

template <typename T>
std::shared_ptr<T> pinned_array(const std::size_t size) {
  auto *const arena = amrex::The_Pinned_Arena();
  auto *const data = static_cast<T *>(arena->alloc(size * sizeof(T)));
  return std::shared_ptr<T>(data, [arena](T *const pointer) {
    arena->free(pointer);
  });
}

template <typename T>
std::shared_ptr<T> component_alias(const std::shared_ptr<T> &values,
                                   const std::size_t offset) {
  return std::shared_ptr<T>(values, values.get() + offset);
}

template <typename T>
void define_component(openPMD::RecordComponent &component,
                      const std::uint64_t count) {
  component.resetDataset(openPMD::Dataset(openPMD::determineDatatype<T>(),
                                          openPMD::Extent{count}));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
selected_stable_id(const amrex::Long stable_id,
                   const amrex::Long *const stable_ids,
                   const int stable_id_count) noexcept {
  int lower = 0;
  int upper = stable_id_count;
  while (lower < upper) {
    const int middle = lower + (upper - lower) / 2;
    if (stable_ids[middle] < stable_id)
      lower = middle + 1;
    else
      upper = middle;
  }
  return lower < stable_id_count && stable_ids[lower] == stable_id;
}

enum PointRealComponent : int {
  point_x,
  point_y,
  point_z,
  point_q_theta_theta,
  point_q_theta_phi,
  point_q_phi_phi,
  point_intrinsic_scalar_curvature,
  point_normal_connection_theta,
  point_normal_connection_phi,
  point_area_weight,
  point_invariant_coordinate,
  point_axial_vector_theta,
  point_axial_vector_phi,
  point_base_real_components,
  point_electric_xx = point_base_real_components,
  point_electric_xy,
  point_electric_xz,
  point_electric_yy,
  point_electric_yz,
  point_electric_zz,
  point_magnetic_xx,
  point_magnetic_xy,
  point_magnetic_xz,
  point_magnetic_yy,
  point_magnetic_yz,
  point_magnetic_zz,
  point_weyl_real_components,
};

enum PointIntComponent : int {
  point_generation,
  point_angular_level,
  point_angular_levels,
  point_itheta,
  point_iphi,
  point_ntheta,
  point_nphi,
  point_lifecycle,
  point_role,
  point_intrinsic_valid,
  point_axial_valid,
  point_weyl_valid,
  point_int_components,
};

void define_point_components(openPMD::ParticleSpecies &species,
                             const std::uint64_t global_count,
                             const bool include_weyl) {
  auto &position = species["position"];
  auto &position_offset = species["positionOffset"];
  for (const char *const axis : {"x", "y", "z"}) {
    define_component<amrex::Real>(position[axis], global_count);
    define_component<amrex::Real>(position_offset[axis], global_count);
    position_offset[axis].makeConstant(amrex::Real(0));
  }
  auto &surface_id =
      species["surfaceId"][openPMD::RecordComponent::SCALAR];
  define_component<std::int64_t>(surface_id, global_count);
  for (const char *const name :
       {"generation", "angularLevel", "angularLevels", "itheta", "iphi",
        "ntheta", "nphi", "lifecycle", "role", "intrinsicValid",
        "axialValid", "weylValid"})
    define_component<std::int32_t>(
        species[name][openPMD::RecordComponent::SCALAR], global_count);

  auto &metric = species["intrinsicMetric"];
  for (const char *const component : {"thetaTheta", "thetaPhi", "phiPhi"})
    define_component<amrex::Real>(metric[component], global_count);
  define_component<amrex::Real>(
      species["intrinsicScalarCurvature"]
             [openPMD::RecordComponent::SCALAR],
      global_count);
  auto &normal_connection = species["normalBundleConnection"];
  define_component<amrex::Real>(normal_connection["theta"], global_count);
  define_component<amrex::Real>(normal_connection["phi"], global_count);
  define_component<amrex::Real>(
      species["areaWeight"][openPMD::RecordComponent::SCALAR], global_count);
  define_component<amrex::Real>(
      species["invariantCoordinate"]
             [openPMD::RecordComponent::SCALAR],
      global_count);
  auto &axial_vector = species["axialVector"];
  define_component<amrex::Real>(axial_vector["theta"], global_count);
  define_component<amrex::Real>(axial_vector["phi"], global_count);
  if (include_weyl)
    for (const char *const record : {"electricWeyl", "magneticWeyl"}) {
      auto &tensor = species[record];
      for (const char *const component : {"xx", "xy", "xz", "yy", "yz",
                                          "zz"})
        define_component<amrex::Real>(tensor[component], global_count);
    }
}

void store_real_components(openPMD::ParticleSpecies &species,
                           const std::shared_ptr<amrex::Real> &values,
                           const int count,
                           const openPMD::Offset &offset,
                           const openPMD::Extent &extent,
                           const bool include_weyl) {
  const auto component = [&](const int index) {
    return component_alias(values, static_cast<std::size_t>(index) * count);
  };
  auto &position = species["position"];
  position["x"].storeChunk(component(point_x), offset, extent);
  position["y"].storeChunk(component(point_y), offset, extent);
  position["z"].storeChunk(component(point_z), offset, extent);
  auto &metric = species["intrinsicMetric"];
  metric["thetaTheta"].storeChunk(component(point_q_theta_theta), offset,
                                   extent);
  metric["thetaPhi"].storeChunk(component(point_q_theta_phi), offset, extent);
  metric["phiPhi"].storeChunk(component(point_q_phi_phi), offset, extent);
  species["intrinsicScalarCurvature"]
         [openPMD::RecordComponent::SCALAR]
             .storeChunk(component(point_intrinsic_scalar_curvature), offset,
                         extent);
  auto &normal_connection = species["normalBundleConnection"];
  normal_connection["theta"].storeChunk(
      component(point_normal_connection_theta), offset, extent);
  normal_connection["phi"].storeChunk(
      component(point_normal_connection_phi), offset, extent);
  species["areaWeight"][openPMD::RecordComponent::SCALAR].storeChunk(
      component(point_area_weight), offset, extent);
  species["invariantCoordinate"][openPMD::RecordComponent::SCALAR].storeChunk(
      component(point_invariant_coordinate), offset, extent);
  auto &axial_vector = species["axialVector"];
  axial_vector["theta"].storeChunk(component(point_axial_vector_theta), offset,
                                    extent);
  axial_vector["phi"].storeChunk(component(point_axial_vector_phi), offset,
                                  extent);
  if (include_weyl) {
    constexpr const char *components[6]{"xx", "xy", "xz", "yy", "yz",
                                         "zz"};
    auto &electric = species["electricWeyl"];
    auto &magnetic = species["magneticWeyl"];
    for (int tensor = 0; tensor < 6; ++tensor) {
      electric[components[tensor]].storeChunk(
          component(point_electric_xx + tensor), offset, extent);
      magnetic[components[tensor]].storeChunk(
          component(point_magnetic_xx + tensor), offset, extent);
    }
  }
}

void store_int_components(openPMD::ParticleSpecies &species,
                          const std::shared_ptr<std::int32_t> &values,
                          const int count,
                          const openPMD::Offset &offset,
                          const openPMD::Extent &extent) {
  constexpr const char *names[point_int_components]{
      "generation",  "angularLevel",   "angularLevels", "itheta",
      "iphi",        "ntheta",         "nphi",          "lifecycle",
      "role",        "intrinsicValid", "axialValid",    "weylValid"};
  for (int component = 0; component < point_int_components; ++component)
    species[names[component]][openPMD::RecordComponent::SCALAR].storeChunk(
        component_alias(values,
                        static_cast<std::size_t>(component) * count),
        offset, extent);
}

#endif // HAVE_CAPABILITY_openPMD_api

} // namespace

bool surface_analysis_point_output_available() {
#ifdef HAVE_CAPABILITY_openPMD_api
  return true;
#else
  return false;
#endif
}

void write_current_surface_analysis_points(
    const cGH *const cctkGH, const int iteration, const CCTK_REAL time,
    const std::vector<std::int64_t> &stable_ids,
    const SurfaceAnalysisPointOutputOptions &options,
    SurfaceAnalysisPointOutputPerformance *const performance) {
  if (performance)
    *performance = {};
  const amrex::Real total_start = amrex::second();
  validate_point_options(options);
  validate_stable_ids(stable_ids);
  if (!cctkGH || iteration < 0 || !std::isfinite(time) ||
      !(cctkGH->cctk_delta_time > 0) ||
      !std::isfinite(cctkGH->cctk_delta_time))
    amrex::Abort("ParticleAHFinderX received invalid point-frame metadata");

#ifndef HAVE_CAPABILITY_openPMD_api
  amrex::ignore_unused(cctkGH, iteration, time, stable_ids, options);
  amrex::Abort("ParticleAHFinderX analysis point output requires openPMD_api");
#else
  validate_collective_stable_ids(stable_ids);
  if (performance) {
    performance->stable_id_count = static_cast<int>(stable_ids.size());
    performance->metadata_collectives = 2;
  }

  const auto communicator = amrex::ParallelDescriptor::Communicator();
  const int rank = amrex::ParallelDescriptor::MyProc();
  if (rank == 0 && CCTK_CreateDirectory(0755, options.directory) < 0)
    amrex::Abort("ParticleAHFinderX could not create the analysis point-output "
                 "directory");
  MPI_Barrier(communicator);
  if (performance)
    ++performance->metadata_collectives;

  const std::filesystem::path directory(options.directory);
  const std::filesystem::path final_path =
      directory /
      point_frame_name(iteration, stable_ids, options.backend, false);
  const std::filesystem::path temporary_path =
      directory /
      point_frame_name(iteration, stable_ids, options.backend, true);
  int existing = 0;
  if (rank == 0) {
    std::error_code error;
    existing = std::filesystem::exists(final_path, error) ? 1 : 0;
    if (error)
      amrex::Abort("ParticleAHFinderX could not inspect the analysis point "
                   "frame");
  }
  MPI_Bcast(&existing, 1, MPI_INT, 0, communicator);
  if (performance)
    ++performance->metadata_collectives;
  if (existing) {
    if (performance) {
      performance->existing_frame = true;
      performance->total_seconds = amrex::second() - total_start;
    }
    return;
  }

  if (rank == 0) {
    std::error_code error;
    std::filesystem::remove_all(temporary_path, error);
    if (error)
      amrex::Abort("ParticleAHFinderX could not remove an abandoned temporary "
                   "analysis point frame");
  }
  MPI_Barrier(communicator);
  if (performance)
    ++performance->metadata_collectives;

  auto &state = runtime();
  if (!state.logical_surfaces || !state.logical_surfaces->initialized())
    amrex::Abort("ParticleAHFinderX point output has no logical surface "
                 "storage");
  long long expected_global_count = 0;
  for (const std::int64_t stable_id : stable_ids) {
    const auto found = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), stable_id,
        [](const SurfaceRecord &surface_record, const std::int64_t value) {
          return surface_record.stable_id < value;
        });
    if (found == state.surfaces.end() || found->stable_id != stable_id ||
        found->last_search_iteration != iteration ||
        found->relaxation_state != SurfaceRelaxationState::converged ||
        found->lifecycle == ParticleLifecycle::retired)
      amrex::Abort("ParticleAHFinderX point output received a stale or invalid "
                   "surface ID");
    const long long points =
        static_cast<long long>(found->ntheta) * found->nphi;
    if (points <= 0 ||
        expected_global_count >
            std::numeric_limits<long long>::max() - points)
      amrex::Abort("ParticleAHFinderX analysis point count overflowed");
    expected_global_count += points;
  }
  const auto point_view = state.logical_surfaces->analysis_point_device_view(
      options.invariant_spin_requested, options.weyl_curvature_requested);
  const amrex::Long local_population =
      state.logical_surfaces->local_point_count();
  if (local_population > std::numeric_limits<int>::max())
    amrex::Abort("ParticleAHFinderX local logical point population exceeds the "
                 "portable compaction range");
  const int local_population_int = static_cast<int>(local_population);
  std::vector<amrex::Long> ids(stable_ids.begin(), stable_ids.end());
  amrex::Gpu::DeviceVector<amrex::Long> device_ids(ids.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, ids.begin(), ids.end(),
                   device_ids.begin());
  amrex::Gpu::DeviceVector<int> selection_flags(local_population_int);
  amrex::Gpu::DeviceVector<int> selection_offsets(local_population_int);
  const auto *const selected_ids = device_ids.dataPtr();
  const int selected_id_count = static_cast<int>(ids.size());
  auto *const flags = selection_flags.dataPtr();
  auto *const offsets = selection_offsets.dataPtr();
  const auto surface = point_view.surface;
  const amrex::Real selection_start = amrex::second();
  int local_count = 0;
  if (local_population > 0) {
    amrex::ParallelFor(
        local_population,
        [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
          const auto &descriptor =
              surface.descriptors[surface.surface_ordinal[point]];
          flags[point] =
              descriptor.due &&
                      descriptor.angular_level + 1 ==
                          descriptor.angular_levels &&
                      selected_stable_id(descriptor.stable_id, selected_ids,
                                         selected_id_count)
                  ? 1
                  : 0;
        });
    local_count = amrex::Scan::ExclusiveSum(
        local_population_int, flags, offsets, amrex::Scan::retSum);
  }
  amrex::Gpu::DeviceVector<int> selected_points(local_count);
  auto *const compact_points = selected_points.dataPtr();
  if (local_population > 0)
    amrex::ParallelFor(
        local_population,
        [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
          if (flags[point])
            compact_points[offsets[point]] = static_cast<int>(point);
        });
  if (local_population > 0)
    amrex::Gpu::streamSynchronize();
  if (performance) {
    performance->selection_kernel_launches = local_population > 0 ? 2 : 0;
    performance->selection_seconds = amrex::second() - selection_start;
    performance->local_point_count = local_count;
    performance->explicit_device_synchronizations =
        local_population > 0 ? 1 : 0;
  }

  const long long local_count_long = local_count;
  long long rank_offset = 0;
  long long global_count = 0;
  MPI_Exscan(&local_count_long, &rank_offset, 1, MPI_LONG_LONG, MPI_SUM,
             communicator);
  MPI_Allreduce(&local_count_long, &global_count, 1, MPI_LONG_LONG, MPI_SUM,
                communicator);
  if (rank == 0)
    rank_offset = 0;
  if (global_count <= 0)
    amrex::Abort("ParticleAHFinderX point output selected no logical points");
  if (global_count != expected_global_count)
    amrex::Abort("ParticleAHFinderX point output did not select every target "
                 "point exactly once");
  if (performance) {
    performance->metadata_collectives += 2;
    performance->global_point_count = global_count;
  }

  const int chunk_points = options.chunk_points;
  const int local_chunks =
      local_count == 0 ? 0 : (local_count + chunk_points - 1) / chunk_points;
  int maximum_chunks = 0;
  MPI_Allreduce(&local_chunks, &maximum_chunks, 1, MPI_INT, MPI_MAX,
                communicator);
  if (performance)
    ++performance->metadata_collectives;

  const int real_components =
      options.weyl_curvature_requested ? point_weyl_real_components
                                       : point_base_real_components;
  if (performance) {
    const int local_chunk_capacity =
        std::min(options.chunk_points, local_count);
    performance->device_workspace_bytes =
        static_cast<std::int64_t>(ids.size() * sizeof(amrex::Long)) +
        static_cast<std::int64_t>(2 * local_population_int + local_count) *
            sizeof(int) +
        static_cast<std::int64_t>(local_chunk_capacity) *
            (real_components * sizeof(amrex::Real) +
             point_int_components * sizeof(std::int32_t) +
             sizeof(std::int64_t));
  }

  const amrex::Real output_start = amrex::second();
  openPMD::Series series(temporary_path.string(), openPMD::Access::CREATE,
                         communicator);
  series.setSoftware("ParticleAHFinderX", "0.1.0");
  auto &output_iteration = series.iterations[iteration];
  output_iteration.setTime(static_cast<double>(time));
  output_iteration.setDt(static_cast<double>(cctkGH->cctk_delta_time));
  output_iteration.setAttribute("schemaVersion", surface_analysis_point_schema);
  output_iteration.setAttribute("cactusTime", static_cast<double>(time));
  output_iteration.setAttribute("coordinateSystem", std::string("Cartesian"));
  output_iteration.setAttribute(
      "angularTopology", std::string("cell_centered_latitude_longitude"));
  output_iteration.setAttribute("coordinateUnits",
                                std::string("Cactus code units"));
  output_iteration.setAttribute("stableIdCount",
                                static_cast<std::uint64_t>(stable_ids.size()));
  output_iteration.setAttribute("stableIdHash", stable_id_hash(stable_ids));
  output_iteration.setAttribute("weylFieldsIncluded",
                                options.weyl_curvature_requested);
  auto &species = output_iteration.particles[analysis_point_dataset_name];
  species.setAttribute(
      "description",
      std::string("GPU-native intrinsic horizon-analysis target points"));
  species.setAttribute("angularIndexing",
                       std::string("zero-based latitude-longitude cells"));
  species.setAttribute("lifecycleCodes",
                       std::string("0=candidate;1=active;2=retired"));
  species.setAttribute("roleCodes",
                       std::string("0=individual;1=common;2=other"));
  species.setAttribute(
      "normalConnectionConvention",
      std::string("omega_A=-e_A^i K_ij s^j"));
  species.setAttribute(
      "validityMeaning",
      std::string("intrinsicValid, axialValid, and weylValid independently "
                  "guard their corresponding point fields"));
  define_point_components(species, static_cast<std::uint64_t>(global_count),
                          options.weyl_curvature_requested);

  for (int chunk = 0; chunk < maximum_chunks; ++chunk) {
    const int begin = chunk * chunk_points;
    const int count =
        begin < local_count ? std::min(chunk_points, local_count - begin) : 0;
    if (count > 0) {
      const amrex::Real packing_start = amrex::second();
      amrex::Gpu::DeviceVector<amrex::Real> device_reals(
          static_cast<std::size_t>(real_components) * count);
      amrex::Gpu::DeviceVector<std::int32_t> device_ints(
          static_cast<std::size_t>(point_int_components) * count);
      amrex::Gpu::DeviceVector<std::int64_t> device_longs(count);
      auto *const reals = device_reals.dataPtr();
      auto *const ints = device_ints.dataPtr();
      auto *const longs = device_longs.dataPtr();
      const bool axial_requested = options.invariant_spin_requested;
      const bool weyl_requested = options.weyl_curvature_requested;
      amrex::ParallelFor(
          count, [=] AMREX_GPU_DEVICE(const int slot) noexcept {
            const amrex::Long point = compact_points[begin + slot];
            const int ordinal = surface.surface_ordinal[point];
            const auto &descriptor = surface.descriptors[ordinal];
            reals[point_x * count + slot] = surface.position[0][point];
            reals[point_y * count + slot] = surface.position[1][point];
            reals[point_z * count + slot] = surface.position[2][point];
            const bool intrinsic_valid =
                point_view.intrinsic_status[point] == 0;
            for (int component = 0; component < 3; ++component)
              reals[(point_q_theta_theta + component) * count + slot] =
                  intrinsic_valid
                      ? point_view.intrinsic_metric[
                            amrex::Long(component) * surface.point_count + point]
                      : 0;
            reals[point_intrinsic_scalar_curvature * count + slot] =
                intrinsic_valid
                    ? point_view.intrinsic_scalar_curvature[point]
                    : 0;
            reals[point_normal_connection_theta * count + slot] =
                intrinsic_valid
                    ? point_view.intrinsic_normal_connection[point]
                    : 0;
            reals[point_normal_connection_phi * count + slot] =
                intrinsic_valid
                    ? point_view.intrinsic_normal_connection[
                          surface.point_count + point]
                    : 0;
            const amrex::Real weight = surface.area_weight[point];
            reals[point_area_weight * count + slot] =
                weight > 0 && amrex::Math::isfinite(weight) ? weight : 0;

            bool axial_valid =
                axial_requested && point_view.axial_workspace_available &&
                intrinsic_valid &&
                point_view.surface_axial_status[ordinal] ==
                    static_cast<int>(AxialSymmetryStatus::valid);
            amrex::Real invariant_coordinate = 0;
            amrex::Real axial_theta = 0;
            amrex::Real axial_phi = 0;
            if (axial_valid) {
              const int surface_count = surface.surface_count;
              const amrex::Real scale =
                  point_view.surface_axial_values[ordinal];
              const amrex::Real mean =
                  point_view.surface_axial_values[8 * surface_count + ordinal];
              const amrex::Real inverse_radius_squared =
                  point_view.surface_axial_values[11 * surface_count + ordinal];
              invariant_coordinate =
                  scale * (point_view.axial_potential[point] - mean) *
                  inverse_radius_squared;
              axial_theta = scale * point_view.axial_vector[point];
              axial_phi = scale *
                           point_view.axial_vector[surface.point_count + point];
              axial_valid =
                  amrex::Math::isfinite(invariant_coordinate) &&
                  amrex::Math::isfinite(axial_theta) &&
                  amrex::Math::isfinite(axial_phi);
              if (!axial_valid) {
                invariant_coordinate = 0;
                axial_theta = 0;
                axial_phi = 0;
              }
            }
            reals[point_invariant_coordinate * count + slot] =
                invariant_coordinate;
            reals[point_axial_vector_theta * count + slot] = axial_theta;
            reals[point_axial_vector_phi * count + slot] = axial_phi;

            bool weyl_valid =
                weyl_requested && point_view.weyl_workspace_available &&
                point_view.weyl_arrival[point] == 1;
            if (weyl_requested) {
              for (int component = 0; component < 6; ++component) {
                const amrex::Real electric =
                    weyl_valid
                        ? point_view.weyl_electric[
                              amrex::Long(component) * surface.point_count +
                              point]
                        : 0;
                const amrex::Real magnetic =
                    weyl_valid
                        ? point_view.weyl_magnetic[
                              amrex::Long(component) * surface.point_count +
                              point]
                        : 0;
                if (!amrex::Math::isfinite(electric) ||
                    !amrex::Math::isfinite(magnetic))
                  weyl_valid = false;
                reals[(point_electric_xx + component) * count + slot] =
                    electric;
                reals[(point_magnetic_xx + component) * count + slot] =
                    magnetic;
              }
              if (!weyl_valid)
                for (int component = point_electric_xx;
                     component < point_weyl_real_components; ++component)
                  reals[component * count + slot] = 0;
            }

            longs[slot] = static_cast<std::int64_t>(descriptor.stable_id);
            ints[point_generation * count + slot] = descriptor.generation;
            ints[point_angular_level * count + slot] =
                descriptor.angular_level;
            ints[point_angular_levels * count + slot] =
                descriptor.angular_levels;
            ints[point_itheta * count + slot] = surface.itheta[point];
            ints[point_iphi * count + slot] = surface.iphi[point];
            ints[point_ntheta * count + slot] = descriptor.ntheta;
            ints[point_nphi * count + slot] = descriptor.nphi;
            ints[point_lifecycle * count + slot] = descriptor.lifecycle;
            ints[point_role * count + slot] = descriptor.role;
            ints[point_intrinsic_valid * count + slot] = intrinsic_valid;
            ints[point_axial_valid * count + slot] = axial_valid;
            ints[point_weyl_valid * count + slot] = weyl_valid;
          });
      amrex::Gpu::streamSynchronize();
      if (performance) {
        ++performance->packing_kernel_launches;
        performance->packing_seconds += amrex::second() - packing_start;
        ++performance->explicit_device_synchronizations;
      }

      auto host_reals = pinned_array<amrex::Real>(device_reals.size());
      auto host_ints = pinned_array<std::int32_t>(device_ints.size());
      auto host_longs = pinned_array<std::int64_t>(device_longs.size());
      const amrex::Real transfer_start = amrex::second();
      amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, device_reals.begin(),
                            device_reals.end(), host_reals.get());
      amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, device_ints.begin(),
                            device_ints.end(), host_ints.get());
      amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, device_longs.begin(),
                            device_longs.end(), host_longs.get());
      amrex::Gpu::streamSynchronize();
      if (performance) {
        performance->device_to_host_copies += 3;
        ++performance->explicit_device_synchronizations;
        performance->pinned_host_bytes +=
            static_cast<std::int64_t>(device_reals.size() *
                                          sizeof(amrex::Real) +
                                      device_ints.size() *
                                          sizeof(std::int32_t) +
                                      device_longs.size() *
                                          sizeof(std::int64_t));
        performance->device_to_host_seconds +=
            amrex::second() - transfer_start;
      }

      const openPMD::Offset offset{
          static_cast<std::uint64_t>(rank_offset + begin)};
      const openPMD::Extent extent{static_cast<std::uint64_t>(count)};
      store_real_components(species, host_reals, count, offset, extent,
                            options.weyl_curvature_requested);
      species["surfaceId"][openPMD::RecordComponent::SCALAR].storeChunk(
          host_longs, offset, extent);
      store_int_components(species, host_ints, count, offset, extent);
    }
    // Every rank flushes the same number of times even when owner loads differ.
    series.flush();
  }
  output_iteration.close();
  series.flush();
  series.close();

  MPI_Barrier(communicator);
  if (rank == 0) {
    std::error_code error;
    std::filesystem::rename(temporary_path, final_path, error);
    if (error)
      amrex::Abort("ParticleAHFinderX could not commit the analysis point "
                   "frame");
  }
  MPI_Barrier(communicator);
  if (performance) {
    performance->wrote_frame = true;
    performance->output_seconds = amrex::second() - output_start;
    performance->total_seconds = amrex::second() - total_start;
    long long local_counters[6]{
        performance->selection_kernel_launches,
        performance->packing_kernel_launches,
        performance->device_to_host_copies,
        static_cast<long long>(performance->device_workspace_bytes),
        static_cast<long long>(performance->pinned_host_bytes),
        performance->explicit_device_synchronizations};
    long long global_counters[6]{};
    MPI_Allreduce(local_counters, global_counters, 6, MPI_LONG_LONG, MPI_SUM,
                  communicator);
    performance->selection_kernel_launches =
        static_cast<int>(global_counters[0]);
    performance->packing_kernel_launches =
        static_cast<int>(global_counters[1]);
    performance->device_to_host_copies =
        static_cast<int>(global_counters[2]);
    performance->device_workspace_bytes = global_counters[3];
    performance->pinned_host_bytes = global_counters[4];
    performance->explicit_device_synchronizations =
        static_cast<int>(global_counters[5]);
    double local_times[5]{double(performance->selection_seconds),
                          double(performance->packing_seconds),
                          double(performance->device_to_host_seconds),
                          double(performance->output_seconds),
                          double(performance->total_seconds)};
    double maximum_times[5]{};
    MPI_Allreduce(local_times, maximum_times, 5, MPI_DOUBLE, MPI_MAX,
                  communicator);
    performance->selection_seconds = maximum_times[0];
    performance->packing_seconds = maximum_times[1];
    performance->device_to_host_seconds = maximum_times[2];
    performance->output_seconds = maximum_times[3];
    performance->total_seconds = maximum_times[4];
    performance->metadata_collectives += 4;
  }
#endif
}

void reconcile_surface_analysis_point_output(
    const cGH *const cctkGH, const char *const directory,
    const char *const backend, const int recovered_iteration) {
  if (!cctkGH || !directory || directory[0] == '\0' || !backend ||
      std::string(backend) != "bp5" || recovered_iteration < 0)
    amrex::Abort("ParticleAHFinderX received invalid point-output recovery "
                 "metadata");
#ifndef HAVE_CAPABILITY_openPMD_api
  amrex::ignore_unused(cctkGH, directory, backend, recovered_iteration);
  return;
#else
  if (amrex::ParallelDescriptor::MyProc() == 0) {
    const std::filesystem::path root(directory);
    std::error_code error;
    const bool exists = std::filesystem::exists(root, error);
    if (error)
      amrex::Abort("ParticleAHFinderX could not inspect the analysis point "
                   "output directory during recovery");
    if (exists) {
      for (const auto &entry : std::filesystem::directory_iterator(root)) {
        int frame_iteration = -1;
        bool temporary = false;
        if (!recognized_point_frame(entry.path().filename().string(), backend,
                                    frame_iteration, temporary) ||
            (!temporary && frame_iteration <= recovered_iteration))
          continue;
        std::filesystem::remove_all(entry.path(), error);
        if (error)
          amrex::Abort("ParticleAHFinderX could not reconcile a future or "
                       "temporary analysis point frame");
      }
    }
  }
  MPI_Barrier(amrex::ParallelDescriptor::Communicator());
#endif
}

} // namespace ParticleAHFinderX
