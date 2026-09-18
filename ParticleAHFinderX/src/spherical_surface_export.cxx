/**
 * \file spherical_surface_export.cxx
 * \brief Bounded optional export of converged surfaces for legacy consumers.
 *
 * The solver never reads this compatibility representation. Resampling occurs
 * on the logical owner GPU, followed by one bounded target-grid host transfer
 * and broadcast per mapped, newly converged surface.
 */
#include "runtime.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Functions.h>
#include <cctk_Parameters.h>

#include <driver.hxx>

#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <vector>

namespace ParticleAHFinderX {
namespace {

struct SurfaceVariables {
  CCTK_REAL *active = nullptr;
  CCTK_REAL *valid = nullptr;
  CCTK_REAL *radius = nullptr;
  CCTK_REAL *origin[3]{nullptr, nullptr, nullptr};
  CCTK_REAL *area = nullptr;
  CCTK_REAL *mean_radius = nullptr;
  CCTK_REAL *centroid[3]{nullptr, nullptr, nullptr};
  CCTK_REAL *quadrupole[6]{nullptr, nullptr, nullptr,
                           nullptr, nullptr, nullptr};
  CCTK_REAL *minimum_radius = nullptr;
  CCTK_REAL *maximum_radius = nullptr;
  CCTK_REAL *position_minimum[3]{nullptr, nullptr, nullptr};
  CCTK_REAL *position_maximum[3]{nullptr, nullptr, nullptr};
  CCTK_REAL *ntheta = nullptr;
  CCTK_REAL *nphi = nullptr;
  CCTK_REAL *nghoststheta = nullptr;
  CCTK_REAL *nghostsphi = nullptr;
  CCTK_REAL *origin_theta = nullptr;
  CCTK_REAL *origin_phi = nullptr;
  CCTK_REAL *delta_theta = nullptr;
  CCTK_REAL *delta_phi = nullptr;
  int surface_count = 0;
  int maximum_ntheta = 0;
  int maximum_nphi = 0;
};

CCTK_REAL *real_data(const cGH *const cctkGH, const char *const name) {
  int variable = CCTK_VarIndex(name);
  std::string registered_name(name);
  if (variable < 0) {
    // SphericalSurface declares one Cactus vector group per surface. Cactus
    // registers the storage base with an explicit element index such as
    // SphericalSurface::sf_active[0], while the CCL interface and generated
    // C arguments expose the unindexed group name. The first element pointer
    // is the contiguous base used by existing SphericalSurface consumers.
    registered_name += "[0]";
    variable = CCTK_VarIndex(registered_name.c_str());
  }
  if (variable < 0 || CCTK_VarTypeI(variable) != CCTK_VARIABLE_REAL)
    CCTK_VERROR("ParticleAHFinderX cannot resolve compatibility variable "
                "'%s'",
                name);
  auto *const data =
      static_cast<CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, 0, variable));
  if (!data)
    CCTK_VERROR("ParticleAHFinderX compatibility variable '%s' has no "
                "storage; ensure SphericalSurface is active",
                name);
  return data;
}

int integer_parameter(const char *const name) {
  int type = -1;
  const void *const value = CCTK_ParameterGet(name, "SphericalSurface", &type);
  if (!value || type != PARAMETER_INT)
    CCTK_VERROR("ParticleAHFinderX cannot read SphericalSurface::%s", name);
  const CCTK_INT result = *static_cast<const CCTK_INT *>(value);
  if (result < 0 ||
      result > static_cast<CCTK_INT>(std::numeric_limits<int>::max()))
    CCTK_VERROR("ParticleAHFinderX found an invalid SphericalSurface::%s=%lld",
                name, static_cast<long long>(result));
  return static_cast<int>(result);
}

SurfaceVariables surface_variables(const cGH *const cctkGH) {
  SurfaceVariables variables;
  variables.surface_count = integer_parameter("nsurfaces");
  variables.maximum_ntheta = integer_parameter("maxntheta");
  variables.maximum_nphi = integer_parameter("maxnphi");
  variables.active = real_data(cctkGH, "SphericalSurface::sf_active");
  variables.valid = real_data(cctkGH, "SphericalSurface::sf_valid");
  variables.radius = real_data(cctkGH, "SphericalSurface::sf_radius");
  variables.origin[0] =
      real_data(cctkGH, "SphericalSurface::sf_origin_x");
  variables.origin[1] =
      real_data(cctkGH, "SphericalSurface::sf_origin_y");
  variables.origin[2] =
      real_data(cctkGH, "SphericalSurface::sf_origin_z");
  variables.area = real_data(cctkGH, "SphericalSurface::sf_area");
  variables.mean_radius =
      real_data(cctkGH, "SphericalSurface::sf_mean_radius");
  variables.centroid[0] =
      real_data(cctkGH, "SphericalSurface::sf_centroid_x");
  variables.centroid[1] =
      real_data(cctkGH, "SphericalSurface::sf_centroid_y");
  variables.centroid[2] =
      real_data(cctkGH, "SphericalSurface::sf_centroid_z");
  constexpr std::array<const char *, 6> quadrupole_names{{
      "SphericalSurface::sf_quadrupole_xx",
      "SphericalSurface::sf_quadrupole_xy",
      "SphericalSurface::sf_quadrupole_xz",
      "SphericalSurface::sf_quadrupole_yy",
      "SphericalSurface::sf_quadrupole_yz",
      "SphericalSurface::sf_quadrupole_zz",
  }};
  for (int component = 0; component < 6; ++component)
    variables.quadrupole[component] =
        real_data(cctkGH, quadrupole_names[component]);
  variables.minimum_radius =
      real_data(cctkGH, "SphericalSurface::sf_min_radius");
  variables.maximum_radius =
      real_data(cctkGH, "SphericalSurface::sf_max_radius");
  constexpr std::array<const char *, 3> minimum_names{{
      "SphericalSurface::sf_min_x", "SphericalSurface::sf_min_y",
      "SphericalSurface::sf_min_z",
  }};
  constexpr std::array<const char *, 3> maximum_names{{
      "SphericalSurface::sf_max_x", "SphericalSurface::sf_max_y",
      "SphericalSurface::sf_max_z",
  }};
  for (int d = 0; d < 3; ++d) {
    variables.position_minimum[d] = real_data(cctkGH, minimum_names[d]);
    variables.position_maximum[d] = real_data(cctkGH, maximum_names[d]);
  }
  variables.ntheta = real_data(cctkGH, "SphericalSurface::sf_ntheta");
  variables.nphi = real_data(cctkGH, "SphericalSurface::sf_nphi");
  variables.nghoststheta =
      real_data(cctkGH, "SphericalSurface::sf_nghoststheta");
  variables.nghostsphi =
      real_data(cctkGH, "SphericalSurface::sf_nghostsphi");
  variables.origin_theta =
      real_data(cctkGH, "SphericalSurface::sf_origin_theta");
  variables.origin_phi =
      real_data(cctkGH, "SphericalSurface::sf_origin_phi");
  variables.delta_theta =
      real_data(cctkGH, "SphericalSurface::sf_delta_theta");
  variables.delta_phi =
      real_data(cctkGH, "SphericalSurface::sf_delta_phi");
  return variables;
}

int shape_dimension(const CCTK_REAL value, const char *const name,
                    const int target) {
  if (!amrex::Math::isfinite(value) || value < 0 ||
      value > std::numeric_limits<int>::max())
    CCTK_VERROR("ParticleAHFinderX target SphericalSurface %d has invalid "
                "%s=%.17g",
                target, name, static_cast<double>(value));
  const int dimension = static_cast<int>(value);
  if (value != dimension)
    CCTK_VERROR("ParticleAHFinderX target SphericalSurface %d has "
                "non-integral %s=%.17g",
                target, name, static_cast<double>(value));
  return dimension;
}

int mapped_surface(const int slot, const int surface_count) {
  DECLARE_CCTK_PARAMETERS;
  const int fallback = spherical_surface_index[slot];
  const char *const name = spherical_surface_name[slot];
  if (fallback < 0 && name[0] == '\0')
    return -1;
  const int target = sf_IdFromName(fallback, name);
  if (target < 0 || target >= surface_count)
    CCTK_VERROR("ParticleAHFinderX compatibility slot %d maps to invalid "
                "SphericalSurface target %d; valid targets are [0,%d)",
                slot + 1, target, surface_count);
  return target;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
positive_modulo(const int value, const int modulus) noexcept {
  const int remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cubic_weights(const amrex::Real fraction, amrex::Real *const weights) noexcept {
  weights[0] = -fraction * (fraction - 1) * (fraction - 2) / 6;
  weights[1] = (fraction + 1) * (fraction - 1) * (fraction - 2) / 2;
  weights[2] = -(fraction + 1) * fraction * (fraction - 2) / 2;
  weights[3] = (fraction + 1) * fraction * (fraction - 1) / 6;
}

std::vector<amrex::Real>
resample_on_owner(Runtime &state, const SurfaceRecord &surface,
                  const int target_ntheta, const int target_nphi,
                  const amrex::Real origin_theta,
                  const amrex::Real origin_phi,
                  const amrex::Real delta_theta,
                  const amrex::Real delta_phi) {
  const amrex::Long target_count_long =
      amrex::Long(target_ntheta) * target_nphi;
  if (target_count_long <= 0 ||
      target_count_long > std::numeric_limits<int>::max())
    CCTK_ERROR("ParticleAHFinderX compatibility target exceeds the MPI and "
               "portable kernel index range");
  const int target_count = static_cast<int>(target_count_long);
  std::vector<amrex::Real> host_radius(target_count);
  if (amrex::ParallelDescriptor::MyProc() != surface.logical_owner_rank)
    return host_radius;

  const auto source = state.logical_surfaces->device_view();
  if (surface.logical_owner_offset < 0 || surface.particle_count <= 0 ||
      surface.particle_count > source.point_count ||
      surface.logical_owner_offset >
          source.point_count - surface.particle_count)
    CCTK_ERROR("ParticleAHFinderX compatibility export found invalid logical "
               "owner bounds");
  amrex::Gpu::DeviceVector<amrex::Real> device_radius(target_count);
  auto *const target = device_radius.dataPtr();
  const auto *const height =
      source.height + surface.logical_owner_offset;
  const int source_ntheta = surface.ntheta;
  const int source_nphi = surface.nphi;
  const amrex::Real pi = amrex::Math::pi<amrex::Real>();
  const amrex::Real two_pi = 2 * pi;
  const amrex::Real source_dtheta = pi / source_ntheta;
  const amrex::Real source_dphi = two_pi / source_nphi;

  amrex::ParallelFor(
      target_count, [=] AMREX_GPU_DEVICE(const int point) noexcept {
        const int itheta = point % target_ntheta;
        const int iphi = point / target_ntheta;
        amrex::Real theta = origin_theta + itheta * delta_theta;
        amrex::Real phi = origin_phi + iphi * delta_phi;
        theta -= two_pi * amrex::Math::floor(theta / two_pi);
        if (theta < 0)
          theta += two_pi;
        if (theta > pi) {
          theta = two_pi - theta;
          phi += pi;
        }
        phi -= two_pi * amrex::Math::floor(phi / two_pi);

        const amrex::Real theta_coordinate = theta / source_dtheta - 0.5;
        const amrex::Real phi_coordinate = phi / source_dphi - 0.5;
        const int theta_base =
            static_cast<int>(amrex::Math::floor(theta_coordinate));
        const int phi_base =
            static_cast<int>(amrex::Math::floor(phi_coordinate));
        amrex::Real theta_weight[4];
        amrex::Real phi_weight[4];
        cubic_weights(theta_coordinate - theta_base, theta_weight);
        cubic_weights(phi_coordinate - phi_base, phi_weight);

        amrex::Real value = 0;
        for (int a = 0; a < 4; ++a) {
          int source_i = theta_base + a - 1;
          int phi_shift = 0;
          while (source_i < 0 || source_i >= source_ntheta) {
            if (source_i < 0)
              source_i = -source_i - 1;
            else
              source_i = 2 * source_ntheta - source_i - 1;
            phi_shift += source_nphi / 2;
          }
          for (int b = 0; b < 4; ++b) {
            const int source_j = positive_modulo(
                phi_base + b - 1 + phi_shift, source_nphi);
            value += theta_weight[a] * phi_weight[b] *
                     height[amrex::Long(source_i) * source_nphi + source_j];
          }
        }
        target[point] = value;
      });
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_radius.begin(),
                   device_radius.end(), host_radius.begin());
  return host_radius;
}

void broadcast_radius(std::vector<amrex::Real> &radius, const int owner) {
  static_assert(sizeof(amrex::Real) == sizeof(float) ||
                    sizeof(amrex::Real) == sizeof(double),
                "ParticleAHFinderX requires float or double AMReX Real");
  if (radius.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    CCTK_ERROR("ParticleAHFinderX compatibility broadcast exceeds MPI count");
  const MPI_Datatype datatype =
      sizeof(amrex::Real) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
  const int error = MPI_Bcast(
      radius.data(), static_cast<int>(radius.size()), datatype, owner,
      amrex::ParallelDescriptor::Communicator());
  if (error != MPI_SUCCESS)
    CCTK_ERROR("ParticleAHFinderX compatibility radius broadcast failed");
}

void publish_metadata(const SurfaceVariables &variables, const int target,
                      const SurfaceRecord &surface) {
  for (int d = 0; d < 3; ++d) {
    variables.origin[d][target] = surface.center[d];
    variables.centroid[d][target] = surface.centroid[d];
    variables.position_minimum[d][target] = surface.position_minimum[d];
    variables.position_maximum[d][target] = surface.position_maximum[d];
  }
  variables.area[target] = surface.area;
  variables.mean_radius[target] = surface.mean_radius;
  for (int component = 0; component < 6; ++component)
    variables.quadrupole[component][target] =
        surface.coordinate_quadrupole[component];
  variables.minimum_radius[target] = surface.minimum_radius;
  variables.maximum_radius[target] = surface.maximum_radius;
}

} // namespace

extern "C" void ParticleAHFinderX_ExportSphericalSurfaces(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  if (!export_spherical_surfaces)
    return;
  auto &state = runtime();
  if (!state.initial_surfaces_created || !state.logical_surfaces ||
      state.last_spherical_export_iteration == cctk_iteration)
    return;

  bool has_mapping = false;
  for (int slot = 0; slot < max_compatibility_slots; ++slot)
    has_mapping = has_mapping || spherical_surface_index[slot] >= 0 ||
                  spherical_surface_name[slot][0] != '\0';
  if (!has_mapping) {
    state.last_spherical_export_iteration = cctk_iteration;
    return;
  }

  if (!CCTK_IsThornActive("SphericalSurface") ||
      !CCTK_IsFunctionAliased("sf_IdFromName"))
    CCTK_ERROR("ParticleAHFinderX compatibility export lost its active "
               "SphericalSurface provider");

  const SurfaceVariables variables = surface_variables(cctkGH);
  std::vector<int> slot_for_target(variables.surface_count, -1);
  std::array<int, max_compatibility_slots> target_for_slot{};
  target_for_slot.fill(-1);
  for (int slot = 0; slot < max_compatibility_slots; ++slot) {
    const int target = mapped_surface(slot, variables.surface_count);
    target_for_slot[slot] = target;
    if (target < 0)
      continue;
    if (slot_for_target[target] >= 0)
      CCTK_VERROR("ParticleAHFinderX compatibility slots %d and %d both map "
                  "to SphericalSurface target %d",
                  slot_for_target[target] + 1, slot + 1, target);
    slot_for_target[target] = slot;
  }

  const bool synchronized = CarpetX::all_levels_synchronized();
  int exported_surfaces = 0;
  for (int slot = 0; slot < max_compatibility_slots; ++slot) {
    const int target = target_for_slot[slot];
    if (target < 0)
      continue;
    const auto found = std::find_if(
        state.surfaces.begin(), state.surfaces.end(),
        [=](const SurfaceRecord &surface) {
          return surface.compatibility_slot == slot + 1;
        });
    if (found == state.surfaces.end() || !found->published ||
        found->lifecycle == ParticleLifecycle::retired) {
      variables.active[target] = 0;
      variables.valid[target] = 0;
      continue;
    }

    const auto &surface = *found;
    variables.active[target] = 1;
    if (!synchronized || surface.last_search_iteration != cctk_iteration) {
      variables.valid[target] = 0;
      continue;
    }
    if (surface.relaxation_state != SurfaceRelaxationState::converged ||
        surface.invalid_points != 0 || !(surface.area > 0)) {
      variables.valid[target] = -1;
      continue;
    }

    const int target_ntheta =
        shape_dimension(variables.ntheta[target], "ntheta", target);
    const int target_nphi =
        shape_dimension(variables.nphi[target], "nphi", target);
    const int target_nghoststheta =
        shape_dimension(variables.nghoststheta[target], "nghoststheta",
                        target);
    const int target_nghostsphi =
        shape_dimension(variables.nghostsphi[target], "nghostsphi", target);
    if (target_ntheta <= 0 || target_nphi <= 0 || target_nghoststheta < 0 ||
        target_nghostsphi < 0 ||
        target_ntheta > variables.maximum_ntheta ||
        target_nphi > variables.maximum_nphi ||
        target_ntheta < 2 * target_nghoststheta + 1 ||
        target_nphi < 2 * target_nghostsphi + 1)
      CCTK_VERROR("ParticleAHFinderX target SphericalSurface %d has invalid "
                  "shape %d x %d with ghost widths %d x %d",
                  target, target_ntheta, target_nphi, target_nghoststheta,
                  target_nghostsphi);

    auto radius = resample_on_owner(
        state, surface, target_ntheta, target_nphi,
        variables.origin_theta[target], variables.origin_phi[target],
        variables.delta_theta[target], variables.delta_phi[target]);
    broadcast_radius(radius, surface.logical_owner_rank);
    for (int j = 0; j < target_nphi; ++j)
      for (int i = 0; i < target_ntheta; ++i)
        variables.radius[i + variables.maximum_ntheta *
                                (j + variables.maximum_nphi * target)] =
            radius[i + target_ntheta * j];
    publish_metadata(variables, target, surface);
    variables.valid[target] = 1;
    ++exported_surfaces;
  }
  if (verbose && exported_surfaces > 0)
    CCTK_VINFO("Exported %d converged persistent surface(s) to "
               "SphericalSurface at iteration %d",
               exported_surfaces, cctk_iteration);
  state.last_spherical_export_iteration = cctk_iteration;
}

} // namespace ParticleAHFinderX
