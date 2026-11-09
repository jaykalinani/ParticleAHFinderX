/**
 * \file logical_surface_storage.cxx
 * \brief Batched initialization of persistent logical surfaces on owner GPUs.
 */
#include "logical_surface_storage.hxx"
#include "angular_geometry.hxx"
#include "intrinsic_geometry.hxx"
#include "interpolation.hxx"
#include "multipoles.hxx"

#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_Math.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <limits>

namespace ParticleAHFinderX {
namespace {

struct SurfaceConvergenceValues {
  amrex::Real ratio;
  amrex::Real l2_times_mass;
  amrex::Real linf_times_mass;
  bool geometry_valid;
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
low_mode_extrapolation_displacement(
    const LogicalSurfaceDeviceDescriptor &surface, const amrex::Long point,
    const int ordinal, const int surface_count,
    const amrex::Real *const mode_displacement) noexcept {
  const amrex::Long logical_index = point - surface.offset;
  const int itheta = static_cast<int>(logical_index / surface.nphi);
  const int iphi = static_cast<int>(logical_index % surface.nphi);
  const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                            (itheta + amrex::Real(0.5)) / surface.ntheta;
  const amrex::Real phi = 2 * amrex::Math::pi<amrex::Real>() *
                          (iphi + amrex::Real(0.5)) / surface.nphi;
  const AngularGeometry basis = angular_geometry(theta, phi);
  amrex::Real displacement = mode_displacement[ordinal];
  for (int d = 0; d < 3; ++d)
    displacement += mode_displacement[(d + 1) * surface_count + ordinal] *
                    basis.radial[d];
  return displacement;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE SurfaceConvergenceValues
surface_convergence_values(
    const LogicalSurfaceDeviceDescriptor &surface,
    const amrex::Real area, const amrex::Real theta_squared,
    const amrex::Real theta_linf, const amrex::Long invalid) noexcept {
  SurfaceConvergenceValues values{0, 0, 0, area > 0 && invalid == 0};
  if (!values.geometry_valid)
    return values;
  const amrex::Real mean_squared = theta_squared / area;
  const amrex::Real l2 =
      mean_squared > 0 ? mean_squared * amrex::Math::rsqrt(mean_squared) : 0;
  values.l2_times_mass = l2 * surface.mass_scale;
  values.linf_times_mass = theta_linf * surface.mass_scale;
  values.ratio = amrex::max(
      values.l2_times_mass / surface.theta_l2_tolerance,
      values.linf_times_mass / surface.theta_linf_tolerance);
  return values;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
update_surface_relaxation_state(
    LogicalSurfaceDeviceDescriptor &surface,
    const SurfaceConvergenceValues &values,
    const amrex::Real cutoff_indicator,
    const bool accept_convergence,
    const amrex::Real cutoff_indicator_threshold,
    const amrex::Real helper_tolerance_factor,
    const bool fail_invalid_geometry) noexcept {
  if (surface.relaxation_state !=
      static_cast<int>(SurfaceRelaxationState::active))
    return;
  if (!values.geometry_valid) {
    if (fail_invalid_geometry)
      surface.relaxation_state =
          static_cast<int>(SurfaceRelaxationState::failed);
    return;
  }
  if (accept_convergence && cutoff_indicator_threshold > 0 &&
      surface.angular_level + 1 < surface.angular_levels &&
      cutoff_indicator > cutoff_indicator_threshold) {
    surface.relaxation_state =
        static_cast<int>(SurfaceRelaxationState::refine);
    return;
  }
  const amrex::Real accepted_ratio =
      surface.relaxed_helper_tolerance &&
              surface.angular_level + 1 < surface.angular_levels
          ? helper_tolerance_factor
          : amrex::Real(1);
  if (accept_convergence && values.ratio <= accepted_ratio)
    surface.relaxation_state =
        static_cast<int>(SurfaceRelaxationState::converged);
}

template <typename T>
void grow_component_major(amrex::Gpu::DeviceVector<T> &values,
                          const int components,
                          const amrex::Long old_point_count,
                          const amrex::Long new_point_count) {
  amrex::Gpu::DeviceVector<T> expanded(
      static_cast<std::size_t>(components * new_point_count));
  for (int component = 0; component < components; ++component)
    amrex::Gpu::copy(
        amrex::Gpu::deviceToDevice,
        values.begin() + component * old_point_count,
        values.begin() + (component + 1) * old_point_count,
        expanded.begin() + component * new_point_count);
  values.swap(expanded);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
positive_modulo(const int value, const int modulus) noexcept {
  const int remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
angular_status(const int *const status, const amrex::Long offset,
               const int ntheta, const int nphi, int itheta,
               int iphi) noexcept {
  while (itheta < 0 || itheta >= ntheta) {
    if (itheta < 0)
      itheta = -itheta - 1;
    else
      itheta = 2 * ntheta - itheta - 1;
    iphi += nphi / 2;
  }
  iphi = positive_modulo(iphi, nphi);
  return status[offset + amrex::Long(itheta) * nphi + iphi];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
intrinsic_stencil_valid(const int *const status, const amrex::Long offset,
                        const int ntheta, const int nphi, const int itheta,
                        const int iphi) noexcept {
  for (int stencil = -3; stencil <= 3; ++stencil)
    if (angular_status(status, offset, ntheta, nphi, itheta + stencil,
                       iphi) != 0 ||
        angular_status(status, offset, ntheta, nphi, itheta,
                       iphi + stencil) != 0)
      return false;
  return true;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
selected_surface_layer(const LogicalSurfaceDeviceDescriptor &surface,
                       const SurfaceLayerSelection *const selections,
                       const int selection_count) noexcept {
  int lower = 0;
  int upper = selection_count;
  while (lower < upper) {
    const int middle = lower + (upper - lower) / 2;
    if (selections[middle].stable_id < surface.stable_id ||
        (selections[middle].stable_id == surface.stable_id &&
         selections[middle].angular_level < surface.angular_level))
      lower = middle + 1;
    else
      upper = middle;
  }
  return lower < selection_count &&
         selections[lower].stable_id == surface.stable_id &&
         selections[lower].angular_level == surface.angular_level;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
surface_extrapolation_control(
    const LogicalSurfaceDeviceDescriptor &surface,
    const SurfaceExtrapolationControl *const controls,
    const int control_count) noexcept {
  int lower = 0;
  int upper = control_count;
  while (lower < upper) {
    const int middle = lower + (upper - lower) / 2;
    if (controls[middle].stable_id < surface.stable_id ||
        (controls[middle].stable_id == surface.stable_id &&
         controls[middle].angular_level < surface.angular_level))
      lower = middle + 1;
    else
      upper = middle;
  }
  return lower < control_count &&
                 controls[lower].stable_id == surface.stable_id &&
                 controls[lower].angular_level == surface.angular_level
             ? lower
             : -1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cubic_weights(const amrex::Real fraction, amrex::Real *const weights) noexcept {
  weights[0] = -fraction * (fraction - 1) * (fraction - 2) / 6;
  weights[1] = (fraction + 1) * (fraction - 1) * (fraction - 2) / 2;
  weights[2] = -(fraction + 1) * fraction * (fraction - 2) / 2;
  weights[3] = (fraction + 1) * fraction * (fraction - 1) / 6;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
surface_line_element(const LogicalSurfaceDeviceView &view,
                     const amrex::Long point, const amrex::Real theta,
                     const amrex::Real phi, const bool theta_direction)
    noexcept {
  const AngularGeometry basis = angular_geometry(theta, phi);
  const amrex::Real *const direction =
      theta_direction ? basis.radial_theta : basis.radial_phi;
  const amrex::Real h = view.height[point];
  const amrex::Real dh = view.angular_component(theta_direction ? 0 : 1,
                                                point);
  amrex::Real tangent[3];
  for (int d = 0; d < 3; ++d)
    tangent[d] = dh * basis.radial[d] + h * direction[d];
  const amrex::Real norm_squared =
      view.gamma(0, point) * tangent[0] * tangent[0] +
      2 * view.gamma(1, point) * tangent[0] * tangent[1] +
      2 * view.gamma(2, point) * tangent[0] * tangent[2] +
      view.gamma(3, point) * tangent[1] * tangent[1] +
      2 * view.gamma(4, point) * tangent[1] * tangent[2] +
      view.gamma(5, point) * tangent[2] * tangent[2];
  return norm_squared > 0 && amrex::Math::isfinite(norm_squared)
             ? norm_squared * amrex::Math::rsqrt(norm_squared)
             : -1;
}

} // namespace

void LogicalSurfaceStorage::initialize(
    const std::vector<LogicalSurfaceSeed> &local_surfaces) {
  if (initialized_)
    amrex::Abort("ParticleAHFinderX logical surface storage was initialized "
                 "more than once");
  if (local_surfaces.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX logical surface count exceeds the "
                 "portable device-kernel index range");

  local_surface_count_ = static_cast<int>(local_surfaces.size());
  std::vector<LogicalSurfaceDeviceDescriptor> host_descriptors;
  host_descriptors.reserve(local_surfaces.size());
  for (const auto &surface : local_surfaces) {
    if (surface.owner_offset != local_point_count_ ||
        surface.particle_count <= 0 || surface.angular_level < 0 ||
        surface.angular_levels <= 0 ||
        surface.angular_level >= surface.angular_levels ||
        surface.ntheta <= 0 || surface.nphi <= 0 || surface.radius <= 0)
      amrex::Abort("ParticleAHFinderX invalid logical owner layout");
    if (local_point_count_ > std::numeric_limits<amrex::Long>::max() -
                                 surface.particle_count)
      amrex::Abort("ParticleAHFinderX logical owner layout overflowed");

    LogicalSurfaceDeviceDescriptor descriptor{};
    descriptor.stable_id = surface.stable_id;
    descriptor.first_particle_id = surface.first_particle_id;
    descriptor.offset = surface.owner_offset;
    descriptor.count = surface.particle_count;
    descriptor.generation = surface.generation;
    descriptor.angular_level = surface.angular_level;
    descriptor.angular_levels = surface.angular_levels;
    descriptor.ntheta = surface.ntheta;
    descriptor.nphi = surface.nphi;
    descriptor.lifecycle = surface.lifecycle;
    descriptor.role = surface.role;
    for (int d = 0; d < 3; ++d)
      descriptor.center[d] = surface.center[d];
    descriptor.radius = surface.radius;
    descriptor.mass_scale = surface.mass_scale;
    descriptor.theta_l2_tolerance = surface.theta_l2_tolerance;
    descriptor.theta_linf_tolerance = surface.theta_linf_tolerance;
    descriptor.due = 1;
    descriptor.relaxed_helper_tolerance = 0;
    descriptor.relaxation_state =
        static_cast<int>(SurfaceRelaxationState::idle);
    host_descriptors.push_back(descriptor);
    local_point_count_ += surface.particle_count;
  }

  descriptors_.resize(host_descriptors.size());
  if (!host_descriptors.empty())
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_descriptors.begin(),
                     host_descriptors.end(), descriptors_.begin());

  const auto count = static_cast<std::size_t>(local_point_count_);
  for (auto &component : position_)
    component.resize(count);
  height_.resize(count);
  relaxation_velocity_.resize(count);
  expansion_.resize(count);
  angular_derivatives_.resize(5 * count);
  adm_.resize(30 * count);
  normal_.resize(3 * count);
  area_weight_.resize(count);
  particle_id_.resize(count);
  surface_id_.resize(count);
  generation_.resize(count);
  angular_level_.resize(count);
  angular_levels_.resize(count);
  itheta_.resize(count);
  iphi_.resize(count);
  lifecycle_.resize(count);
  role_.resize(count);
  surface_ordinal_.resize(count);
  arrival_.resize(count);
  geometry_status_.resize(count);
  sampled_level_.resize(count);
  sampled_level_changed_.resize(count);
  relaxation_status_.resize(count);
  relaxation_base_.resize(2 * count);
  relaxation_rhs_.resize(8 * count);
  committed_state_.resize(2 * count);
  extrapolation_reference_.resize(count);
  surface_extrapolation_displacement_.resize(4 * local_surface_count_);
  surface_extrapolation_weight_.resize(4 * local_surface_count_);
  surface_area_.resize(local_surface_count_);
  surface_theta_squared_.resize(local_surface_count_);
  surface_theta_integral_.resize(local_surface_count_);
  surface_theta_linf_.resize(local_surface_count_);
  surface_cutoff_indicator_.resize(local_surface_count_);
  surface_invalid_.resize(local_surface_count_);
  surface_minimum_sampled_level_.resize(local_surface_count_);
  surface_maximum_sampled_level_.resize(local_surface_count_);
  surface_sampled_level_changes_.resize(local_surface_count_);
  surface_position_integral_.resize(3 * local_surface_count_);
  surface_position_second_integral_.resize(6 * local_surface_count_);
  surface_position_minimum_.resize(3 * local_surface_count_);
  surface_position_maximum_.resize(3 * local_surface_count_);
  surface_height_integral_.resize(local_surface_count_);
  surface_height_minimum_.resize(local_surface_count_);
  surface_height_maximum_.resize(local_surface_count_);
  surface_circumference_.resize(3 * local_surface_count_);
  surface_analysis_integral_.resize(surface_analysis_real_components *
                                    local_surface_count_);
  surface_analysis_invalid_.resize(local_surface_count_);
  surface_intrinsic_invalid_.resize(local_surface_count_);
  surface_multipole_invalid_.resize(local_surface_count_);
  surface_coordinate_flux_invalid_.resize(local_surface_count_);
  surface_weyl_invalid_.resize(local_surface_count_);
  surface_weinberg_invalid_.resize(local_surface_count_);
  surface_ricci_invalid_.resize(local_surface_count_);
  surface_speciality_index_invalid_.resize(local_surface_count_);
  surface_speciality_difference_invalid_.resize(local_surface_count_);
  surface_axial_values_.resize(axial_surface_real_components *
                               local_surface_count_);
  surface_axial_status_.resize(local_surface_count_);
  surface_axial_iterations_.resize(local_surface_count_);
  surface_time_step_.resize(local_surface_count_);
  surface_pseudo_time_.resize(local_surface_count_);
  source_rank_.resize(count);
  source_index_.resize(count);

  if (local_point_count_ > 0) {
    const auto *const descriptors = descriptors_.dataPtr();
    const int num_surfaces = local_surface_count_;
    auto *const x = position_[0].dataPtr();
    auto *const y = position_[1].dataPtr();
    auto *const z = position_[2].dataPtr();
    auto *const height = height_.dataPtr();
    auto *const velocity = relaxation_velocity_.dataPtr();
    auto *const expansion = expansion_.dataPtr();
    auto *const angular = angular_derivatives_.dataPtr();
    auto *const adm = adm_.dataPtr();
    auto *const normal = normal_.dataPtr();
    auto *const area_weight = area_weight_.dataPtr();
    auto *const particle_id = particle_id_.dataPtr();
    auto *const surface_id = surface_id_.dataPtr();
    auto *const generation = generation_.dataPtr();
    auto *const angular_level = angular_level_.dataPtr();
    auto *const angular_levels = angular_levels_.dataPtr();
    auto *const itheta_values = itheta_.dataPtr();
    auto *const iphi_values = iphi_.dataPtr();
    auto *const lifecycle = lifecycle_.dataPtr();
    auto *const role = role_.dataPtr();
    auto *const surface_ordinal = surface_ordinal_.dataPtr();
    auto *const arrival = arrival_.dataPtr();
    auto *const geometry_status = geometry_status_.dataPtr();
    auto *const sampled_level = sampled_level_.dataPtr();
    auto *const sampled_level_changed = sampled_level_changed_.dataPtr();
    auto *const relaxation_status = relaxation_status_.dataPtr();
    auto *const relaxation_base = relaxation_base_.dataPtr();
    auto *const relaxation_rhs = relaxation_rhs_.dataPtr();
    auto *const committed_state = committed_state_.dataPtr();
    auto *const extrapolation_reference =
        extrapolation_reference_.dataPtr();
    auto *const source_rank = source_rank_.dataPtr();
    auto *const source_index = source_index_.dataPtr();
    const amrex::Long total_points = local_point_count_;

    amrex::ParallelFor(
        local_point_count_,
        [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
          int lower = 0;
          int upper = num_surfaces;
          while (lower + 1 < upper) {
            const int middle = lower + (upper - lower) / 2;
            if (descriptors[middle].offset <= point)
              lower = middle;
            else
              upper = middle;
          }
          const auto &surface = descriptors[lower];
          surface_ordinal[point] = lower;
          const amrex::Long logical_index = point - surface.offset;
          const int itheta = static_cast<int>(logical_index / surface.nphi);
          const int iphi = static_cast<int>(logical_index -
                                             amrex::Long(itheta) *
                                                 surface.nphi);
          const amrex::Real theta =
              amrex::Math::pi<amrex::Real>() *
              (amrex::Real(itheta) + amrex::Real(0.5)) / surface.ntheta;
          const amrex::Real phi =
              amrex::Real(2) * amrex::Math::pi<amrex::Real>() *
              (amrex::Real(iphi) + amrex::Real(0.5)) / surface.nphi;
          const AngularGeometry basis = angular_geometry(theta, phi);

          x[point] = surface.center[0] +
                     surface.radius * basis.radial[0];
          y[point] = surface.center[1] +
                     surface.radius * basis.radial[1];
          z[point] = surface.center[2] +
                     surface.radius * basis.radial[2];
          height[point] = surface.radius;
          velocity[point] = 0;
          expansion[point] = 0;
          particle_id[point] = surface.first_particle_id + logical_index;
          surface_id[point] = surface.stable_id;
          generation[point] = surface.generation;
          angular_level[point] = surface.angular_level;
          angular_levels[point] = surface.angular_levels;
          itheta_values[point] = itheta;
          iphi_values[point] = iphi;
          lifecycle[point] = surface.lifecycle;
          role[point] = surface.role;
          arrival[point] = 0;
          geometry_status[point] = 1;
          sampled_level[point] = -1;
          sampled_level_changed[point] = 0;
          relaxation_status[point] = 0;
          source_rank[point] = -1;
          source_index[point] = -1;
          area_weight[point] = 0;
          for (int component = 0; component < 5; ++component)
            angular[amrex::Long(component) * total_points + point] = 0;
          for (int component = 0; component < 30; ++component)
            adm[amrex::Long(component) * total_points + point] = 0;
          for (int d = 0; d < 3; ++d)
            normal[amrex::Long(d) * total_points + point] = 0;
          for (int component = 0; component < 2; ++component)
            relaxation_base[amrex::Long(component) * total_points + point] =
                0;
          committed_state[point] = surface.radius;
          committed_state[total_points + point] = 0;
          extrapolation_reference[point] = surface.radius;
          for (int component = 0; component < 8; ++component)
            relaxation_rhs[amrex::Long(component) * total_points + point] =
                0;
        });
    amrex::Gpu::streamSynchronize();
  }

  initialized_ = true;
  ++creation_events_;
}

void LogicalSurfaceStorage::append_surfaces(
    const std::vector<LogicalSurfaceSeed> &new_local_surfaces) {
  if (!initialized_)
    amrex::Abort("ParticleAHFinderX cannot append a logical surface before "
                 "initialization");
  if (new_local_surfaces.empty())
    return;
  if (new_local_surfaces.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max() -
                               local_surface_count_))
    amrex::Abort("ParticleAHFinderX appended logical surface count exceeds "
                 "the portable device-kernel index range");

  const int old_surface_count = local_surface_count_;
  const amrex::Long old_point_count = local_point_count_;
  std::vector<LogicalSurfaceDeviceDescriptor> host_descriptors;
  host_descriptors.reserve(new_local_surfaces.size());
  for (const auto &surface : new_local_surfaces) {
    if (surface.owner_offset != local_point_count_ ||
        surface.particle_count <= 0 || surface.angular_level < 0 ||
        surface.angular_levels <= 0 ||
        surface.angular_level >= surface.angular_levels ||
        surface.ntheta <= 0 || surface.nphi <= 0 || surface.radius <= 0)
      amrex::Abort("ParticleAHFinderX invalid appended logical owner layout");
    if (local_point_count_ > std::numeric_limits<amrex::Long>::max() -
                                 surface.particle_count)
      amrex::Abort("ParticleAHFinderX appended logical owner layout "
                   "overflowed");

    LogicalSurfaceDeviceDescriptor descriptor{};
    descriptor.stable_id = surface.stable_id;
    descriptor.first_particle_id = surface.first_particle_id;
    descriptor.offset = surface.owner_offset;
    descriptor.count = surface.particle_count;
    descriptor.generation = surface.generation;
    descriptor.angular_level = surface.angular_level;
    descriptor.angular_levels = surface.angular_levels;
    descriptor.ntheta = surface.ntheta;
    descriptor.nphi = surface.nphi;
    descriptor.lifecycle = surface.lifecycle;
    descriptor.role = surface.role;
    for (int d = 0; d < 3; ++d)
      descriptor.center[d] = surface.center[d];
    descriptor.radius = surface.radius;
    descriptor.mass_scale = surface.mass_scale;
    descriptor.theta_l2_tolerance = surface.theta_l2_tolerance;
    descriptor.theta_linf_tolerance = surface.theta_linf_tolerance;
    descriptor.due = 1;
    descriptor.relaxed_helper_tolerance = 0;
    descriptor.relaxation_state =
        static_cast<int>(SurfaceRelaxationState::idle);
    host_descriptors.push_back(descriptor);
    local_point_count_ += surface.particle_count;
  }
  local_surface_count_ += static_cast<int>(new_local_surfaces.size());

  descriptors_.resize(local_surface_count_);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_descriptors.begin(),
                   host_descriptors.end(),
                   descriptors_.begin() + old_surface_count);

  const auto count = static_cast<std::size_t>(local_point_count_);
  for (auto &component : position_)
    component.resize(count);
  height_.resize(count);
  relaxation_velocity_.resize(count);
  expansion_.resize(count);
  grow_component_major(angular_derivatives_, 5, old_point_count,
                       local_point_count_);
  grow_component_major(adm_, 30, old_point_count, local_point_count_);
  grow_component_major(normal_, 3, old_point_count, local_point_count_);
  area_weight_.resize(count);
  particle_id_.resize(count);
  surface_id_.resize(count);
  generation_.resize(count);
  angular_level_.resize(count);
  angular_levels_.resize(count);
  itheta_.resize(count);
  iphi_.resize(count);
  lifecycle_.resize(count);
  role_.resize(count);
  surface_ordinal_.resize(count);
  arrival_.resize(count);
  geometry_status_.resize(count);
  sampled_level_.resize(count);
  sampled_level_changed_.resize(count);
  relaxation_status_.resize(count);
  grow_component_major(relaxation_base_, 2, old_point_count,
                       local_point_count_);
  grow_component_major(relaxation_rhs_, 8, old_point_count,
                       local_point_count_);
  grow_component_major(committed_state_, 2, old_point_count,
                       local_point_count_);
  extrapolation_reference_.resize(count);
  surface_extrapolation_displacement_.resize(4 * local_surface_count_);
  surface_extrapolation_weight_.resize(4 * local_surface_count_);
  surface_area_.resize(local_surface_count_);
  surface_theta_squared_.resize(local_surface_count_);
  surface_theta_integral_.resize(local_surface_count_);
  surface_theta_linf_.resize(local_surface_count_);
  surface_cutoff_indicator_.resize(local_surface_count_);
  surface_invalid_.resize(local_surface_count_);
  surface_minimum_sampled_level_.resize(local_surface_count_);
  surface_maximum_sampled_level_.resize(local_surface_count_);
  surface_sampled_level_changes_.resize(local_surface_count_);
  surface_position_integral_.resize(3 * local_surface_count_);
  surface_position_second_integral_.resize(6 * local_surface_count_);
  surface_position_minimum_.resize(3 * local_surface_count_);
  surface_position_maximum_.resize(3 * local_surface_count_);
  surface_height_integral_.resize(local_surface_count_);
  surface_height_minimum_.resize(local_surface_count_);
  surface_height_maximum_.resize(local_surface_count_);
  surface_circumference_.resize(3 * local_surface_count_);
  surface_analysis_integral_.resize(surface_analysis_real_components *
                                    local_surface_count_);
  surface_analysis_invalid_.resize(local_surface_count_);
  surface_intrinsic_invalid_.resize(local_surface_count_);
  surface_multipole_invalid_.resize(local_surface_count_);
  surface_coordinate_flux_invalid_.resize(local_surface_count_);
  surface_weyl_invalid_.resize(local_surface_count_);
  surface_weinberg_invalid_.resize(local_surface_count_);
  surface_ricci_invalid_.resize(local_surface_count_);
  surface_speciality_index_invalid_.resize(local_surface_count_);
  surface_speciality_difference_invalid_.resize(local_surface_count_);
  surface_axial_values_.resize(axial_surface_real_components *
                               local_surface_count_);
  surface_axial_status_.resize(local_surface_count_);
  surface_axial_iterations_.resize(local_surface_count_);
  surface_time_step_.resize(local_surface_count_);
  surface_pseudo_time_.resize(local_surface_count_);
  source_rank_.resize(count);
  source_index_.resize(count);

  auto *const descriptors = descriptors_.dataPtr();
  const int num_surfaces = local_surface_count_;
  auto *const x = position_[0].dataPtr();
  auto *const y = position_[1].dataPtr();
  auto *const z = position_[2].dataPtr();
  auto *const height = height_.dataPtr();
  auto *const velocity = relaxation_velocity_.dataPtr();
  auto *const expansion = expansion_.dataPtr();
  auto *const angular = angular_derivatives_.dataPtr();
  auto *const adm = adm_.dataPtr();
  auto *const normal = normal_.dataPtr();
  auto *const area_weight = area_weight_.dataPtr();
  auto *const particle_id = particle_id_.dataPtr();
  auto *const surface_id = surface_id_.dataPtr();
  auto *const generation = generation_.dataPtr();
  auto *const angular_level = angular_level_.dataPtr();
  auto *const angular_levels = angular_levels_.dataPtr();
  auto *const itheta_values = itheta_.dataPtr();
  auto *const iphi_values = iphi_.dataPtr();
  auto *const lifecycle = lifecycle_.dataPtr();
  auto *const role = role_.dataPtr();
  auto *const surface_ordinal = surface_ordinal_.dataPtr();
  auto *const arrival = arrival_.dataPtr();
  auto *const geometry_status = geometry_status_.dataPtr();
  auto *const sampled_level = sampled_level_.dataPtr();
  auto *const sampled_level_changed = sampled_level_changed_.dataPtr();
  auto *const relaxation_status = relaxation_status_.dataPtr();
  auto *const relaxation_base = relaxation_base_.dataPtr();
  auto *const relaxation_rhs = relaxation_rhs_.dataPtr();
  auto *const committed_state = committed_state_.dataPtr();
  auto *const extrapolation_reference =
      extrapolation_reference_.dataPtr();
  auto *const source_rank = source_rank_.dataPtr();
  auto *const source_index = source_index_.dataPtr();
  const amrex::Long total_points = local_point_count_;
  const amrex::Long new_point_count = total_points - old_point_count;

  amrex::ParallelFor(
      new_point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long new_index) noexcept {
        const amrex::Long point = old_point_count + new_index;
        int lower = old_surface_count;
        int upper = num_surfaces;
        while (lower + 1 < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (descriptors[middle].offset <= point)
            lower = middle;
          else
            upper = middle;
        }
        const auto &surface = descriptors[lower];
        surface_ordinal[point] = lower;
        const amrex::Long logical_index = point - surface.offset;
        const int itheta = static_cast<int>(logical_index / surface.nphi);
        const int iphi = static_cast<int>(
            logical_index - amrex::Long(itheta) * surface.nphi);
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (amrex::Real(itheta) + amrex::Real(0.5)) / surface.ntheta;
        const amrex::Real phi =
            2 * amrex::Math::pi<amrex::Real>() *
            (amrex::Real(iphi) + amrex::Real(0.5)) / surface.nphi;
        const AngularGeometry basis = angular_geometry(theta, phi);

        x[point] = surface.center[0] +
                   surface.radius * basis.radial[0];
        y[point] = surface.center[1] +
                   surface.radius * basis.radial[1];
        z[point] = surface.center[2] + surface.radius * basis.radial[2];
        height[point] = surface.radius;
        velocity[point] = 0;
        expansion[point] = 0;
        particle_id[point] = surface.first_particle_id + logical_index;
        surface_id[point] = surface.stable_id;
        generation[point] = surface.generation;
        angular_level[point] = surface.angular_level;
        angular_levels[point] = surface.angular_levels;
        itheta_values[point] = itheta;
        iphi_values[point] = iphi;
        lifecycle[point] = surface.lifecycle;
        role[point] = surface.role;
        arrival[point] = 0;
        geometry_status[point] = 1;
        sampled_level[point] = -1;
        sampled_level_changed[point] = 0;
        relaxation_status[point] = 0;
        source_rank[point] = -1;
        source_index[point] = -1;
        area_weight[point] = 0;
        for (int component = 0; component < 5; ++component)
          angular[amrex::Long(component) * total_points + point] = 0;
        for (int component = 0; component < 30; ++component)
          adm[amrex::Long(component) * total_points + point] = 0;
        for (int d = 0; d < 3; ++d)
          normal[amrex::Long(d) * total_points + point] = 0;
        for (int component = 0; component < 2; ++component)
          relaxation_base[amrex::Long(component) * total_points + point] = 0;
        committed_state[point] = surface.radius;
        committed_state[total_points + point] = 0;
        extrapolation_reference[point] = surface.radius;
        for (int component = 0; component < 8; ++component)
          relaxation_rhs[amrex::Long(component) * total_points + point] = 0;
      });
  amrex::Gpu::streamSynchronize();
  ++creation_events_;
}

void LogicalSurfaceStorage::update_surface_centers(
    const std::vector<SurfaceCenterUpdate> &updates) {
  if (updates.empty() || local_surface_count_ == 0)
    return;
  if (updates.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX center-update table exceeds the portable "
                 "device-kernel index range");

  amrex::Gpu::DeviceVector<SurfaceCenterUpdate> device_updates(updates.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, updates.begin(), updates.end(),
                   device_updates.begin());
  const auto *const center_updates = device_updates.dataPtr();
  const int update_count = static_cast<int>(updates.size());
  const auto view = device_view();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const amrex::Long surface_id = view.surface_id[point];
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
          view.position[d][point] += center_updates[lower].new_center[d] -
                                     center_updates[lower].old_center[d];
      });

  auto *const descriptors = descriptors_.dataPtr();
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        const amrex::Long surface_id = descriptors[surface].stable_id;
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
          descriptors[surface].center[d] =
              center_updates[lower].new_center[d];
      });
  amrex::Gpu::streamSynchronize();
}

void LogicalSurfaceStorage::update_surface_states(
    const std::vector<SurfaceStateUpdate> &updates) {
  if (updates.empty() || local_surface_count_ == 0)
    return;
  amrex::Gpu::DeviceVector<SurfaceStateUpdate> device_updates(updates.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, updates.begin(), updates.end(),
                   device_updates.begin());
  const auto *const state_updates = device_updates.dataPtr();
  const int update_count = static_cast<int>(updates.size());
  auto *const descriptors = descriptors_.dataPtr();
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        const amrex::Long surface_id = descriptors[surface].stable_id;
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
        descriptors[surface].lifecycle =
            static_cast<int>(state_updates[lower].lifecycle);
        descriptors[surface].role =
            static_cast<int>(state_updates[lower].role);
      });
  const auto view = device_view();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const amrex::Long surface_id = view.surface_id[point];
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
        view.lifecycle[point] =
            static_cast<int>(state_updates[lower].lifecycle);
        view.role[point] = static_cast<int>(state_updates[lower].role);
      });
  amrex::Gpu::streamSynchronize();
}

void LogicalSurfaceStorage::reset_surface_spheres(
    const std::vector<SurfaceSphereReset> &resets) {
  if (resets.empty() || local_surface_count_ == 0)
    return;
  if (resets.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX logical sphere-reset table exceeds the "
                 "portable device-kernel index range");
  amrex::Gpu::DeviceVector<SurfaceSphereReset> device_resets(resets.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, resets.begin(), resets.end(),
                   device_resets.begin());
  const auto *const sphere_resets = device_resets.dataPtr();
  const int reset_count = static_cast<int>(resets.size());
  auto *const descriptors = descriptors_.dataPtr();
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        const amrex::Long surface_id = descriptors[surface].stable_id;
        int lower = 0;
        int upper = reset_count;
        while (lower < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (sphere_resets[middle].surface_id < surface_id)
            lower = middle + 1;
          else
            upper = middle;
        }
        if (lower < reset_count &&
            sphere_resets[lower].surface_id == surface_id) {
          descriptors[surface].radius = sphere_resets[lower].radius;
          for (int d = 0; d < 3; ++d)
            descriptors[surface].center[d] = sphere_resets[lower].center[d];
        }
      });

  const auto view = device_view();
  const amrex::Long point_count = local_point_count_;
  auto *const committed = committed_state_.dataPtr();
  auto *const geometry_status = geometry_status_.dataPtr();
  auto *const arrival = arrival_.dataPtr();
  auto *const source_rank = source_rank_.dataPtr();
  auto *const source_index = source_index_.dataPtr();
  amrex::ParallelFor(
      point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const amrex::Long surface_id = view.surface_id[point];
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
        const auto &surface =
            view.descriptors[view.surface_ordinal[point]];
        const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                                  (view.itheta[point] + amrex::Real(0.5)) /
                                  surface.ntheta;
        const amrex::Real phi = 2 * amrex::Math::pi<amrex::Real>() *
                                (view.iphi[point] + amrex::Real(0.5)) /
                                surface.nphi;
        const AngularGeometry basis = angular_geometry(theta, phi);
        const auto &reset = sphere_resets[lower];
        view.position[0][point] =
            reset.center[0] + reset.radius * basis.radial[0];
        view.position[1][point] =
            reset.center[1] + reset.radius * basis.radial[1];
        view.position[2][point] = reset.center[2] +
                                  reset.radius * basis.radial[2];
        view.height[point] = reset.radius;
        view.relaxation_velocity[point] = 0;
        view.expansion[point] = 0;
        committed[point] = reset.radius;
        committed[point_count + point] = 0;
        geometry_status[point] = 1;
        view.sampled_level[point] = -1;
        view.sampled_level_changed[point] = 0;
        arrival[point] = 0;
        source_rank[point] = -1;
        source_index[point] = -1;
      });
  amrex::Gpu::streamSynchronize();
}

void LogicalSurfaceStorage::compact(
    const std::vector<LogicalSurfaceSeed> &live_local_surfaces) {
  if (!initialized_)
    amrex::Abort("ParticleAHFinderX cannot compact uninitialized logical "
                 "surface storage");

  LogicalSurfaceStorage replacement;
  replacement.initialize(live_local_surfaces);
  const auto old_view = device_view();
  const auto new_view = replacement.device_view();
  amrex::Gpu::DeviceVector<int> device_error(1, 0);
  int *const error = device_error.dataPtr();
  const auto *const old_descriptors = descriptors_.dataPtr();
  const int old_surface_count = local_surface_count_;
  const amrex::Long old_point_count = local_point_count_;
  const amrex::Long new_point_count = replacement.local_point_count_;
  const auto *const old_committed = committed_state_.dataPtr();
  auto *const new_committed = replacement.committed_state_.dataPtr();
  amrex::ParallelFor(
      new_point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long new_point) noexcept {
        const auto &new_surface =
            new_view.descriptors[new_view.surface_ordinal[new_point]];
        int lower = 0;
        int upper = old_surface_count;
        while (lower < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (old_descriptors[middle].stable_id < new_surface.stable_id ||
              (old_descriptors[middle].stable_id == new_surface.stable_id &&
               old_descriptors[middle].angular_level <
                   new_surface.angular_level))
            lower = middle + 1;
          else
            upper = middle;
        }
        if (lower >= old_surface_count ||
            old_descriptors[lower].stable_id != new_surface.stable_id ||
            old_descriptors[lower].generation != new_surface.generation ||
            old_descriptors[lower].angular_level !=
                new_surface.angular_level ||
            old_descriptors[lower].angular_levels !=
                new_surface.angular_levels ||
            old_descriptors[lower].count != new_surface.count ||
            old_descriptors[lower].ntheta != new_surface.ntheta ||
            old_descriptors[lower].nphi != new_surface.nphi) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }
        const amrex::Long logical_index = new_point - new_surface.offset;
        const amrex::Long old_point =
            old_descriptors[lower].offset + logical_index;
        for (int d = 0; d < 3; ++d)
          new_view.position[d][new_point] = old_view.position[d][old_point];
        new_view.height[new_point] = old_view.height[old_point];
        new_view.relaxation_velocity[new_point] =
            old_view.relaxation_velocity[old_point];
        new_view.expansion[new_point] = old_view.expansion[old_point];
        for (int component = 0; component < 5; ++component)
          new_view.angular_component(component, new_point) =
              old_view.angular_component(component, old_point);
        for (int component = 0; component < 6; ++component) {
          new_view.gamma(component, new_point) =
              old_view.gamma(component, old_point);
          new_view.curv(component, new_point) =
              old_view.curv(component, old_point);
          for (int d = 0; d < 3; ++d)
            new_view.d_gamma(d, component, new_point) =
                old_view.d_gamma(d, component, old_point);
        }
        for (int d = 0; d < 3; ++d)
          new_view.normal_component(d, new_point) =
              old_view.normal_component(d, old_point);
        new_view.area_weight[new_point] = old_view.area_weight[old_point];
        new_view.particle_id[new_point] = old_view.particle_id[old_point];
        new_view.surface_id[new_point] = old_view.surface_id[old_point];
        new_view.generation[new_point] = old_view.generation[old_point];
        new_view.angular_level[new_point] =
            old_view.angular_level[old_point];
        new_view.angular_levels[new_point] =
            old_view.angular_levels[old_point];
        new_view.itheta[new_point] = old_view.itheta[old_point];
        new_view.iphi[new_point] = old_view.iphi[old_point];
        new_view.lifecycle[new_point] = new_surface.lifecycle;
        new_view.role[new_point] = new_surface.role;
        new_view.geometry_status[new_point] =
            old_view.geometry_status[old_point];
        new_view.sampled_level[new_point] =
            old_view.sampled_level[old_point];
        new_view.sampled_level_changed[new_point] =
            old_view.sampled_level_changed[old_point];
        new_committed[new_point] = old_committed[old_point];
        new_committed[new_point_count + new_point] =
            old_committed[old_point_count + old_point];
      });
  int host_error = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_error.begin(),
                   device_error.end(), &host_error);
  if (host_error)
    amrex::Abort("ParticleAHFinderX logical compaction could not match a "
                 "live surface to its previous device state");

  replacement.creation_events_ = creation_events_;
  descriptors_.swap(replacement.descriptors_);
  position_.swap(replacement.position_);
  height_.swap(replacement.height_);
  relaxation_velocity_.swap(replacement.relaxation_velocity_);
  expansion_.swap(replacement.expansion_);
  angular_derivatives_.swap(replacement.angular_derivatives_);
  adm_.swap(replacement.adm_);
  normal_.swap(replacement.normal_);
  area_weight_.swap(replacement.area_weight_);
  particle_id_.swap(replacement.particle_id_);
  surface_id_.swap(replacement.surface_id_);
  generation_.swap(replacement.generation_);
  angular_level_.swap(replacement.angular_level_);
  angular_levels_.swap(replacement.angular_levels_);
  itheta_.swap(replacement.itheta_);
  iphi_.swap(replacement.iphi_);
  lifecycle_.swap(replacement.lifecycle_);
  role_.swap(replacement.role_);
  surface_ordinal_.swap(replacement.surface_ordinal_);
  arrival_.swap(replacement.arrival_);
  geometry_status_.swap(replacement.geometry_status_);
  sampled_level_.swap(replacement.sampled_level_);
  sampled_level_changed_.swap(replacement.sampled_level_changed_);
  relaxation_status_.swap(replacement.relaxation_status_);
  relaxation_base_.swap(replacement.relaxation_base_);
  relaxation_rhs_.swap(replacement.relaxation_rhs_);
  committed_state_.swap(replacement.committed_state_);
  extrapolation_reference_.swap(replacement.extrapolation_reference_);
  surface_extrapolation_displacement_.swap(
      replacement.surface_extrapolation_displacement_);
  surface_extrapolation_weight_.swap(
      replacement.surface_extrapolation_weight_);
  surface_area_.swap(replacement.surface_area_);
  surface_theta_squared_.swap(replacement.surface_theta_squared_);
  surface_theta_integral_.swap(replacement.surface_theta_integral_);
  surface_theta_linf_.swap(replacement.surface_theta_linf_);
  surface_cutoff_indicator_.swap(replacement.surface_cutoff_indicator_);
  surface_invalid_.swap(replacement.surface_invalid_);
  surface_minimum_sampled_level_.swap(
      replacement.surface_minimum_sampled_level_);
  surface_maximum_sampled_level_.swap(
      replacement.surface_maximum_sampled_level_);
  surface_sampled_level_changes_.swap(
      replacement.surface_sampled_level_changes_);
  surface_position_integral_.swap(replacement.surface_position_integral_);
  surface_position_second_integral_.swap(
      replacement.surface_position_second_integral_);
  surface_position_minimum_.swap(replacement.surface_position_minimum_);
  surface_position_maximum_.swap(replacement.surface_position_maximum_);
  surface_height_integral_.swap(replacement.surface_height_integral_);
  surface_height_minimum_.swap(replacement.surface_height_minimum_);
  surface_height_maximum_.swap(replacement.surface_height_maximum_);
  surface_circumference_.swap(replacement.surface_circumference_);
  surface_analysis_integral_.swap(
      replacement.surface_analysis_integral_);
  surface_analysis_invalid_.swap(
      replacement.surface_analysis_invalid_);
  surface_intrinsic_invalid_.swap(
      replacement.surface_intrinsic_invalid_);
  surface_multipole_invalid_.swap(
      replacement.surface_multipole_invalid_);
  surface_coordinate_flux_invalid_.swap(
      replacement.surface_coordinate_flux_invalid_);
  surface_weyl_invalid_.swap(replacement.surface_weyl_invalid_);
  surface_weinberg_invalid_.swap(
      replacement.surface_weinberg_invalid_);
  surface_ricci_invalid_.swap(replacement.surface_ricci_invalid_);
  surface_speciality_index_invalid_.swap(
      replacement.surface_speciality_index_invalid_);
  surface_speciality_difference_invalid_.swap(
      replacement.surface_speciality_difference_invalid_);
  intrinsic_metric_.swap(replacement.intrinsic_metric_);
  intrinsic_normal_connection_.swap(
      replacement.intrinsic_normal_connection_);
  intrinsic_connection_difference_.swap(
      replacement.intrinsic_connection_difference_);
  intrinsic_scalar_curvature_.swap(
      replacement.intrinsic_scalar_curvature_);
  intrinsic_point_status_.swap(replacement.intrinsic_point_status_);
  intrinsic_connection_status_.swap(
      replacement.intrinsic_connection_status_);
  axial_basis_.swap(replacement.axial_basis_);
  axial_gradient_.swap(replacement.axial_gradient_);
  axial_covector_.swap(replacement.axial_covector_);
  axial_deformation_.swap(replacement.axial_deformation_);
  axial_stiffness_.swap(replacement.axial_stiffness_);
  axial_mass_.swap(replacement.axial_mass_);
  axial_factor_.swap(replacement.axial_factor_);
  axial_work_.swap(replacement.axial_work_);
  axial_eigenvectors_.swap(replacement.axial_eigenvectors_);
  axial_coefficients_.swap(replacement.axial_coefficients_);
  axial_potential_.swap(replacement.axial_potential_);
  axial_vector_.swap(replacement.axial_vector_);
  surface_axial_values_.swap(replacement.surface_axial_values_);
  surface_axial_status_.swap(replacement.surface_axial_status_);
  surface_axial_iterations_.swap(replacement.surface_axial_iterations_);
  surface_time_step_.swap(replacement.surface_time_step_);
  surface_pseudo_time_.swap(replacement.surface_pseudo_time_);
  source_rank_.swap(replacement.source_rank_);
  source_index_.swap(replacement.source_index_);
  local_point_count_ = new_point_count;
  local_surface_count_ = replacement.local_surface_count_;
  creation_events_ = replacement.creation_events_;
  initialized_ = true;
}

LogicalSurfaceDeviceView LogicalSurfaceStorage::device_view() {
  LogicalSurfaceDeviceView view;
  view.descriptors = descriptors_.dataPtr();
  view.surface_ordinal = surface_ordinal_.dataPtr();
  for (int d = 0; d < 3; ++d)
    view.position[d] = position_[d].dataPtr();
  view.height = height_.dataPtr();
  view.relaxation_velocity = relaxation_velocity_.dataPtr();
  view.expansion = expansion_.dataPtr();
  view.angular = angular_derivatives_.dataPtr();
  view.adm = adm_.dataPtr();
  view.normal = normal_.dataPtr();
  view.area_weight = area_weight_.dataPtr();
  view.particle_id = particle_id_.dataPtr();
  view.surface_id = surface_id_.dataPtr();
  view.generation = generation_.dataPtr();
  view.angular_level = angular_level_.dataPtr();
  view.angular_levels = angular_levels_.dataPtr();
  view.itheta = itheta_.dataPtr();
  view.iphi = iphi_.dataPtr();
  view.lifecycle = lifecycle_.dataPtr();
  view.role = role_.dataPtr();
  view.arrival = arrival_.dataPtr();
  view.geometry_status = geometry_status_.dataPtr();
  view.sampled_level = sampled_level_.dataPtr();
  view.sampled_level_changed = sampled_level_changed_.dataPtr();
  view.source_rank = source_rank_.dataPtr();
  view.source_index = source_index_.dataPtr();
  view.point_count = local_point_count_;
  view.surface_count = local_surface_count_;
  return view;
}

LogicalAnalysisPointDeviceView
LogicalSurfaceStorage::analysis_point_device_view(
    const bool invariant_spin_requested,
    const bool weyl_curvature_requested) {
  const auto point_count = static_cast<std::size_t>(local_point_count_);
  const auto surface_count = static_cast<std::size_t>(local_surface_count_);
  if (intrinsic_metric_.size() != intrinsic_metric_components * point_count ||
      intrinsic_normal_connection_.size() != 2 * point_count ||
      intrinsic_scalar_curvature_.size() != point_count ||
      intrinsic_point_status_.size() != point_count)
    amrex::Abort("ParticleAHFinderX point output has no current intrinsic "
                 "geometry workspace");

  const bool axial_workspace_available =
      axial_potential_.size() == point_count &&
      axial_vector_.size() == 2 * point_count &&
      surface_axial_values_.size() ==
          axial_surface_real_components * surface_count &&
      surface_axial_status_.size() == surface_count;
  if (invariant_spin_requested && !axial_workspace_available)
    amrex::Abort("ParticleAHFinderX point output has no current axial "
                 "workspace");

  const bool weyl_workspace_available =
      weyl_electric_.size() == 6 * point_count &&
      weyl_magnetic_.size() == 6 * point_count &&
      weyl_arrival_.size() == point_count;
  if (weyl_curvature_requested && !weyl_workspace_available)
    amrex::Abort("ParticleAHFinderX point output has no current Weyl "
                 "workspace");

  LogicalAnalysisPointDeviceView view;
  view.surface = device_view();
  view.intrinsic_metric = intrinsic_metric_.dataPtr();
  view.intrinsic_normal_connection =
      intrinsic_normal_connection_.dataPtr();
  view.intrinsic_scalar_curvature =
      intrinsic_scalar_curvature_.dataPtr();
  view.intrinsic_status = intrinsic_point_status_.dataPtr();
  if (axial_workspace_available) {
    view.axial_potential = axial_potential_.dataPtr();
    view.axial_vector = axial_vector_.dataPtr();
    view.surface_axial_values = surface_axial_values_.dataPtr();
    view.surface_axial_status = surface_axial_status_.dataPtr();
  }
  if (weyl_workspace_available) {
    view.weyl_electric = weyl_electric_.dataPtr();
    view.weyl_magnetic = weyl_magnetic_.dataPtr();
    view.weyl_arrival = weyl_arrival_.dataPtr();
  }
  view.axial_workspace_available = axial_workspace_available;
  view.weyl_workspace_available = weyl_workspace_available;
  return view;
}

LogicalWeylDeviceView LogicalSurfaceStorage::begin_weyl_routing() {
  const auto count = static_cast<std::size_t>(local_point_count_);
  weyl_electric_.resize(6 * count);
  weyl_magnetic_.resize(6 * count);
  weyl_arrival_.resize(count);
  auto *const arrival = weyl_arrival_.dataPtr();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        arrival[point] = 0;
      });
  return {weyl_electric_.dataPtr(), weyl_magnetic_.dataPtr(), arrival,
          local_point_count_};
}

LogicalArrivalStats LogicalSurfaceStorage::local_weyl_arrival_stats() const {
  amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
  amrex::ReduceData<amrex::Long, amrex::Long> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  const int *const arrival = weyl_arrival_.dataPtr();
  const int *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  reduce_ops.eval(
      local_point_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept -> ReduceTuple {
        const auto &descriptor = descriptors[surface_ordinal[point]];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels)
          return {0, 0};
        return {arrival[point] == 0, arrival[point] > 1};
      });
  const auto result = reduce_data.value();
  return {amrex::get<0>(result), amrex::get<1>(result)};
}

LogicalGaugeDeviceView LogicalSurfaceStorage::begin_gauge_routing() {
  const auto count = static_cast<std::size_t>(local_point_count_);
  gauge_value_.resize(FourGaugeLayout::component_count * count);
  gauge_arrival_.resize(count);
  auto *const arrival = gauge_arrival_.dataPtr();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        arrival[point] = 0;
      });
  return {gauge_value_.dataPtr(), arrival, local_point_count_};
}

LogicalArrivalStats LogicalSurfaceStorage::local_gauge_arrival_stats() const {
  amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
  amrex::ReduceData<amrex::Long, amrex::Long> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  const int *const arrival = gauge_arrival_.dataPtr();
  const int *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  reduce_ops.eval(
      local_point_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept -> ReduceTuple {
        const auto &descriptor = descriptors[surface_ordinal[point]];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels)
          return {0, 0};
        return {arrival[point] == 0, arrival[point] > 1};
      });
  const auto result = reduce_data.value();
  return {amrex::get<0>(result), amrex::get<1>(result)};
}

LogicalMatterDeviceView LogicalSurfaceStorage::begin_matter_routing() {
  const auto count = static_cast<std::size_t>(local_point_count_);
  matter_value_.resize(FourMatterLayout::component_count * count);
  matter_arrival_.resize(count);
  auto *const arrival = matter_arrival_.dataPtr();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        arrival[point] = 0;
      });
  return {matter_value_.dataPtr(), arrival, local_point_count_};
}

LogicalArrivalStats LogicalSurfaceStorage::local_matter_arrival_stats() const {
  amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
  amrex::ReduceData<amrex::Long, amrex::Long> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  const int *const arrival = matter_arrival_.dataPtr();
  const int *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  reduce_ops.eval(
      local_point_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept -> ReduceTuple {
        const auto &descriptor = descriptors[surface_ordinal[point]];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels)
          return {0, 0};
        return {arrival[point] == 0, arrival[point] > 1};
      });
  const auto result = reduce_data.value();
  return {amrex::get<0>(result), amrex::get<1>(result)};
}

void LogicalSurfaceStorage::set_due_surface_ids(
    const std::vector<amrex::Long> &stable_ids) {
  amrex::Gpu::DeviceVector<amrex::Long> device_ids(stable_ids.size());
  if (!stable_ids.empty())
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, stable_ids.begin(),
                     stable_ids.end(), device_ids.begin());
  auto *const descriptors = descriptors_.dataPtr();
  const auto *const ids = device_ids.dataPtr();
  const int count = static_cast<int>(stable_ids.size());
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        descriptors[surface].due = 0;
        descriptors[surface].relaxation_state =
            static_cast<int>(SurfaceRelaxationState::idle);
        for (int i = 0; i < count; ++i)
          if (descriptors[surface].stable_id == ids[i])
            descriptors[surface].due = 1;
      });
  amrex::Gpu::streamSynchronize();
}

void LogicalSurfaceStorage::set_due_surface_layers(
    const std::vector<SurfaceLayerSelection> &selections) {
  if (selections.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX due-layer table exceeds the portable "
                 "device-kernel index range");
  amrex::Gpu::DeviceVector<SurfaceLayerSelection> device_selections(
      selections.size());
  if (!selections.empty())
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, selections.begin(),
                     selections.end(), device_selections.begin());
  auto *const descriptors = descriptors_.dataPtr();
  const auto *const selected = device_selections.dataPtr();
  const int count = static_cast<int>(selections.size());
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        descriptors[surface].due = 0;
        descriptors[surface].relaxation_state =
            static_cast<int>(SurfaceRelaxationState::idle);
        for (int i = 0; i < count; ++i)
          if (descriptors[surface].stable_id == selected[i].stable_id &&
              descriptors[surface].angular_level ==
                  selected[i].angular_level)
            descriptors[surface].due = 1;
      });
  amrex::Gpu::streamSynchronize();
}

void LogicalSurfaceStorage::set_relaxed_helper_surface_ids(
    const std::vector<amrex::Long> &stable_ids) {
  if (stable_ids.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX relaxed-helper table exceeds the portable "
                 "device-kernel index range");
  for (std::size_t index = 0; index < stable_ids.size(); ++index)
    if (stable_ids[index] <= 0 ||
        (index > 0 && stable_ids[index - 1] >= stable_ids[index]))
      amrex::Abort("ParticleAHFinderX relaxed-helper IDs must be positive, "
                   "unique, and ordered");

  amrex::Gpu::DeviceVector<amrex::Long> device_ids(stable_ids.size());
  if (!stable_ids.empty())
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, stable_ids.begin(),
                     stable_ids.end(), device_ids.begin());
  auto *const descriptors = descriptors_.dataPtr();
  const auto *const ids = device_ids.dataPtr();
  const int count = static_cast<int>(stable_ids.size());
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        const amrex::Long stable_id = descriptors[surface].stable_id;
        int lower = 0;
        int upper = count;
        while (lower < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (ids[middle] < stable_id)
            lower = middle + 1;
          else
            upper = middle;
        }
        descriptors[surface].relaxed_helper_tolerance =
            lower < count && ids[lower] == stable_id;
      });
  amrex::Gpu::streamSynchronize();
}

void LogicalSurfaceStorage::restrict_target_to_coarse_layers(
    const std::vector<amrex::Long> &stable_ids) {
  if (stable_ids.empty())
    return;
  if (stable_ids.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX restriction table exceeds the portable "
                 "device-kernel index range");
  for (std::size_t index = 0; index < stable_ids.size(); ++index)
    if (stable_ids[index] <= 0 ||
        (index > 0 && stable_ids[index - 1] >= stable_ids[index]))
      amrex::Abort("ParticleAHFinderX restriction IDs must be positive, "
                   "unique, and ordered");

  amrex::Gpu::DeviceVector<amrex::Long> device_ids(stable_ids.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, stable_ids.begin(),
                   stable_ids.end(), device_ids.begin());
  const auto *const selected_ids = device_ids.dataPtr();
  const int selected_count = static_cast<int>(stable_ids.size());
  const auto view = device_view();
  const int surface_count = local_surface_count_;
  amrex::Gpu::DeviceVector<int> device_error(1, 0);
  int *const error = device_error.dataPtr();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int destination_ordinal = view.surface_ordinal[point];
        const auto &destination = view.descriptors[destination_ordinal];
        if (destination.angular_level + 1 >= destination.angular_levels)
          return;
        int lower = 0;
        int upper = selected_count;
        while (lower < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (selected_ids[middle] < destination.stable_id)
            lower = middle + 1;
          else
            upper = middle;
        }
        if (lower >= selected_count ||
            selected_ids[lower] != destination.stable_id)
          return;

        const int target_ordinal =
            destination_ordinal + destination.angular_levels - 1 -
            destination.angular_level;
        if (target_ordinal < 0 || target_ordinal >= surface_count) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }
        const auto &target = view.descriptors[target_ordinal];
        if (target.stable_id != destination.stable_id ||
            target.angular_level + 1 != target.angular_levels ||
            target.angular_levels != destination.angular_levels ||
            target.ntheta % destination.ntheta != 0 ||
            target.nphi % destination.nphi != 0) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }

        const int theta_ratio = target.ntheta / destination.ntheta;
        const int phi_ratio = target.nphi / destination.nphi;
        const int theta_begin = view.itheta[point] * theta_ratio;
        const int phi_begin = view.iphi[point] * phi_ratio;
        amrex::Real weighted_height = 0;
        amrex::Real weight_sum = 0;
        for (int di = 0; di < theta_ratio; ++di) {
          const int itheta = theta_begin + di;
          const amrex::Real theta =
              amrex::Math::pi<amrex::Real>() *
              (itheta + amrex::Real(0.5)) / target.ntheta;
          const amrex::Real theta_weight =
              amrex::Math::sincos(theta).first;
          for (int dj = 0; dj < phi_ratio; ++dj) {
            const int iphi = phi_begin + dj;
            weighted_height +=
                theta_weight *
                view.height[target.offset + amrex::Long(itheta) *
                                                target.nphi +
                            iphi];
            weight_sum += theta_weight;
          }
        }
        if (!(weight_sum > 0) || !amrex::Math::isfinite(weighted_height)) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }
        const amrex::Real restricted_height = weighted_height / weight_sum;
        if (!(restricted_height > 0) ||
            !amrex::Math::isfinite(restricted_height)) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }
        view.height[point] = restricted_height;
        view.relaxation_velocity[point] = 0;
        view.expansion[point] = 0;
        view.area_weight[point] = 0;
        for (int d = 0; d < 3; ++d)
          view.normal_component(d, point) = 0;
        view.geometry_status[point] = 1;
      });
  int host_error = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_error.begin(),
                   device_error.end(), &host_error);
  if (host_error)
    amrex::Abort("ParticleAHFinderX could not restrict a target angular "
                 "surface into its coarse hierarchy");
}

void LogicalSurfaceStorage::prolong_surface_layers(
    const std::vector<SurfaceLayerProlongation> &source_layers) {
  if (source_layers.empty())
    return;
  if (source_layers.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX prolongation table exceeds the portable "
                 "device-kernel index range");
  for (std::size_t index = 0; index < source_layers.size(); ++index)
    if (source_layers[index].stable_id <= 0 ||
        source_layers[index].angular_level < 0 ||
        (index > 0 &&
         (source_layers[index - 1].stable_id >
              source_layers[index].stable_id ||
          (source_layers[index - 1].stable_id ==
               source_layers[index].stable_id &&
           source_layers[index - 1].angular_level >=
               source_layers[index].angular_level))))
      amrex::Abort("ParticleAHFinderX prolongation sources must be unique and "
                   "ordered");
  amrex::Gpu::DeviceVector<SurfaceLayerProlongation> device_sources(
      source_layers.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, source_layers.begin(),
                   source_layers.end(), device_sources.begin());
  const auto *const sources = device_sources.dataPtr();
  const int source_count = static_cast<int>(source_layers.size());
  const auto view = device_view();
  const auto *const committed = committed_state_.dataPtr();
  amrex::Gpu::DeviceVector<int> device_error(1, 0);
  int *const error = device_error.dataPtr();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int destination_ordinal = view.surface_ordinal[point];
        const auto &destination = view.descriptors[destination_ordinal];
        int selected = -1;
        for (int i = 0; i < source_count; ++i)
          if (destination.stable_id == sources[i].stable_id &&
              destination.angular_level == sources[i].angular_level + 1)
            selected = i;
        if (selected < 0)
          return;
        if (destination_ordinal <= 0) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }
        const auto &source = view.descriptors[destination_ordinal - 1];
        if (source.stable_id != destination.stable_id ||
            source.angular_level + 1 != destination.angular_level ||
            source.angular_levels != destination.angular_levels ||
            (source.relaxation_state !=
                 static_cast<int>(SurfaceRelaxationState::converged) &&
             source.relaxation_state !=
                 static_cast<int>(SurfaceRelaxationState::refine))) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }

        const int destination_itheta = view.itheta[point];
        const int destination_iphi = view.iphi[point];
        const amrex::Real theta_coordinate =
            (destination_itheta + amrex::Real(0.5)) * source.ntheta /
                destination.ntheta -
            amrex::Real(0.5);
        const amrex::Real phi_coordinate =
            (destination_iphi + amrex::Real(0.5)) * source.nphi /
                destination.nphi -
            amrex::Real(0.5);
        const int theta_cell =
            static_cast<int>(amrex::Math::floor(theta_coordinate));
        const int phi_cell =
            static_cast<int>(amrex::Math::floor(phi_coordinate));
        amrex::GpuArray<amrex::Real, 4> theta_weight;
        amrex::GpuArray<amrex::Real, 4> theta_derivative;
        amrex::GpuArray<amrex::Real, 4> phi_weight;
        amrex::GpuArray<amrex::Real, 4> phi_derivative;
        cubic_lagrange_weights(theta_coordinate - theta_cell, theta_weight,
                               theta_derivative);
        cubic_lagrange_weights(phi_coordinate - phi_cell, phi_weight,
                               phi_derivative);
        const auto mode = sources[selected].mode;
        if (mode != SurfaceProlongationMode::replace &&
            mode != SurfaceProlongationMode::additive_correction) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }
        amrex::Real prolonged_value = 0;
        for (int j = 0; j < 4; ++j)
          for (int i = 0; i < 4; ++i) {
            const int source_itheta = theta_cell - 1 + i;
            const int source_iphi = phi_cell - 1 + j;
            amrex::Real value =
                angular_scalar(view.height, source.offset, source.ntheta,
                               source.nphi, source_itheta, source_iphi);
            if (mode == SurfaceProlongationMode::additive_correction)
              value -= angular_scalar(committed, source.offset,
                                      source.ntheta, source.nphi,
                                      source_itheta, source_iphi);
            prolonged_value += theta_weight[i] * phi_weight[j] * value;
          }
        const amrex::Real prolonged_height =
            mode == SurfaceProlongationMode::replace
                ? prolonged_value
                : view.height[point] + prolonged_value;
        if (!(prolonged_height > 0) ||
            !amrex::Math::isfinite(prolonged_height)) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }
        view.height[point] = prolonged_height;
        view.relaxation_velocity[point] = 0;
        view.expansion[point] = 0;
        view.area_weight[point] = 0;
        for (int d = 0; d < 3; ++d)
          view.normal_component(d, point) = 0;
        view.geometry_status[point] = 1;
      });
  int host_error = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_error.begin(),
                   device_error.end(), &host_error);
  if (host_error)
    amrex::Abort("ParticleAHFinderX could not prolong an angular surface "
                 "layer");
}

void LogicalSurfaceStorage::begin_sample_routing() {
  const auto view = device_view();
  amrex::ParallelFor(local_point_count_,
                     [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
                       view.arrival[point] = 0;
                     });
}

LogicalArrivalStats LogicalSurfaceStorage::local_arrival_stats() const {
  amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
  amrex::ReduceData<amrex::Long, amrex::Long> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  const int *const arrival = arrival_.dataPtr();
  const int *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  reduce_ops.eval(
      local_point_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept -> ReduceTuple {
        if (!descriptors[surface_ordinal[point]].due)
          return {0, 0};
        return {arrival[point] == 0, arrival[point] > 1};
      });
  const auto result = reduce_data.value();
  return {amrex::get<0>(result), amrex::get<1>(result)};
}

void LogicalSurfaceStorage::evaluate_surface_geometry() {
  const auto view = device_view();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &surface =
            view.descriptors[view.surface_ordinal[point]];
        if (!surface.due)
          return;
        const auto derivatives = angular_derivatives(
            view.height, surface.offset, surface.ntheta, surface.nphi,
            view.itheta[point], view.iphi[point]);
        view.angular_component(0, point) = derivatives.theta;
        view.angular_component(1, point) = derivatives.phi;
        view.angular_component(2, point) = derivatives.theta_theta;
        view.angular_component(3, point) = derivatives.theta_phi;
        view.angular_component(4, point) = derivatives.phi_phi;
        ExpansionInput input;
        for (int component = 0; component < 6; ++component) {
          input.gamma[component] = view.gamma(component, point);
          input.curv[component] = view.curv(component, point);
          for (int d = 0; d < 3; ++d)
            input.d_gamma[d][component] =
                view.d_gamma(d, component, point);
        }
        input.height = view.height[point];
        input.h_theta = derivatives.theta;
        input.h_phi = derivatives.phi;
        input.h_theta_theta = derivatives.theta_theta;
        input.h_theta_phi = derivatives.theta_phi;
        input.h_phi_phi = derivatives.phi_phi;
        input.theta = amrex::Math::pi<amrex::Real>() *
                      (view.itheta[point] + amrex::Real(0.5)) /
                      surface.ntheta;
        input.phi = 2 * amrex::Math::pi<amrex::Real>() *
                    (view.iphi[point] + amrex::Real(0.5)) / surface.nphi;
        input.dtheta =
            amrex::Math::pi<amrex::Real>() / surface.ntheta;
        input.dphi =
            2 * amrex::Math::pi<amrex::Real>() / surface.nphi;
        ExpansionOutput output;
        if (!ParticleAHFinderX::evaluate_expansion(input, output)) {
          view.geometry_status[point] = 1;
          view.expansion[point] = 0;
          view.area_weight[point] = 0;
          for (int d = 0; d < 3; ++d)
            view.normal_component(d, point) = 0;
          return;
        }
        view.geometry_status[point] = 0;
        view.expansion[point] = output.theta;
        view.area_weight[point] = output.area_weight;
        for (int d = 0; d < 3; ++d)
          view.normal_component(d, point) = output.normal[d];
      });
}

void LogicalSurfaceStorage::evaluate_solver_residuals() {
  if (local_surface_count_ == 0)
    return;
  auto *const surface_area = surface_area_.dataPtr();
  auto *const surface_theta_squared = surface_theta_squared_.dataPtr();
  auto *const surface_theta_linf = surface_theta_linf_.dataPtr();
  auto *const surface_cutoff_indicator =
      surface_cutoff_indicator_.dataPtr();
  auto *const surface_invalid = surface_invalid_.dataPtr();
  const int surface_count = local_surface_count_;
  amrex::ParallelFor(surface_count,
                     [=] AMREX_GPU_DEVICE(const int surface) noexcept {
                       surface_area[surface] = 0;
                       surface_theta_squared[surface] = 0;
                       surface_theta_linf[surface] = 0;
                       surface_cutoff_indicator[surface] = 0;
                       surface_invalid[surface] = 0;
                     });

  const auto view = device_view();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int surface = view.surface_ordinal[point];
        const auto &descriptor = view.descriptors[surface];
        if (!descriptor.due)
          return;
        if (view.geometry_status[point] != 0) {
          amrex::Gpu::Atomic::AddNoRet(surface_invalid + surface,
                                       amrex::Long(1));
          return;
        }
        const amrex::Real theta = view.expansion[point];
        const amrex::Real absolute_theta = theta < 0 ? -theta : theta;
        amrex::Gpu::Atomic::AddNoRet(surface_area + surface,
                                     view.area_weight[point]);
        amrex::Gpu::Atomic::AddNoRet(
            surface_theta_squared + surface,
            theta * theta * view.area_weight[point]);
        amrex::Gpu::Atomic::Max(surface_theta_linf + surface,
                                absolute_theta);
        if (descriptor.angular_level + 1 < descriptor.angular_levels) {
          const auto filter = angular_dissipation(
              view.height, descriptor.offset, descriptor.ntheta,
              descriptor.nphi, view.itheta[point], view.iphi[point]);
          const amrex::Real dtheta =
              amrex::Math::pi<amrex::Real>() / descriptor.ntheta;
          const amrex::Real dphi =
              2 * amrex::Math::pi<amrex::Real>() / descriptor.nphi;
          const amrex::Real height = view.height[point] < 0
                                         ? -view.height[point]
                                         : view.height[point];
          const amrex::Real denominator = amrex::max(
              height, std::numeric_limits<amrex::Real>::min());
          const amrex::Real theta_indicator =
              dtheta * (filter.theta < 0 ? -filter.theta : filter.theta) /
              denominator;
          const amrex::Real phi_indicator =
              dphi * (filter.phi < 0 ? -filter.phi : filter.phi) /
              denominator;
          amrex::Gpu::Atomic::Max(
              surface_cutoff_indicator + surface,
              amrex::max(theta_indicator, phi_indicator));
        }
      });
}

void LogicalSurfaceStorage::update_solver_convergence_on_device(
    const bool accept_convergence,
    const amrex::Real cutoff_indicator_threshold,
    const amrex::Real helper_tolerance_factor,
    const bool fail_invalid_geometry) {
  if (local_surface_count_ == 0)
    return;
  evaluate_solver_residuals();
  const auto *const surface_area = surface_area_.dataPtr();
  const auto *const surface_theta_squared = surface_theta_squared_.dataPtr();
  const auto *const surface_theta_linf = surface_theta_linf_.dataPtr();
  const auto *const surface_cutoff_indicator =
      surface_cutoff_indicator_.dataPtr();
  const auto *const surface_invalid = surface_invalid_.dataPtr();
  auto *const descriptors = descriptors_.dataPtr();
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        auto &descriptor = descriptors[surface];
        const auto values = surface_convergence_values(
            descriptor, surface_area[surface],
            surface_theta_squared[surface], surface_theta_linf[surface],
            surface_invalid[surface]);
        update_surface_relaxation_state(
            descriptor, values, surface_cutoff_indicator[surface],
            accept_convergence, cutoff_indicator_threshold,
            helper_tolerance_factor, fail_invalid_geometry);
      });
}

SolverConvergenceStats LogicalSurfaceStorage::update_solver_convergence(
    const bool accept_convergence,
    const amrex::Real cutoff_indicator_threshold,
    const amrex::Real helper_tolerance_factor,
    const bool fail_invalid_geometry) {
  if (local_surface_count_ == 0)
    return {};
  evaluate_solver_residuals();
  auto *const surface_area = surface_area_.dataPtr();
  auto *const surface_theta_squared = surface_theta_squared_.dataPtr();
  auto *const surface_theta_linf = surface_theta_linf_.dataPtr();
  auto *const surface_cutoff_indicator =
      surface_cutoff_indicator_.dataPtr();
  auto *const surface_invalid = surface_invalid_.dataPtr();
  const int surface_count = local_surface_count_;

  // Each descriptor belongs to exactly one logical owner. Update its state
  // and reduce residual/state telemetry in one surface kernel so every
  // pseudo-step needs only one synchronization-producing device reduction.
  amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                   amrex::ReduceOpMax, amrex::ReduceOpMax,
                   amrex::ReduceOpSum, amrex::ReduceOpSum,
                   amrex::ReduceOpSum, amrex::ReduceOpSum,
                   amrex::ReduceOpSum>
      reduce_ops;
  amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                    amrex::Long, amrex::Long, amrex::Long, amrex::Long,
                    amrex::Long>
      reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  auto *const descriptors = descriptors_.dataPtr();
  reduce_ops.eval(
      surface_count, reduce_data,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept -> ReduceTuple {
        auto &descriptor = descriptors[surface];
        const auto values = surface_convergence_values(
            descriptor, surface_area[surface],
            surface_theta_squared[surface], surface_theta_linf[surface],
            surface_invalid[surface]);
        update_surface_relaxation_state(
            descriptor, values, surface_cutoff_indicator[surface],
            accept_convergence, cutoff_indicator_threshold,
            helper_tolerance_factor, fail_invalid_geometry);

        if (!descriptor.due)
          return {0, 0, 0, 0, 0, 0, 0, 0, 0};
        const int state = descriptor.relaxation_state;
        return {
            values.geometry_valid ? values.ratio : amrex::Real(0),
            values.geometry_valid ? values.l2_times_mass : amrex::Real(0),
            values.geometry_valid ? values.linf_times_mass : amrex::Real(0),
            values.geometry_valid ? surface_cutoff_indicator[surface]
                                  : amrex::Real(0),
            values.geometry_valid ? surface_invalid[surface]
                                  : surface_invalid[surface] + 1,
            state == static_cast<int>(SurfaceRelaxationState::active),
            state == static_cast<int>(SurfaceRelaxationState::converged),
            state == static_cast<int>(SurfaceRelaxationState::refine),
            state == static_cast<int>(SurfaceRelaxationState::failed),
        };
      });
  const auto result = reduce_data.value();
  return {amrex::get<0>(result), amrex::get<1>(result),
          amrex::get<2>(result), amrex::get<3>(result),
          amrex::get<4>(result), amrex::get<5>(result),
          amrex::get<6>(result), amrex::get<7>(result),
          amrex::get<8>(result)};
}

void LogicalSurfaceStorage::reactivate_due_layers_for_direct_validation() {
  auto *const descriptors = descriptors_.dataPtr();
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        auto &descriptor = descriptors[surface];
        if (!descriptor.due)
          return;
        if (descriptor.relaxation_state ==
                static_cast<int>(SurfaceRelaxationState::converged) ||
            descriptor.relaxation_state ==
                static_cast<int>(SurfaceRelaxationState::refine))
          descriptor.relaxation_state =
              static_cast<int>(SurfaceRelaxationState::active);
      });
}

std::vector<LocalSolverSurfaceState>
LogicalSurfaceStorage::local_solver_surface_states() {
  if (local_surface_count_ == 0)
    return {};
  solver_surface_state_.resize(local_surface_count_);
  auto *const records = solver_surface_state_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  const auto *const surface_area = surface_area_.dataPtr();
  const auto *const surface_theta_squared = surface_theta_squared_.dataPtr();
  const auto *const surface_theta_linf = surface_theta_linf_.dataPtr();
  const auto *const surface_invalid = surface_invalid_.dataPtr();
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        const auto &descriptor = descriptors[surface];
        auto &record = records[surface];
        record.stable_id = descriptor.stable_id;
        record.angular_level = descriptor.angular_level;
        record.relaxation_state = static_cast<SurfaceRelaxationState>(
            descriptor.relaxation_state);
        record.due = descriptor.due != 0;
        record.expansion_l2 = 0;
        record.expansion_linf = surface_theta_linf[surface];
        record.invalid_points = surface_invalid[surface];
        if (surface_area[surface] > 0) {
          const amrex::Real mean_squared =
              surface_theta_squared[surface] / surface_area[surface];
          record.expansion_l2 =
              mean_squared > 0
                  ? mean_squared * amrex::Math::rsqrt(mean_squared)
                  : 0;
        }
      });
  std::vector<LocalSolverSurfaceState> records_host(local_surface_count_);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, solver_surface_state_.begin(),
                   solver_surface_state_.end(), records_host.begin());
  return records_host;
}

std::vector<LocalSurfaceDiagnostics>
LogicalSurfaceStorage::local_surface_diagnostics() {
  if (local_surface_count_ == 0)
    return {};
  auto *const surface_area = surface_area_.dataPtr();
  auto *const surface_theta_squared = surface_theta_squared_.dataPtr();
  auto *const surface_theta_integral = surface_theta_integral_.dataPtr();
  auto *const surface_theta_linf = surface_theta_linf_.dataPtr();
  auto *const surface_invalid = surface_invalid_.dataPtr();
  auto *const surface_minimum_sampled_level =
      surface_minimum_sampled_level_.dataPtr();
  auto *const surface_maximum_sampled_level =
      surface_maximum_sampled_level_.dataPtr();
  auto *const surface_sampled_level_changes =
      surface_sampled_level_changes_.dataPtr();
  auto *const surface_position_integral =
      surface_position_integral_.dataPtr();
  auto *const surface_position_second_integral =
      surface_position_second_integral_.dataPtr();
  auto *const surface_position_minimum =
      surface_position_minimum_.dataPtr();
  auto *const surface_position_maximum =
      surface_position_maximum_.dataPtr();
  auto *const surface_height_integral = surface_height_integral_.dataPtr();
  auto *const surface_height_minimum = surface_height_minimum_.dataPtr();
  auto *const surface_height_maximum = surface_height_maximum_.dataPtr();
  auto *const surface_circumference = surface_circumference_.dataPtr();
  const int surface_count = local_surface_count_;
  amrex::ParallelFor(
      surface_count, [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        surface_area[surface] = 0;
        surface_theta_squared[surface] = 0;
        surface_theta_integral[surface] = 0;
        surface_theta_linf[surface] = 0;
        surface_invalid[surface] = 0;
        surface_minimum_sampled_level[surface] =
            std::numeric_limits<int>::max();
        surface_maximum_sampled_level[surface] = -1;
        surface_sampled_level_changes[surface] = 0;
        surface_height_integral[surface] = 0;
        surface_height_minimum[surface] =
            std::numeric_limits<amrex::Real>::max();
        surface_height_maximum[surface] = 0;
        for (int d = 0; d < 3; ++d) {
          surface_position_integral[d * surface_count + surface] = 0;
          surface_position_minimum[d * surface_count + surface] =
              std::numeric_limits<amrex::Real>::max();
          surface_position_maximum[d * surface_count + surface] =
              std::numeric_limits<amrex::Real>::lowest();
        }
        for (int component = 0; component < 6; ++component)
          surface_position_second_integral[component * surface_count +
                                           surface] = 0;
        for (int plane = 0; plane < 3; ++plane)
          surface_circumference[plane * surface_count + surface] = 0;
      });

  const auto view = device_view();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int surface = view.surface_ordinal[point];
        if (!view.descriptors[surface].due)
          return;
        if (view.geometry_status[point] != 0) {
          amrex::Gpu::Atomic::AddNoRet(surface_invalid + surface,
                                       amrex::Long(1));
          return;
        }
        const amrex::Real weight = view.area_weight[point];
        const amrex::Real theta = view.expansion[point];
        const amrex::Real absolute_theta = theta < 0 ? -theta : theta;
        const amrex::Real height = view.height[point];
        amrex::Gpu::Atomic::Min(surface_minimum_sampled_level + surface,
                                view.sampled_level[point]);
        amrex::Gpu::Atomic::Max(surface_maximum_sampled_level + surface,
                                view.sampled_level[point]);
        amrex::Gpu::Atomic::AddNoRet(
            surface_sampled_level_changes + surface,
            amrex::Long(view.sampled_level_changed[point] != 0));
        amrex::Gpu::Atomic::AddNoRet(surface_area + surface, weight);
        amrex::Gpu::Atomic::AddNoRet(surface_theta_squared + surface,
                                     theta * theta * weight);
        amrex::Gpu::Atomic::AddNoRet(surface_theta_integral + surface,
                                     theta * weight);
        amrex::Gpu::Atomic::Max(surface_theta_linf + surface,
                                absolute_theta);
        amrex::Gpu::Atomic::AddNoRet(surface_height_integral + surface,
                                     height * weight);
        amrex::Gpu::Atomic::Min(surface_height_minimum + surface, height);
        amrex::Gpu::Atomic::Max(surface_height_maximum + surface, height);
        for (int d = 0; d < 3; ++d) {
          amrex::Gpu::Atomic::AddNoRet(
              surface_position_integral + d * surface_count + surface,
              view.position[d][point] * weight);
          amrex::Gpu::Atomic::Min(
              surface_position_minimum + d * surface_count + surface,
              view.position[d][point]);
          amrex::Gpu::Atomic::Max(
              surface_position_maximum + d * surface_count + surface,
              view.position[d][point]);
        }
        const amrex::Real x = view.position[0][point];
        const amrex::Real y = view.position[1][point];
        const amrex::Real z = view.position[2][point];
        const amrex::Real second[6]{x * x, x * y, x * z,
                                    y * y, y * z, z * z};
        for (int component = 0; component < 6; ++component)
          amrex::Gpu::Atomic::AddNoRet(
              surface_position_second_integral +
                  component * surface_count + surface,
              second[component] * weight);
      });

  // Sample the three coordinate-plane great circles from the native logical
  // grid. The line elements and the cubic angular interpolation both remain
  // on the owner GPU; only the final three scalars per surface are copied.
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int surface = view.surface_ordinal[point];
        const auto &descriptor = view.descriptors[surface];
        if (!descriptor.due || view.geometry_status[point] != 0)
          return;
        const int itheta = view.itheta[point];
        const int iphi = view.iphi[point];
        const amrex::Real pi = amrex::Math::pi<amrex::Real>();
        const amrex::Real dtheta = pi / descriptor.ntheta;
        const amrex::Real dphi = 2 * pi / descriptor.nphi;

        if (itheta == 0) {
          const amrex::Real coordinate =
              (pi / 2) / dtheta - amrex::Real(0.5);
          const int base =
              static_cast<int>(amrex::Math::floor(coordinate));
          amrex::Real weight[4];
          cubic_weights(coordinate - base, weight);
          amrex::Real integrand = 0;
          for (int a = 0; a < 4; ++a) {
            const int source_i = base + a - 1;
            const amrex::Long source_point =
                descriptor.offset +
                amrex::Long(source_i) * descriptor.nphi + iphi;
            const amrex::Real source_theta =
                (source_i + amrex::Real(0.5)) * dtheta;
            const amrex::Real source_phi =
                (iphi + amrex::Real(0.5)) * dphi;
            integrand +=
                weight[a] * surface_line_element(
                                view, source_point, source_theta, source_phi,
                                false);
          }
          if (!(integrand > 0) || !amrex::Math::isfinite(integrand))
            amrex::Gpu::Atomic::AddNoRet(surface_invalid + surface,
                                         amrex::Long(1));
          else
            amrex::Gpu::Atomic::AddNoRet(
                surface_circumference + surface, integrand * dphi);
        }

        if (iphi == 0) {
          const amrex::Real theta =
              (itheta + amrex::Real(0.5)) * dtheta;
          const amrex::Real target_phi[4]{0, pi, pi / 2, 3 * pi / 2};
          amrex::Real integrand[4]{0, 0, 0, 0};
          for (int curve = 0; curve < 4; ++curve) {
            const amrex::Real coordinate =
                target_phi[curve] / dphi - amrex::Real(0.5);
            const int base =
                static_cast<int>(amrex::Math::floor(coordinate));
            amrex::Real weight[4];
            cubic_weights(coordinate - base, weight);
            for (int b = 0; b < 4; ++b) {
              const int source_j = positive_modulo(base + b - 1,
                                                   descriptor.nphi);
              const amrex::Long source_point =
                  descriptor.offset +
                  amrex::Long(itheta) * descriptor.nphi + source_j;
              const amrex::Real source_phi =
                  (source_j + amrex::Real(0.5)) * dphi;
              integrand[curve] +=
                  weight[b] * surface_line_element(
                                  view, source_point, theta, source_phi,
                                  true);
            }
          }
          const amrex::Real xz = integrand[0] + integrand[1];
          const amrex::Real yz = integrand[2] + integrand[3];
          if (!(xz > 0) || !(yz > 0) || !amrex::Math::isfinite(xz) ||
              !amrex::Math::isfinite(yz))
            amrex::Gpu::Atomic::AddNoRet(surface_invalid + surface,
                                         amrex::Long(1));
          else {
            amrex::Gpu::Atomic::AddNoRet(
                surface_circumference + surface_count + surface,
                xz * dtheta);
            amrex::Gpu::Atomic::AddNoRet(
                surface_circumference + 2 * surface_count + surface,
                yz * dtheta);
          }
        }
      });

  std::vector<LogicalSurfaceDeviceDescriptor> descriptors(surface_count);
  std::vector<amrex::Real> areas(surface_count);
  std::vector<amrex::Real> theta_squared(surface_count);
  std::vector<amrex::Real> theta_integral(surface_count);
  std::vector<amrex::Real> theta_linf(surface_count);
  std::vector<amrex::Long> invalid(surface_count);
  std::vector<int> minimum_sampled_level(surface_count);
  std::vector<int> maximum_sampled_level(surface_count);
  std::vector<amrex::Long> sampled_level_changes(surface_count);
  std::vector<amrex::Real> position_integral(3 * surface_count);
  std::vector<amrex::Real> position_second_integral(6 * surface_count);
  std::vector<amrex::Real> position_minimum(3 * surface_count);
  std::vector<amrex::Real> position_maximum(3 * surface_count);
  std::vector<amrex::Real> height_integral(surface_count);
  std::vector<amrex::Real> height_minimum(surface_count);
  std::vector<amrex::Real> height_maximum(surface_count);
  std::vector<amrex::Real> circumference(3 * surface_count);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, descriptors_.begin(),
                   descriptors_.end(), descriptors.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, surface_area_.begin(),
                   surface_area_.end(), areas.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, surface_theta_squared_.begin(),
                   surface_theta_squared_.end(), theta_squared.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, surface_theta_integral_.begin(),
                   surface_theta_integral_.end(), theta_integral.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, surface_theta_linf_.begin(),
                   surface_theta_linf_.end(), theta_linf.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, surface_invalid_.begin(),
                   surface_invalid_.end(), invalid.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   surface_minimum_sampled_level_.begin(),
                   surface_minimum_sampled_level_.end(),
                   minimum_sampled_level.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   surface_maximum_sampled_level_.begin(),
                   surface_maximum_sampled_level_.end(),
                   maximum_sampled_level.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   surface_sampled_level_changes_.begin(),
                   surface_sampled_level_changes_.end(),
                   sampled_level_changes.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   surface_position_integral_.begin(),
                   surface_position_integral_.end(),
                   position_integral.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   surface_position_second_integral_.begin(),
                   surface_position_second_integral_.end(),
                   position_second_integral.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   surface_position_minimum_.begin(),
                   surface_position_minimum_.end(), position_minimum.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   surface_position_maximum_.begin(),
                   surface_position_maximum_.end(), position_maximum.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   surface_height_integral_.begin(),
                   surface_height_integral_.end(), height_integral.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   surface_height_minimum_.begin(),
                   surface_height_minimum_.end(), height_minimum.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   surface_height_maximum_.begin(),
                   surface_height_maximum_.end(), height_maximum.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   surface_circumference_.begin(),
                   surface_circumference_.end(), circumference.begin());

  std::vector<LocalSurfaceDiagnostics> diagnostics(surface_count);
  for (int surface = 0; surface < surface_count; ++surface) {
    auto &result = diagnostics[surface];
    result.stable_id = descriptors[surface].stable_id;
    result.angular_level = descriptors[surface].angular_level;
    result.angular_levels = descriptors[surface].angular_levels;
    result.relaxation_state = static_cast<SurfaceRelaxationState>(
        descriptors[surface].relaxation_state);
    result.due = descriptors[surface].due != 0;
    result.area = areas[surface];
    result.expansion_linf = theta_linf[surface];
    result.invalid_points = invalid[surface];
    result.minimum_sampled_level =
        minimum_sampled_level[surface] == std::numeric_limits<int>::max()
            ? -1
            : minimum_sampled_level[surface];
    result.maximum_sampled_level = maximum_sampled_level[surface];
    result.sampled_level_changes = sampled_level_changes[surface];
    if (!(result.area > 0) || result.invalid_points != 0)
      continue;
    const amrex::Real mean_theta_squared =
        theta_squared[surface] / result.area;
    result.expansion_l2 =
        mean_theta_squared > 0
            ? mean_theta_squared * amrex::Math::rsqrt(mean_theta_squared)
            : 0;
    result.mean_expansion = theta_integral[surface] / result.area;
    result.mean_radius = height_integral[surface] / result.area;
    result.minimum_radius = height_minimum[surface];
    result.maximum_radius = height_maximum[surface];
    for (int plane = 0; plane < 3; ++plane)
      result.proper_circumference[plane] =
          circumference[plane * surface_count + surface];
    for (int d = 0; d < 3; ++d) {
      result.centroid[d] =
          position_integral[d * surface_count + surface] / result.area;
      result.position_minimum[d] =
          position_minimum[d * surface_count + surface];
      result.position_maximum[d] =
          position_maximum[d * surface_count + surface];
    }
    constexpr int first_index[6]{0, 0, 0, 1, 1, 2};
    constexpr int second_index[6]{0, 1, 2, 1, 2, 2};
    for (int component = 0; component < 6; ++component)
      result.coordinate_quadrupole[component] =
          position_second_integral[component * surface_count + surface] /
              result.area -
          result.centroid[first_index[component]] *
              result.centroid[second_index[component]];
  }
  return diagnostics;
}

void LogicalSurfaceStorage::prepare_intrinsic_geometry() {
  if (local_point_count_ == 0)
    return;
  const auto point_count = static_cast<std::size_t>(local_point_count_);
  intrinsic_metric_.resize(intrinsic_metric_components * point_count);
  intrinsic_normal_connection_.resize(2 * point_count);
  intrinsic_connection_difference_.resize(
      intrinsic_connection_components * point_count);
  intrinsic_scalar_curvature_.resize(point_count);
  intrinsic_point_status_.resize(point_count);
  intrinsic_connection_status_.resize(point_count);

  const auto view = device_view();
  auto *const metric = intrinsic_metric_.dataPtr();
  auto *const normal_connection =
      intrinsic_normal_connection_.dataPtr();
  auto *const connection_difference =
      intrinsic_connection_difference_.dataPtr();
  auto *const scalar_curvature = intrinsic_scalar_curvature_.dataPtr();
  auto *const point_status = intrinsic_point_status_.dataPtr();
  auto *const connection_status =
      intrinsic_connection_status_.dataPtr();
  const amrex::Real epsilon =
      std::numeric_limits<amrex::Real>::epsilon();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        point_status[point] = 1;
        connection_status[point] = 1;
        scalar_curvature[point] = 0;
        for (int component = 0; component < intrinsic_metric_components;
             ++component)
          metric[amrex::Long(component) * view.point_count + point] = 0;
        for (int angular = 0; angular < 2; ++angular)
          normal_connection[amrex::Long(angular) * view.point_count +
                            point] = 0;

        const auto &surface =
            view.descriptors[view.surface_ordinal[point]];
        if (!surface.due ||
            surface.angular_level + 1 != surface.angular_levels ||
            view.geometry_status[point] != 0)
          return;
        IntrinsicGeometryInput input;
        for (int component = 0; component < 6; ++component) {
          input.gamma[component] = view.gamma(component, point);
          input.curv[component] = view.curv(component, point);
        }
        for (int d = 0; d < 3; ++d)
          input.normal[d] = view.normal_component(d, point);
        input.height = view.height[point];
        input.h_theta = view.angular_component(0, point);
        input.h_phi = view.angular_component(1, point);
        input.theta = amrex::Math::pi<amrex::Real>() *
                      (view.itheta[point] + amrex::Real(0.5)) /
                      surface.ntheta;
        input.phi = 2 * amrex::Math::pi<amrex::Real>() *
                    (view.iphi[point] + amrex::Real(0.5)) /
                    surface.nphi;
        IntrinsicGeometryOutput output;
        if (!evaluate_intrinsic_geometry(input, output))
          return;
        const amrex::Real intrinsic_weight = surface_cell_area_weight(
            output.sqrt_determinant,
            amrex::Math::pi<amrex::Real>() / surface.ntheta,
            2 * amrex::Math::pi<amrex::Real>() / surface.nphi);
        const amrex::Real native_weight = view.area_weight[point];
        const amrex::Real weight_scale =
            amrex::max(amrex::Real(1),
                       amrex::max(intrinsic_weight, native_weight));
        const amrex::Real weight_difference =
            intrinsic_weight > native_weight
                ? intrinsic_weight - native_weight
                : native_weight - intrinsic_weight;
        if (!(intrinsic_weight > 0) || !(native_weight > 0) ||
            !amrex::Math::isfinite(intrinsic_weight) ||
            weight_difference > amrex::Real(256) * epsilon * weight_scale)
          return;
        for (int component = 0; component < intrinsic_metric_components;
             ++component)
          metric[amrex::Long(component) * view.point_count + point] =
              output.metric[component];
        for (int angular = 0; angular < 2; ++angular)
          normal_connection[amrex::Long(angular) * view.point_count +
                            point] =
              output.normal_connection[angular];
        point_status[point] = 0;
      });

  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &surface =
            view.descriptors[view.surface_ordinal[point]];
        if (!surface.due ||
            surface.angular_level + 1 != surface.angular_levels ||
            point_status[point] != 0 ||
            !intrinsic_stencil_valid(
                point_status, surface.offset, surface.ntheta, surface.nphi,
                view.itheta[point], view.iphi[point]))
          return;
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (view.itheta[point] + amrex::Real(0.5)) / surface.ntheta;
        amrex::Real difference[2][2][2];
        if (!intrinsic_connection_difference(
                metric, view.point_count, surface.offset, surface.ntheta,
                surface.nphi, view.itheta[point], view.iphi[point], theta,
                difference))
          return;
        for (int upper = 0; upper < 2; ++upper)
          for (int first = 0; first < 2; ++first)
            for (int second = first; second < 2; ++second)
              connection_difference
                  [amrex::Long(intrinsic_connection_component(
                       upper, first, second)) *
                       view.point_count +
                   point] = difference[upper][first][second];
        connection_status[point] = 0;
      });

  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &surface =
            view.descriptors[view.surface_ordinal[point]];
        if (!surface.due ||
            surface.angular_level + 1 != surface.angular_levels)
          return;
        point_status[point] = 1;
        if (connection_status[point] != 0 ||
            !intrinsic_stencil_valid(
                connection_status, surface.offset, surface.ntheta,
                surface.nphi, view.itheta[point], view.iphi[point]))
          return;
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (view.itheta[point] + amrex::Real(0.5)) / surface.ntheta;
        amrex::Real value = 0;
        if (!intrinsic_scalar_curvature(
                metric, connection_difference, view.point_count,
                surface.offset, surface.ntheta, surface.nphi,
                view.itheta[point], view.iphi[point], theta, value))
          return;
        scalar_curvature[point] = value;
        point_status[point] = 0;
      });
}

void LogicalSurfaceStorage::prepare_axial_symmetry(
    const AxialSymmetryControl &control) {
  if (control.compute_multipoles && !control.enabled)
    amrex::Abort("ParticleAHFinderX multipoles require axial-symmetry "
                 "analysis");
  if (local_surface_count_ == 0)
    return;
  auto *const surface_status = surface_axial_status_.dataPtr();
  auto *const surface_iterations = surface_axial_iterations_.dataPtr();
  auto *const surface_values = surface_axial_values_.dataPtr();
  const int surface_count = local_surface_count_;
  const int initial_status = static_cast<int>(
      control.enabled ? AxialSymmetryStatus::valid
                      : AxialSymmetryStatus::not_requested);
  amrex::ParallelFor(
      surface_count, [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        surface_status[surface] = initial_status;
        surface_iterations[surface] = 0;
        for (int component = 0; component < axial_surface_real_components;
             ++component)
          surface_values[component * surface_count + surface] = 0;
      });
  if (!control.enabled)
    return;
  if (control.harmonic_degree < 1 ||
      control.harmonic_degree > maximum_axial_harmonic_degree ||
      control.maximum_jacobi_sweeps <= 0 ||
      !(control.eigen_tolerance > 0) ||
      !(control.minimum_axis_separation > 0))
    amrex::Abort("ParticleAHFinderX received invalid axial-symmetry control");

  const int basis_count = axial_basis_count(control.harmonic_degree);
  const auto checked_product = [](const std::size_t first,
                                  const std::size_t second) {
    if (second != 0 &&
        first > std::numeric_limits<std::size_t>::max() / second)
      amrex::Abort("ParticleAHFinderX axial-symmetry workspace overflowed");
    return first * second;
  };
  const std::size_t point_count =
      static_cast<std::size_t>(local_point_count_);
  const std::size_t mode_points =
      checked_product(static_cast<std::size_t>(basis_count), point_count);
  const std::size_t matrix_size = checked_product(
      checked_product(static_cast<std::size_t>(surface_count),
                      static_cast<std::size_t>(basis_count)),
      static_cast<std::size_t>(basis_count));
  if (mode_points >
          static_cast<std::size_t>(std::numeric_limits<amrex::Long>::max()) ||
      matrix_size >
          static_cast<std::size_t>(std::numeric_limits<amrex::Long>::max()) ||
      (basis_count > 0 &&
       local_point_count_ >
           std::numeric_limits<amrex::Long>::max() / basis_count /
               basis_count))
    amrex::Abort("ParticleAHFinderX axial-symmetry launch range overflowed");

  axial_basis_.resize(mode_points);
  axial_gradient_.resize(2 * mode_points);
  axial_covector_.resize(2 * mode_points);
  axial_deformation_.resize(3 * mode_points);
  axial_stiffness_.resize(matrix_size);
  axial_mass_.resize(matrix_size);
  axial_factor_.resize(matrix_size);
  axial_work_.resize(matrix_size);
  axial_eigenvectors_.resize(matrix_size);
  axial_coefficients_.resize(
      checked_product(static_cast<std::size_t>(surface_count),
                      static_cast<std::size_t>(basis_count)));
  axial_potential_.resize(point_count);
  axial_vector_.resize(2 * point_count);

  const auto view = device_view();
  auto *const basis = axial_basis_.dataPtr();
  auto *const gradient = axial_gradient_.dataPtr();
  auto *const covector = axial_covector_.dataPtr();
  auto *const deformation = axial_deformation_.dataPtr();
  auto *const stiffness = axial_stiffness_.dataPtr();
  auto *const mass = axial_mass_.dataPtr();
  auto *const factor = axial_factor_.dataPtr();
  auto *const work = axial_work_.dataPtr();
  auto *const eigenvectors = axial_eigenvectors_.dataPtr();
  auto *const coefficients = axial_coefficients_.dataPtr();
  auto *const potential = axial_potential_.dataPtr();
  auto *const axial_vector = axial_vector_.dataPtr();
  const auto *const metric = intrinsic_metric_.dataPtr();
  const auto *const connection_difference =
      intrinsic_connection_difference_.dataPtr();
  const auto *const intrinsic_status = intrinsic_point_status_.dataPtr();
  const int harmonic_degree = control.harmonic_degree;
  const amrex::Long point_mode_count =
      static_cast<amrex::Long>(mode_points);
  const amrex::Long projected_count =
      static_cast<amrex::Long>(matrix_size);

  amrex::ParallelFor(
      projected_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long index) noexcept {
        stiffness[index] = 0;
        mass[index] = 0;
        factor[index] = 0;
        work[index] = 0;
        eigenvectors[index] = 0;
      });
  amrex::ParallelFor(
      amrex::Long(surface_count) * basis_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long index) noexcept {
        coefficients[index] = 0;
      });
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        potential[point] = 0;
        axial_vector[point] = 0;
        axial_vector[view.point_count + point] = 0;
      });

  amrex::ParallelFor(
      point_mode_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long index) noexcept {
        const int mode = static_cast<int>(index / view.point_count);
        const amrex::Long point = index - amrex::Long(mode) * view.point_count;
        basis[index] = 0;
        for (int component = 0; component < 2; ++component) {
          gradient[(amrex::Long(component) * basis_count + mode) *
                       view.point_count +
                   point] = 0;
          covector[(amrex::Long(component) * basis_count + mode) *
                       view.point_count +
                   point] = 0;
        }
        for (int component = 0; component < 3; ++component)
          deformation[(amrex::Long(component) * basis_count + mode) *
                          view.point_count +
                      point] = 0;
        const int surface = view.surface_ordinal[point];
        const auto &descriptor = view.descriptors[surface];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels)
          return;
        if (intrinsic_status[point] != 0) {
          amrex::Gpu::Atomic::Max(
              surface_status + surface,
              static_cast<int>(
                  AxialSymmetryStatus::invalid_intrinsic_geometry));
          return;
        }
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (view.itheta[point] + amrex::Real(0.5)) / descriptor.ntheta;
        const amrex::Real phi =
            2 * amrex::Math::pi<amrex::Real>() *
            (view.iphi[point] + amrex::Real(0.5)) / descriptor.nphi;
        amrex::Real value = 0;
        if (!axial_basis_value(mode, harmonic_degree, theta, phi, value)) {
          amrex::Gpu::Atomic::Max(
              surface_status + surface,
              static_cast<int>(AxialSymmetryStatus::invalid_basis));
          return;
        }
        basis[index] = value;
      });

  amrex::ParallelFor(
      point_mode_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long index) noexcept {
        const int mode = static_cast<int>(index / view.point_count);
        const amrex::Long point = index - amrex::Long(mode) * view.point_count;
        const int surface = view.surface_ordinal[point];
        const auto &descriptor = view.descriptors[surface];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels ||
            surface_status[surface] !=
                static_cast<int>(AxialSymmetryStatus::valid))
          return;
        const amrex::Real *const field =
            basis + amrex::Long(mode) * view.point_count;
        const AngularFirstDerivatives derivatives =
            angular_first_derivatives(
                field, descriptor.offset, descriptor.ntheta,
                descriptor.nphi, view.itheta[point], view.iphi[point], 1);
        amrex::Real local_metric[3];
        amrex::Real inverse[3];
        amrex::Real determinant = 0;
        for (int component = 0; component < 3; ++component)
          local_metric[component] =
              metric[amrex::Long(component) * view.point_count + point];
        if (!inverse_intrinsic_metric(local_metric, inverse, determinant)) {
          amrex::Gpu::Atomic::Max(
              surface_status + surface,
              static_cast<int>(
                  AxialSymmetryStatus::invalid_intrinsic_geometry));
          return;
        }
        const amrex::Real sqrt_determinant =
            determinant * amrex::Math::rsqrt(determinant);
        const amrex::Real local_gradient[2]{derivatives.theta,
                                            derivatives.phi};
        const amrex::Real contravariant[2]{
            local_gradient[1] / sqrt_determinant,
            -local_gradient[0] / sqrt_determinant};
        amrex::Real local_covector[2]{0, 0};
        for (int first = 0; first < 2; ++first)
          for (int second = 0; second < 2; ++second)
            local_covector[first] +=
                intrinsic_symmetric_value(local_metric, first, second) *
                contravariant[second];
        bool finite = true;
        for (int component = 0; component < 2; ++component) {
          gradient[(amrex::Long(component) * basis_count + mode) *
                       view.point_count +
                   point] = local_gradient[component];
          covector[(amrex::Long(component) * basis_count + mode) *
                       view.point_count +
                   point] = local_covector[component];
          finite = finite &&
                   amrex::Math::isfinite(local_gradient[component]) &&
                   amrex::Math::isfinite(local_covector[component]);
        }
        if (!finite)
          amrex::Gpu::Atomic::Max(
              surface_status + surface,
              static_cast<int>(AxialSymmetryStatus::invalid_basis));
      });

  amrex::ParallelFor(
      point_mode_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long index) noexcept {
        const int mode = static_cast<int>(index / view.point_count);
        const amrex::Long point = index - amrex::Long(mode) * view.point_count;
        const int surface = view.surface_ordinal[point];
        const auto &descriptor = view.descriptors[surface];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels ||
            surface_status[surface] !=
                static_cast<int>(AxialSymmetryStatus::valid))
          return;
        amrex::Real local_metric[3];
        amrex::Real inverse[3];
        amrex::Real determinant = 0;
        for (int component = 0; component < 3; ++component)
          local_metric[component] =
              metric[amrex::Long(component) * view.point_count + point];
        if (!inverse_intrinsic_metric(local_metric, inverse, determinant)) {
          amrex::Gpu::Atomic::Max(
              surface_status + surface,
              static_cast<int>(
                  AxialSymmetryStatus::invalid_intrinsic_geometry));
          return;
        }
        amrex::Real local_covector[2];
        amrex::Real partial[2][2];
        for (int component = 0; component < 2; ++component) {
          const amrex::Real *const field =
              covector +
              (amrex::Long(component) * basis_count + mode) *
                  view.point_count;
          local_covector[component] = field[point];
          const AngularFirstDerivatives derivatives =
              angular_first_derivatives(
                  field, descriptor.offset, descriptor.ntheta,
                  descriptor.nphi, view.itheta[point], view.iphi[point],
                  intrinsic_index_sign(component));
          partial[0][component] = derivatives.theta;
          partial[1][component] = derivatives.phi;
        }
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (view.itheta[point] + amrex::Real(0.5)) / descriptor.ntheta;
        amrex::Real reference[2][2][2];
        unit_sphere_connection(theta, reference);
        amrex::Real covariant[2][2];
        for (int first = 0; first < 2; ++first)
          for (int second = 0; second < 2; ++second) {
            amrex::Real value = partial[first][second];
            for (int contracted = 0; contracted < 2; ++contracted) {
              const int component = intrinsic_connection_component(
                  contracted, first, second);
              const amrex::Real connection =
                  reference[contracted][first][second] +
                  connection_difference[amrex::Long(component) *
                                            view.point_count +
                                        point];
              value -= connection * local_covector[contracted];
            }
            covariant[first][second] = value;
          }
        amrex::Real divergence = 0;
        for (int first = 0; first < 2; ++first)
          for (int second = 0; second < 2; ++second)
            divergence +=
                intrinsic_symmetric_value(inverse, first, second) *
                covariant[first][second];
        bool finite = amrex::Math::isfinite(divergence);
        for (int first = 0; first < 2; ++first)
          for (int second = first; second < 2; ++second) {
            const amrex::Real value =
                amrex::Real(0.5) *
                    (covariant[first][second] +
                     covariant[second][first]) -
                amrex::Real(0.5) *
                    intrinsic_symmetric_value(local_metric, first, second) *
                    divergence;
            deformation[(amrex::Long(intrinsic_metric_component(
                             first, second)) *
                             basis_count +
                         mode) *
                            view.point_count +
                        point] = value;
            finite = finite && amrex::Math::isfinite(value);
          }
        if (!finite)
          amrex::Gpu::Atomic::Max(
              surface_status + surface,
              static_cast<int>(AxialSymmetryStatus::invalid_basis));
      });

  const amrex::Long gram_count =
      amrex::Long(surface_count) * basis_count * basis_count;
  amrex::ParallelFor(
      gram_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long index) noexcept {
        const int surface = static_cast<int>(
            index / (amrex::Long(basis_count) * basis_count));
        const int pair = static_cast<int>(
            index - amrex::Long(surface) * basis_count * basis_count);
        const int first_mode = pair / basis_count;
        const int second_mode = pair - first_mode * basis_count;
        if (second_mode < first_mode)
          return;
        const auto &descriptor = view.descriptors[surface];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels ||
            surface_status[surface] !=
                static_cast<int>(AxialSymmetryStatus::valid))
          return;
        amrex::Real mass_value = 0;
        amrex::Real stiffness_value = 0;
        for (amrex::Long local = 0; local < descriptor.count; ++local) {
          const amrex::Long point = descriptor.offset + local;
          amrex::Real local_metric[3];
          amrex::Real inverse[3];
          amrex::Real determinant = 0;
          for (int component = 0; component < 3; ++component)
            local_metric[component] =
                metric[amrex::Long(component) * view.point_count + point];
          const amrex::Real weight = view.area_weight[point];
          if (!inverse_intrinsic_metric(local_metric, inverse,
                                        determinant) ||
              !(weight > 0) || !amrex::Math::isfinite(weight)) {
            amrex::Gpu::Atomic::Max(
                surface_status + surface,
                static_cast<int>(
                    AxialSymmetryStatus::invalid_intrinsic_geometry));
            return;
          }
          amrex::Real point_mass = 0;
          amrex::Real point_stiffness = 0;
          for (int first = 0; first < 2; ++first)
            for (int second = 0; second < 2; ++second) {
              const amrex::Real inverse_value =
                  intrinsic_symmetric_value(inverse, first, second);
              const amrex::Real first_gradient =
                  gradient[(amrex::Long(first) * basis_count + first_mode) *
                               view.point_count +
                           point];
              const amrex::Real second_gradient =
                  gradient[(amrex::Long(second) * basis_count +
                            second_mode) *
                               view.point_count +
                           point];
              point_mass +=
                  inverse_value * first_gradient * second_gradient;
              for (int third = 0; third < 2; ++third)
                for (int fourth = 0; fourth < 2; ++fourth) {
                  const amrex::Real first_deformation =
                      deformation[(amrex::Long(intrinsic_metric_component(
                                       first, third)) *
                                       basis_count +
                                   first_mode) *
                                      view.point_count +
                                  point];
                  const amrex::Real second_deformation =
                      deformation[(amrex::Long(intrinsic_metric_component(
                                       second, fourth)) *
                                       basis_count +
                                   second_mode) *
                                      view.point_count +
                                  point];
                  point_stiffness +=
                      inverse_value *
                      intrinsic_symmetric_value(inverse, third, fourth) *
                      first_deformation * second_deformation;
                }
            }
          mass_value += point_mass * weight;
          stiffness_value += point_stiffness * weight;
        }
        if (!amrex::Math::isfinite(mass_value) ||
            !amrex::Math::isfinite(stiffness_value)) {
          amrex::Gpu::Atomic::Max(
              surface_status + surface,
              static_cast<int>(AxialSymmetryStatus::invalid_basis));
          return;
        }
        const amrex::Long matrix_offset =
            amrex::Long(surface) * basis_count * basis_count;
        const amrex::Long first_index =
            matrix_offset + first_mode * basis_count + second_mode;
        mass[first_index] = mass_value;
        stiffness[first_index] = stiffness_value;
        if (first_mode != second_mode) {
          const amrex::Long second_index =
              matrix_offset + second_mode * basis_count + first_mode;
          mass[second_index] = mass_value;
          stiffness[second_index] = stiffness_value;
        }
      });

  const int maximum_sweeps = control.maximum_jacobi_sweeps;
  const amrex::Real eigen_tolerance = control.eigen_tolerance;
  amrex::ParallelFor(
      surface_count, [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        const auto &descriptor = view.descriptors[surface];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels ||
            surface_status[surface] !=
                static_cast<int>(AxialSymmetryStatus::valid))
          return;
        const amrex::Long matrix_offset =
            amrex::Long(surface) * basis_count * basis_count;
        amrex::Real eigenvalue = 0;
        amrex::Real eigen_residual = 0;
        int sweeps = 0;
        const AxialSymmetryStatus status =
            smallest_generalized_symmetric_mode(
                stiffness + matrix_offset, mass + matrix_offset,
                factor + matrix_offset, work + matrix_offset,
                eigenvectors + matrix_offset,
                coefficients + amrex::Long(surface) * basis_count,
                eigenvalue, eigen_residual, sweeps, basis_count,
                maximum_sweeps, eigen_tolerance);
        surface_status[surface] = static_cast<int>(status);
        surface_iterations[surface] = sweeps;
        if (status == AxialSymmetryStatus::valid) {
          surface_values[4 * surface_count + surface] = eigenvalue;
          surface_values[5 * surface_count + surface] = eigen_residual;
        }
      });

  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int surface = view.surface_ordinal[point];
        const auto &descriptor = view.descriptors[surface];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels ||
            surface_status[surface] !=
                static_cast<int>(AxialSymmetryStatus::valid))
          return;
        amrex::Real value = 0;
        amrex::Real local_gradient[2]{0, 0};
        for (int mode = 0; mode < basis_count; ++mode) {
          const amrex::Real coefficient =
              coefficients[amrex::Long(surface) * basis_count + mode];
          value += coefficient *
                   basis[amrex::Long(mode) * view.point_count + point];
          for (int component = 0; component < 2; ++component)
            local_gradient[component] +=
                coefficient *
                gradient[(amrex::Long(component) * basis_count + mode) *
                             view.point_count +
                         point];
        }
        amrex::Real local_metric[3];
        amrex::Real inverse[3];
        amrex::Real determinant = 0;
        for (int component = 0; component < 3; ++component)
          local_metric[component] =
              metric[amrex::Long(component) * view.point_count + point];
        if (!inverse_intrinsic_metric(local_metric, inverse, determinant)) {
          amrex::Gpu::Atomic::Max(
              surface_status + surface,
              static_cast<int>(
                  AxialSymmetryStatus::invalid_intrinsic_geometry));
          return;
        }
        const amrex::Real sqrt_determinant =
            determinant * amrex::Math::rsqrt(determinant);
        const amrex::Real first = local_gradient[1] / sqrt_determinant;
        const amrex::Real second = -local_gradient[0] / sqrt_determinant;
        if (!amrex::Math::isfinite(value) ||
            !amrex::Math::isfinite(first) ||
            !amrex::Math::isfinite(second)) {
          amrex::Gpu::Atomic::Max(
              surface_status + surface,
              static_cast<int>(AxialSymmetryStatus::invalid_basis));
          return;
        }
        potential[point] = value;
        axial_vector[point] = first;
        axial_vector[view.point_count + point] = second;
      });

  const amrex::Real minimum_axis_separation =
      control.minimum_axis_separation;
  amrex::ParallelFor(
      surface_count, [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        const auto &descriptor = view.descriptors[surface];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels ||
            surface_status[surface] !=
                static_cast<int>(AxialSymmetryStatus::valid))
          return;
        amrex::Real minimum =
            std::numeric_limits<amrex::Real>::max();
        amrex::Real maximum =
            -std::numeric_limits<amrex::Real>::max();
        amrex::Long minimum_point = -1;
        amrex::Long maximum_point = -1;
        amrex::Real area = 0;
        amrex::Real potential_integral = 0;
        for (amrex::Long local = 0; local < descriptor.count; ++local) {
          const amrex::Long point = descriptor.offset + local;
          const amrex::Real value = potential[point];
          const amrex::Real weight = view.area_weight[point];
          if (intrinsic_status[point] != 0 ||
              !amrex::Math::isfinite(value) || !(weight > 0) ||
              !amrex::Math::isfinite(weight)) {
            surface_status[surface] = static_cast<int>(
                AxialSymmetryStatus::invalid_intrinsic_geometry);
            return;
          }
          area += weight;
          potential_integral += value * weight;
          if (value < minimum) {
            minimum = value;
            minimum_point = point;
          }
          if (value > maximum) {
            maximum = value;
            maximum_point = point;
          }
        }
        const amrex::Real range = maximum - minimum;
        if (!(area > 0) || !(range > 0) || minimum_point < 0 ||
            maximum_point < 0 || !amrex::Math::isfinite(area) ||
            !amrex::Math::isfinite(range)) {
          surface_status[surface] =
              static_cast<int>(AxialSymmetryStatus::invalid_axis);
          return;
        }
        amrex::Real axis[3];
        amrex::Real separation_squared = 0;
        for (int d = 0; d < 3; ++d) {
          axis[d] = view.position[d][maximum_point] -
                    view.position[d][minimum_point];
          separation_squared += axis[d] * axis[d];
        }
        const amrex::Real separation =
            separation_squared > 0
                ? separation_squared *
                      amrex::Math::rsqrt(separation_squared)
                : 0;
        const amrex::Real areal_radius = axial_square_root(
            area / (4 * amrex::Math::pi<amrex::Real>()));
        if (!(separation > minimum_axis_separation * areal_radius) ||
            !amrex::Math::isfinite(separation)) {
          surface_status[surface] =
              static_cast<int>(AxialSymmetryStatus::invalid_axis);
          return;
        }
        for (int d = 0; d < 3; ++d)
          axis[d] /= separation;
        int dominant = 0;
        for (int d = 1; d < 3; ++d)
          if (axial_absolute_value(axis[d]) >
              axial_absolute_value(axis[dominant]))
            dominant = d;
        const amrex::Real orientation = axis[dominant] < 0 ? -1 : 1;
        const amrex::Real target_range =
            area / (2 * amrex::Math::pi<amrex::Real>());
        const amrex::Real scale = orientation * target_range / range;
        const amrex::Real potential_mean = potential_integral / area;
        const amrex::Real inverse_areal_radius_squared =
            4 * amrex::Math::pi<amrex::Real>() / area;
        const amrex::Real first_extremum =
            scale * (minimum - potential_mean) *
            inverse_areal_radius_squared;
        const amrex::Real second_extremum =
            scale * (maximum - potential_mean) *
            inverse_areal_radius_squared;
        const amrex::Real coordinate_minimum =
            amrex::min(first_extremum, second_extremum);
        const amrex::Real coordinate_maximum =
            amrex::max(first_extremum, second_extremum);
        const amrex::Real normalization_residual =
            axial_absolute_value(
                axial_absolute_value(scale) * range - target_range) /
            target_range;
        const amrex::Real eigenvalue =
            surface_values[4 * surface_count + surface];
        const amrex::Real negative_tolerance =
            eigen_tolerance *
            amrex::max(amrex::Real(1), axial_absolute_value(eigenvalue));
        if (!amrex::Math::isfinite(scale) ||
            !amrex::Math::isfinite(potential_mean) ||
            !(inverse_areal_radius_squared > 0) ||
            !amrex::Math::isfinite(inverse_areal_radius_squared) ||
            !amrex::Math::isfinite(coordinate_minimum) ||
            !amrex::Math::isfinite(coordinate_maximum) ||
            !amrex::Math::isfinite(normalization_residual) ||
            eigenvalue < -negative_tolerance) {
          surface_status[surface] =
              static_cast<int>(AxialSymmetryStatus::invalid_normalization);
          return;
        }
        surface_values[surface] = scale;
        for (int d = 0; d < 3; ++d)
          surface_values[(1 + d) * surface_count + surface] =
              orientation * axis[d];
        surface_values[6 * surface_count + surface] =
            areal_radius *
            axial_square_root(amrex::max(amrex::Real(0), eigenvalue));
        surface_values[7 * surface_count + surface] =
            normalization_residual;
        surface_values[8 * surface_count + surface] = potential_mean;
        surface_values[9 * surface_count + surface] = coordinate_minimum;
        surface_values[10 * surface_count + surface] = coordinate_maximum;
        surface_values[11 * surface_count + surface] =
            inverse_areal_radius_squared;
      });
}

void LogicalSurfaceStorage::prepare_surface_analysis(
    const amrex::Long *const global_stable_ids,
    const int global_surface_count, amrex::Real *const global_reals,
    amrex::Long *const global_longs,
    const AxialSymmetryControl &axial_control,
    const bool compute_coordinate_fluxes,
    const bool compute_weyl_curvature,
    const bool compute_weinberg_fluxes,
    const bool compute_ricci_scalars,
    const bool use_matter_ricci) {
  if (global_surface_count < 0 ||
      (global_surface_count > 0 &&
       (!global_stable_ids || !global_reals || !global_longs)))
    amrex::Abort("ParticleAHFinderX received invalid device analysis buffers");
  if (global_surface_count == 0) {
    if (local_surface_count_ != 0)
      amrex::Abort("ParticleAHFinderX has local surfaces but no global analysis "
                   "stable-ID table");
    return;
  }

  prepare_intrinsic_geometry();
  prepare_axial_symmetry(axial_control);
  if (compute_weyl_curvature &&
      (weyl_electric_.size() !=
           static_cast<std::size_t>(6 * local_point_count_) ||
       weyl_magnetic_.size() !=
           static_cast<std::size_t>(6 * local_point_count_) ||
       weyl_arrival_.size() !=
           static_cast<std::size_t>(local_point_count_)))
    amrex::Abort("ParticleAHFinderX Weyl analysis has no routed pointwise "
                 "curvature workspace");
  if ((compute_weinberg_fluxes || compute_ricci_scalars) &&
      (gauge_value_.size() != static_cast<std::size_t>(
                                  FourGaugeLayout::component_count *
                                  local_point_count_) ||
       gauge_arrival_.size() !=
           static_cast<std::size_t>(local_point_count_)))
    amrex::Abort("ParticleAHFinderX four-dimensional analysis has no routed "
                 "pointwise gauge workspace");
  if (use_matter_ricci &&
      (matter_value_.size() != static_cast<std::size_t>(
                                   FourMatterLayout::component_count *
                                   local_point_count_) ||
       matter_arrival_.size() !=
           static_cast<std::size_t>(local_point_count_)))
    amrex::Abort("ParticleAHFinderX Ricci analysis has no routed pointwise "
                 "matter workspace");

  const amrex::Long global_real_count =
      amrex::Long(surface_analysis_real_components) * global_surface_count;
  const amrex::Long global_long_count =
      amrex::Long(surface_analysis_long_components) * global_surface_count + 1;
  amrex::ParallelFor(
      global_real_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long value) noexcept {
        global_reals[value] = 0;
      });
  amrex::ParallelFor(
      global_long_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long value) noexcept {
        global_longs[value] = 0;
      });
  if (local_surface_count_ == 0)
    return;

  auto *const integral = surface_analysis_integral_.dataPtr();
  auto *const invalid = surface_analysis_invalid_.dataPtr();
  auto *const intrinsic_invalid = surface_intrinsic_invalid_.dataPtr();
  auto *const multipole_invalid = surface_multipole_invalid_.dataPtr();
  auto *const coordinate_flux_invalid =
      surface_coordinate_flux_invalid_.dataPtr();
  auto *const weyl_invalid = surface_weyl_invalid_.dataPtr();
  auto *const weinberg_invalid = surface_weinberg_invalid_.dataPtr();
  auto *const ricci_invalid = surface_ricci_invalid_.dataPtr();
  auto *const speciality_index_invalid =
      surface_speciality_index_invalid_.dataPtr();
  auto *const speciality_difference_invalid =
      surface_speciality_difference_invalid_.dataPtr();
  auto *const axial_status = surface_axial_status_.dataPtr();
  const auto *const axial_iterations =
      surface_axial_iterations_.dataPtr();
  const auto *const axial_values = surface_axial_values_.dataPtr();
  const int surface_count = local_surface_count_;
  amrex::ParallelFor(
      surface_count, [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        invalid[surface] = 0;
        intrinsic_invalid[surface] = 0;
        multipole_invalid[surface] = 0;
        coordinate_flux_invalid[surface] = 0;
        weyl_invalid[surface] = 0;
        weinberg_invalid[surface] = 0;
        ricci_invalid[surface] = 0;
        speciality_index_invalid[surface] = 0;
        speciality_difference_invalid[surface] = 0;
        for (int component = 0;
             component < surface_analysis_real_components; ++component)
          integral[component * surface_count + surface] = 0;
        for (int component = 1; component <= 7; ++component)
          integral[(12 + component) * surface_count + surface] =
              axial_values[component * surface_count + surface];
        integral[21 * surface_count + surface] =
            axial_values[9 * surface_count + surface];
        integral[22 * surface_count + surface] =
            axial_values[10 * surface_count + surface];
      });

  const auto view = device_view();
  const auto *const intrinsic_metric = intrinsic_metric_.dataPtr();
  const auto *const intrinsic_normal_connection =
      intrinsic_normal_connection_.dataPtr();
  const auto *const intrinsic_scalar_curvature =
      intrinsic_scalar_curvature_.dataPtr();
  const auto *const intrinsic_status = intrinsic_point_status_.dataPtr();
  const auto *const axial_vector = axial_vector_.dataPtr();
  const auto *const axial_potential = axial_potential_.dataPtr();
  const auto *const weyl_electric = weyl_electric_.dataPtr();
  const auto *const weyl_magnetic = weyl_magnetic_.dataPtr();
  const auto *const weyl_arrival = weyl_arrival_.dataPtr();
  const auto *const gauge_value = gauge_value_.dataPtr();
  const auto *const gauge_arrival = gauge_arrival_.dataPtr();
  const auto *const matter_value = matter_value_.dataPtr();
  const auto *const matter_arrival = matter_arrival_.dataPtr();
  const bool compute_multipoles = axial_control.compute_multipoles;
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int surface = view.surface_ordinal[point];
        const auto &descriptor = view.descriptors[surface];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels)
          return;
        const amrex::Real weight = view.area_weight[point];
        if (view.geometry_status[point] != 0 || !(weight > 0) ||
            !amrex::Math::isfinite(weight)) {
          amrex::Gpu::Atomic::AddNoRet(invalid + surface, amrex::Long(1));
          if (compute_coordinate_fluxes)
            amrex::Gpu::Atomic::AddNoRet(
                coordinate_flux_invalid + surface, amrex::Long(1));
          if (compute_weyl_curvature)
            amrex::Gpu::Atomic::AddNoRet(weyl_invalid + surface,
                                         amrex::Long(1));
          if (compute_weinberg_fluxes)
            amrex::Gpu::Atomic::AddNoRet(weinberg_invalid + surface,
                                         amrex::Long(1));
          if (compute_ricci_scalars)
            amrex::Gpu::Atomic::AddNoRet(ricci_invalid + surface,
                                         amrex::Long(1));
          return;
        }

        amrex::Real gamma[6];
        amrex::Real curv[6];
        amrex::Real normal[3];
        bool finite = true;
        for (int component = 0; component < 6; ++component) {
          gamma[component] = view.gamma(component, point);
          curv[component] = view.curv(component, point);
          finite = finite && amrex::Math::isfinite(gamma[component]) &&
                   amrex::Math::isfinite(curv[component]);
        }
        for (int d = 0; d < 3; ++d) {
          normal[d] = view.normal_component(d, point);
          finite = finite && amrex::Math::isfinite(normal[d]) &&
                   amrex::Math::isfinite(view.position[d][point]);
        }
        amrex::Real inverse[3][3];
        if (!finite || !inverse_metric(gamma, inverse)) {
          amrex::Gpu::Atomic::AddNoRet(invalid + surface, amrex::Long(1));
          if (compute_coordinate_fluxes)
            amrex::Gpu::Atomic::AddNoRet(
                coordinate_flux_invalid + surface, amrex::Long(1));
          if (compute_weyl_curvature)
            amrex::Gpu::Atomic::AddNoRet(weyl_invalid + surface,
                                         amrex::Long(1));
          if (compute_weinberg_fluxes)
            amrex::Gpu::Atomic::AddNoRet(weinberg_invalid + surface,
                                         amrex::Long(1));
          if (compute_ricci_scalars)
            amrex::Gpu::Atomic::AddNoRet(ricci_invalid + surface,
                                         amrex::Long(1));
          return;
        }

        amrex::Real trace_curv = 0;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            trace_curv +=
                inverse[i][j] * symmetric_value(curv, i, j);

        amrex::Real momentum_density[3]{0, 0, 0};
        amrex::Real curvature_density[3]{0, 0, 0};
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) {
            const amrex::Real kij = symmetric_value(curv, i, j);
            curvature_density[i] += kij * normal[j];
            momentum_density[i] +=
                (kij - trace_curv * symmetric_value(gamma, i, j)) *
                normal[j];
          }
        for (int d = 0; d < 3; ++d)
          finite = finite &&
                   amrex::Math::isfinite(momentum_density[d]) &&
                   amrex::Math::isfinite(curvature_density[d]);
        const amrex::Real scale =
            weight /
            (amrex::Real(8) * amrex::Math::pi<amrex::Real>());
        amrex::Real displacement[3];
        for (int d = 0; d < 3; ++d)
          displacement[d] =
              view.position[d][point] - descriptor.center[d];
        const amrex::Real angular_density[3]{
            displacement[1] * curvature_density[2] -
                displacement[2] * curvature_density[1],
            displacement[2] * curvature_density[0] -
                displacement[0] * curvature_density[2],
            displacement[0] * curvature_density[1] -
                displacement[1] * curvature_density[0]};
        for (int d = 0; d < 3; ++d)
          finite = finite && amrex::Math::isfinite(angular_density[d]);
        if (!finite || !amrex::Math::isfinite(trace_curv) ||
            !amrex::Math::isfinite(scale)) {
          amrex::Gpu::Atomic::AddNoRet(invalid + surface, amrex::Long(1));
          if (compute_coordinate_fluxes)
            amrex::Gpu::Atomic::AddNoRet(
                coordinate_flux_invalid + surface, amrex::Long(1));
          if (compute_weyl_curvature)
            amrex::Gpu::Atomic::AddNoRet(weyl_invalid + surface,
                                         amrex::Long(1));
          if (compute_weinberg_fluxes)
            amrex::Gpu::Atomic::AddNoRet(weinberg_invalid + surface,
                                         amrex::Long(1));
          if (compute_ricci_scalars)
            amrex::Gpu::Atomic::AddNoRet(ricci_invalid + surface,
                                         amrex::Long(1));
          return;
        }

        if (compute_coordinate_fluxes) {
          CoordinateFluxInput flux_input{};
          for (int component = 0; component < 6; ++component) {
            flux_input.gamma[component] = gamma[component];
            flux_input.curv[component] = curv[component];
            for (int derivative = 0; derivative < 3; ++derivative)
              flux_input.d_gamma[derivative][component] =
                  view.d_gamma(derivative, component, point);
          }
          for (int d = 0; d < 3; ++d) {
            flux_input.normal[d] = normal[d];
            flux_input.position[d] = view.position[d][point];
            flux_input.center[d] = descriptor.center[d];
          }
          flux_input.physical_area_weight = weight;
          CoordinateFluxOutput flux_output;
          if (!evaluate_coordinate_fluxes(flux_input, inverse, trace_curv,
                                          flux_output)) {
            amrex::Gpu::Atomic::AddNoRet(
                coordinate_flux_invalid + surface, amrex::Long(1));
          } else {
            constexpr int offset =
                surface_analysis_coordinate_flux_offset;
            amrex::Gpu::Atomic::AddNoRet(
                integral +
                    (offset + CoordinateFluxLayout::adm_energy) *
                        surface_count +
                    surface,
                flux_output.adm_energy);
            amrex::Gpu::Atomic::AddNoRet(
                integral +
                    (offset + CoordinateFluxLayout::conformal_adm_energy) *
                        surface_count +
                    surface,
                flux_output.conformal_adm_energy);
            for (int d = 0; d < 3; ++d) {
              amrex::Gpu::Atomic::AddNoRet(
                  integral +
                      (offset + CoordinateFluxLayout::adm_momentum + d) *
                          surface_count +
                      surface,
                  flux_output.adm_momentum[d]);
              amrex::Gpu::Atomic::AddNoRet(
                  integral +
                      (offset + CoordinateFluxLayout::adm_angular_momentum +
                       d) *
                          surface_count +
                      surface,
                  flux_output.adm_angular_momentum[d]);
              amrex::Gpu::Atomic::AddNoRet(
                  integral +
                      (offset + CoordinateFluxLayout::conformal_adm_momentum +
                       d) *
                          surface_count +
                      surface,
                  flux_output.conformal_adm_momentum[d]);
              amrex::Gpu::Atomic::AddNoRet(
                  integral +
                      (offset +
                       CoordinateFluxLayout::conformal_adm_angular_momentum +
                       d) *
                          surface_count +
                      surface,
                  flux_output.conformal_adm_angular_momentum[d]);
            }
          }
        }

        if (compute_weinberg_fluxes || compute_ricci_scalars) {
          bool metric_valid = gauge_arrival[point] == 1;
          FourMetricInput four_input;
          for (int component = 0; component < 6; ++component) {
            four_input.gamma[component] = gamma[component];
            four_input.curv[component] = curv[component];
            for (int derivative = 0; derivative < 3; ++derivative)
              four_input.d_gamma[derivative][component] =
                  view.d_gamma(derivative, component, point);
          }
          if (metric_valid) {
            four_input.lapse =
                gauge_value[amrex::Long(FourGaugeLayout::lapse) *
                                view.point_count +
                            point];
            four_input.dt_lapse =
                gauge_value[amrex::Long(FourGaugeLayout::dt_lapse) *
                                view.point_count +
                            point];
            for (int derivative = 0; derivative < 3; ++derivative) {
              four_input.d_lapse[derivative] =
                  gauge_value[amrex::Long(FourGaugeLayout::d_lapse +
                                          derivative) *
                                  view.point_count +
                              point];
              for (int component = 0; component < 3; ++component)
                four_input.d_shift[derivative][component] =
                    gauge_value[amrex::Long(FourGaugeLayout::d_shift +
                                            3 * derivative + component) *
                                    view.point_count +
                                point];
            }
            for (int component = 0; component < 3; ++component) {
              four_input.shift[component] =
                  gauge_value[amrex::Long(FourGaugeLayout::shift +
                                          component) *
                                  view.point_count +
                              point];
              four_input.dt_shift[component] =
                  gauge_value[amrex::Long(FourGaugeLayout::dt_shift +
                                          component) *
                                  view.point_count +
                              point];
            }
          }
          FourMetricData four_metric;
          metric_valid = metric_valid &&
                         evaluate_four_metric(four_input, four_metric);

          if (compute_weinberg_fluxes) {
            WeinbergFluxIntegrand weinberg;
            const amrex::Real global_position[3]{
                view.position[0][point], view.position[1][point],
                view.position[2][point]};
            const bool valid =
                metric_valid && evaluate_weinberg_flux_integrand(
                                    four_metric, global_position, normal,
                                    weinberg);
            if (!valid) {
              amrex::Gpu::Atomic::AddNoRet(
                  weinberg_invalid + surface, amrex::Long(1));
            } else {
              constexpr int offset = surface_analysis_weinberg_offset;
              amrex::Gpu::Atomic::AddNoRet(
                  integral + offset * surface_count + surface,
                  weinberg.energy * weight);
              for (int component = 0; component < 3; ++component) {
                amrex::Gpu::Atomic::AddNoRet(
                    integral + (offset + 1 + component) * surface_count +
                        surface,
                    weinberg.momentum[component] * weight);
                amrex::Gpu::Atomic::AddNoRet(
                    integral + (offset + 4 + component) * surface_count +
                        surface,
                    weinberg.angular_momentum[component] * weight);
              }
            }
          }

          if (compute_ricci_scalars) {
            bool valid = metric_valid &&
                         (!use_matter_ricci || matter_arrival[point] == 1);
            const amrex::Real theta =
                amrex::Math::pi<amrex::Real>() *
                (view.itheta[point] + amrex::Real(0.5)) /
                descriptor.ntheta;
            const amrex::Real phi =
                2 * amrex::Math::pi<amrex::Real>() *
                (view.iphi[point] + amrex::Real(0.5)) /
                descriptor.nphi;
            const AngularGeometry basis = angular_geometry(theta, phi);
            amrex::Real tangent_theta[3];
            amrex::Real tangent_phi[3];
            for (int d = 0; d < 3; ++d) {
              tangent_theta[d] =
                  view.angular_component(0, point) * basis.radial[d] +
                  view.height[point] * basis.radial_theta[d];
              tangent_phi[d] =
                  view.angular_component(1, point) * basis.radial[d] +
                  view.height[point] * basis.radial_phi[d];
            }

            SurfaceTriad triad;
            NullTetrad tetrad;
            MatterRicciTensor ricci;
            RicciNPScalars scalars;
            valid = valid && surface_triad(gamma, normal, tangent_theta,
                                           tangent_phi, triad) &&
                    evaluate_null_tetrad(four_input, four_metric, triad,
                                         tetrad);
            if (valid && use_matter_ricci) {
              MatterRicciInput matter;
              for (int component = 0;
                   component < FourMatterLayout::component_count; ++component)
                matter.stress_energy[component] =
                    matter_value[amrex::Long(component) * view.point_count +
                                 point];
              valid = evaluate_matter_ricci(matter, four_metric, ricci);
            }
            valid = valid && project_matter_ricci(ricci, tetrad, scalars);
            if (!valid) {
              amrex::Gpu::Atomic::AddNoRet(ricci_invalid + surface,
                                           amrex::Long(1));
            } else {
              amrex::Real values[RicciIntegralLayout::component_count]{
                  scalars.phi00,
                  scalars.phi01.real,
                  scalars.phi01.imag,
                  scalars.phi02.real,
                  scalars.phi02.imag,
                  scalars.phi11,
                  scalars.phi12.real,
                  scalars.phi12.imag,
                  scalars.phi22,
                  scalars.lambda,
              };
              for (int component = 0;
                   component < RicciIntegralLayout::component_count;
                   ++component)
                amrex::Gpu::Atomic::AddNoRet(
                    integral +
                        (surface_analysis_ricci_offset + component) *
                            surface_count +
                        surface,
                    values[component] * weight);
            }
          }
        }

        if (compute_weyl_curvature) {
          if (weyl_arrival[point] != 1) {
            amrex::Gpu::Atomic::AddNoRet(weyl_invalid + surface,
                                         amrex::Long(1));
          } else {
            ElectricMagneticWeyl curvature;
            for (int component = 0; component < 6; ++component) {
              curvature.electric[component] =
                  weyl_electric[amrex::Long(component) *
                                    view.point_count +
                                point];
              curvature.magnetic[component] =
                  weyl_magnetic[amrex::Long(component) *
                                    view.point_count +
                                point];
            }
            const amrex::Real theta =
                amrex::Math::pi<amrex::Real>() *
                (view.itheta[point] + amrex::Real(0.5)) /
                descriptor.ntheta;
            const amrex::Real phi =
                2 * amrex::Math::pi<amrex::Real>() *
                (view.iphi[point] + amrex::Real(0.5)) /
                descriptor.nphi;
            const AngularGeometry basis = angular_geometry(theta, phi);
            amrex::Real tangent_theta[3];
            amrex::Real tangent_phi[3];
            for (int d = 0; d < 3; ++d) {
              tangent_theta[d] =
                  view.angular_component(0, point) * basis.radial[d] +
                  view.height[point] * basis.radial_theta[d];
              tangent_phi[d] =
                  view.angular_component(1, point) * basis.radial[d] +
                  view.height[point] * basis.radial_phi[d];
            }
            SurfaceTriad triad;
            WeylScalars scalars;
            if (!surface_triad(gamma, normal, tangent_theta, tangent_phi,
                               triad) ||
                !weyl_scalars_from_electric_magnetic(curvature, triad,
                                                      scalars)) {
              amrex::Gpu::Atomic::AddNoRet(weyl_invalid + surface,
                                           amrex::Long(1));
            } else {
              ComplexValue values[7];
              for (int scalar = 0; scalar < 5; ++scalar)
                values[scalar] = scalars.psi[scalar];
              values[5] = scalars.invariant_i;
              values[6] = scalars.invariant_j;
              for (int value = 0; value < 7; ++value) {
                const int component =
                    surface_analysis_weyl_offset + 2 * value;
                amrex::Gpu::Atomic::AddNoRet(
                    integral + component * surface_count + surface,
                    values[value].real * weight);
                amrex::Gpu::Atomic::AddNoRet(
                    integral + (component + 1) * surface_count + surface,
                    values[value].imag * weight);
              }
              if (!scalars.speciality_index_valid) {
                amrex::Gpu::Atomic::AddNoRet(
                    speciality_index_invalid + surface, amrex::Long(1));
              } else {
                constexpr int component =
                    surface_analysis_weyl_offset +
                    WeylIntegralLayout::speciality_index;
                amrex::Gpu::Atomic::AddNoRet(
                    integral + component * surface_count + surface,
                    scalars.speciality_index.real * weight);
                amrex::Gpu::Atomic::AddNoRet(
                    integral + (component + 1) * surface_count + surface,
                    scalars.speciality_index.imag * weight);
              }
              if (!scalars.speciality_difference_valid) {
                amrex::Gpu::Atomic::AddNoRet(
                    speciality_difference_invalid + surface,
                    amrex::Long(1));
              } else {
                constexpr int component =
                    surface_analysis_weyl_offset +
                    WeylIntegralLayout::speciality_difference;
                amrex::Gpu::Atomic::AddNoRet(
                    integral + component * surface_count + surface,
                    scalars.speciality_difference.real * weight);
                amrex::Gpu::Atomic::AddNoRet(
                    integral + (component + 1) * surface_count + surface,
                    scalars.speciality_difference.imag * weight);
              }
            }
          }
        }

        if (intrinsic_status[point] != 0) {
          amrex::Gpu::Atomic::AddNoRet(intrinsic_invalid + surface,
                                       amrex::Long(1));
        } else {
          amrex::Real metric[3];
          amrex::Real metric_inverse[3];
          amrex::Real metric_determinant = 0;
          for (int component = 0; component < 3; ++component)
            metric[component] =
                intrinsic_metric[amrex::Long(component) * view.point_count +
                                 point];
          const amrex::Real omega[2]{
              intrinsic_normal_connection[point],
              intrinsic_normal_connection[view.point_count + point]};
          const amrex::Real curvature =
              intrinsic_scalar_curvature[point];
          amrex::Real omega_squared = 0;
          bool intrinsic_finite = inverse_intrinsic_metric(
              metric, metric_inverse, metric_determinant);
          for (int first = 0; first < 2; ++first)
            for (int second = 0; second < 2; ++second)
              omega_squared +=
                  intrinsic_symmetric_value(metric_inverse, first, second) *
                  omega[first] * omega[second];
          intrinsic_finite = intrinsic_finite &&
                             amrex::Math::isfinite(curvature) &&
                             amrex::Math::isfinite(omega_squared) &&
                             omega_squared >= 0;
          if (!intrinsic_finite)
            amrex::Gpu::Atomic::AddNoRet(intrinsic_invalid + surface,
                                         amrex::Long(1));
          else {
            amrex::Gpu::Atomic::AddNoRet(
                integral + 10 * surface_count + surface,
                curvature * weight);
            amrex::Gpu::Atomic::AddNoRet(
                integral + 11 * surface_count + surface,
                omega_squared * weight);
          }
        }

        if (axial_status[surface] ==
            static_cast<int>(AxialSymmetryStatus::valid)) {
          const amrex::Real axial_contraction =
              axial_vector[point] *
                  intrinsic_normal_connection[point] +
              axial_vector[view.point_count + point] *
                  intrinsic_normal_connection[view.point_count + point];
          const amrex::Real spin_density =
              -axial_values[surface] * axial_contraction;
          const amrex::Real axial_spin_integral =
              spin_density * weight /
              (amrex::Real(8) * amrex::Math::pi<amrex::Real>());
          const amrex::Real zeta =
              axial_values[surface] *
              (axial_potential[point] -
               axial_values[8 * surface_count + surface]) *
              axial_values[11 * surface_count + surface];
          if (!amrex::Math::isfinite(axial_spin_integral) ||
              !amrex::Math::isfinite(zeta))
            amrex::Gpu::Atomic::Max(
                axial_status + surface,
                static_cast<int>(
                    AxialSymmetryStatus::invalid_spin_integral));
          else {
            amrex::Gpu::Atomic::AddNoRet(
                integral + 12 * surface_count + surface,
                axial_spin_integral);
            amrex::Gpu::Atomic::AddNoRet(
                integral + 20 * surface_count + surface, zeta * weight);
            if (compute_multipoles) {
              amrex::Real polynomial[multipole_component_count];
              amrex::Real derivative[multipole_component_count];
              if (intrinsic_status[point] != 0 ||
                  !legendre_values_and_derivatives(
                      zeta, polynomial, derivative)) {
                amrex::Gpu::Atomic::AddNoRet(
                    multipole_invalid + surface, amrex::Long(1));
              } else {
                const amrex::Real curvature =
                    intrinsic_scalar_curvature[point];
                for (int degree = 0;
                     degree <= maximum_multipole_degree; ++degree) {
                  const amrex::Real mass_moment =
                      amrex::Real(0.25) * curvature *
                      polynomial[degree] * weight;
                  const amrex::Real spin_moment =
                      spin_density * derivative[degree] * weight;
                  if (!amrex::Math::isfinite(mass_moment) ||
                      !amrex::Math::isfinite(spin_moment)) {
                    amrex::Gpu::Atomic::AddNoRet(
                        multipole_invalid + surface, amrex::Long(1));
                    break;
                  }
                  amrex::Gpu::Atomic::AddNoRet(
                      integral + (23 + degree) * surface_count + surface,
                      mass_moment);
                  amrex::Gpu::Atomic::AddNoRet(
                      integral +
                          (23 + multipole_component_count + degree) *
                              surface_count +
                          surface,
                      spin_moment);
                }
              }
            }
          }
        }

        amrex::Gpu::Atomic::AddNoRet(integral + surface, weight);
        for (int d = 0; d < 3; ++d) {
          amrex::Gpu::Atomic::AddNoRet(
              integral + (1 + d) * surface_count + surface,
              momentum_density[d] * scale);
          amrex::Gpu::Atomic::AddNoRet(
              integral + (4 + d) * surface_count + surface,
              curvature_density[d] * scale);
          amrex::Gpu::Atomic::AddNoRet(
              integral + (7 + d) * surface_count + surface,
              angular_density[d] * scale);
        }
      });

  const auto *const descriptors = descriptors_.dataPtr();
  amrex::ParallelFor(
      surface_count, [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        const auto &descriptor = descriptors[surface];
        if (!descriptor.due ||
            descriptor.angular_level + 1 != descriptor.angular_levels)
          return;

        int lower = 0;
        int upper = global_surface_count;
        while (lower < upper) {
          const int middle = lower + (upper - lower) / 2;
          if (global_stable_ids[middle] < descriptor.stable_id)
            lower = middle + 1;
          else
            upper = middle;
        }
        if (lower >= global_surface_count ||
            global_stable_ids[lower] != descriptor.stable_id) {
          amrex::Gpu::Atomic::AddNoRet(
              global_longs +
                  surface_analysis_long_components * global_surface_count,
              amrex::Long(1));
          return;
        }

        for (int component = 0;
             component < surface_analysis_real_components; ++component)
          amrex::Gpu::Atomic::AddNoRet(
              global_reals +
                  surface_analysis_real_components * lower + component,
              integral[component * surface_count + surface]);
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower,
            amrex::Long(1));
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower + 1,
            invalid[surface]);
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower + 2,
            intrinsic_invalid[surface]);
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower + 3,
            amrex::Long(axial_status[surface]));
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower + 4,
            amrex::Long(axial_iterations[surface]));
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower + 5,
            multipole_invalid[surface]);
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower + 6,
            coordinate_flux_invalid[surface]);
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower + 7,
            weyl_invalid[surface]);
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower + 8,
            speciality_index_invalid[surface]);
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower + 9,
            speciality_difference_invalid[surface]);
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower + 10,
            weinberg_invalid[surface]);
        amrex::Gpu::Atomic::AddNoRet(
            global_longs + surface_analysis_long_components * lower + 11,
            ricci_invalid[surface]);
      });
}

amrex::Real LogicalSurfaceStorage::local_minkowski_expansion_error() const {
  amrex::ReduceOps<amrex::ReduceOpMax> reduce_ops;
  amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  const amrex::Real *const expansion = expansion_.dataPtr();
  const amrex::Real *const height = height_.dataPtr();
  const int *const geometry_status = geometry_status_.dataPtr();
  const int *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  reduce_ops.eval(
      local_point_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept -> ReduceTuple {
        if (!descriptors[surface_ordinal[point]].due)
          return {0};
        if (geometry_status[point] != 0)
          return {0};
        const amrex::Real error = expansion[point] - 2 / height[point];
        return {error < 0 ? -error : error};
      });
  return amrex::get<0>(reduce_data.value());
}

void LogicalSurfaceStorage::begin_relaxation_search(
    const amrex::Real eta_times_mass) {
  auto *const mutable_descriptors = descriptors_.dataPtr();
  auto *const pseudo_time = surface_pseudo_time_.dataPtr();
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        mutable_descriptors[surface].relaxation_state =
            mutable_descriptors[surface].due
                ? static_cast<int>(SurfaceRelaxationState::active)
                : static_cast<int>(SurfaceRelaxationState::idle);
        pseudo_time[surface] = 0;
      });
  auto *const height = height_.dataPtr();
  auto *const velocity = relaxation_velocity_.dataPtr();
  auto *const committed = committed_state_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  const amrex::Long count = local_point_count_;
  amrex::ParallelFor(local_point_count_,
                     [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
                       const amrex::Real eta =
                           eta_times_mass /
                           descriptors[surface_ordinal[point]].mass_scale;
                       if (descriptors[surface_ordinal[point]]
                               .relaxation_state ==
                           static_cast<int>(
                               SurfaceRelaxationState::active)) {
                         committed[point] = height[point];
                         committed[count + point] = velocity[point];
                         velocity[point] = eta * height[point];
                       }
                     });
}

void LogicalSurfaceStorage::finish_relaxation_search() {
  auto *const descriptors = descriptors_.dataPtr();
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        if (descriptors[surface].relaxation_state ==
            static_cast<int>(SurfaceRelaxationState::active))
          descriptors[surface].relaxation_state =
              static_cast<int>(SurfaceRelaxationState::failed);
      });

  const amrex::Long count = local_point_count_;
  auto *const height = height_.dataPtr();
  auto *const velocity = relaxation_velocity_.dataPtr();
  auto *const committed = committed_state_.dataPtr();
  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  amrex::ParallelFor(
      count, [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int state =
            descriptors[surface_ordinal[point]].relaxation_state;
        if (state == static_cast<int>(SurfaceRelaxationState::converged) ||
            state == static_cast<int>(SurfaceRelaxationState::refine)) {
          velocity[point] = 0;
          committed[point] = height[point];
          committed[count + point] = 0;
        } else if (state ==
                   static_cast<int>(SurfaceRelaxationState::failed)) {
          height[point] = committed[point];
          velocity[point] = committed[count + point];
        }
      });
}

void LogicalSurfaceStorage::defer_relaxation_search() {
  const amrex::Long count = local_point_count_;
  auto *const descriptors = descriptors_.dataPtr();
  auto *const height = height_.dataPtr();
  auto *const velocity = relaxation_velocity_.dataPtr();
  const auto *const committed = committed_state_.dataPtr();
  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  amrex::ParallelFor(
      count, [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int surface = surface_ordinal[point];
        if (descriptors[surface].due) {
          height[point] = committed[point];
          velocity[point] = committed[count + point];
        }
      });
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        if (descriptors[surface].due)
          descriptors[surface].relaxation_state =
              static_cast<int>(SurfaceRelaxationState::idle);
      });
}

void LogicalSurfaceStorage::begin_relaxation_step() {
  const amrex::Long count = local_point_count_;
  const auto *const height = height_.dataPtr();
  const auto *const velocity = relaxation_velocity_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  auto *const base = relaxation_base_.dataPtr();
  auto *const status = relaxation_status_.dataPtr();
  amrex::ParallelFor(count,
                     [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
                       if (!descriptors[surface_ordinal[point]].due)
                         return;
                       base[point] = height[point];
                       base[count + point] = velocity[point];
                       status[point] = 0;
                     });
}

void LogicalSurfaceStorage::store_extrapolation_reference(
    const std::vector<SurfaceLayerSelection> &selections) {
  if (selections.empty())
    return;
  if (selections.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX extrapolation reference table exceeds "
                 "the portable device-kernel index range");
  amrex::Gpu::DeviceVector<SurfaceLayerSelection> device_selections(
      selections.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, selections.begin(),
                   selections.end(), device_selections.begin());
  const auto *const selected = device_selections.dataPtr();
  const int selection_count = static_cast<int>(selections.size());
  const auto *const descriptors = descriptors_.dataPtr();
  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const height = height_.dataPtr();
  auto *const reference = extrapolation_reference_.dataPtr();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &surface = descriptors[surface_ordinal[point]];
        if (surface.relaxation_state ==
                static_cast<int>(SurfaceRelaxationState::active) &&
            selected_surface_layer(surface, selected, selection_count))
          reference[point] = height[point];
      });
  amrex::Gpu::streamSynchronize();
}

void LogicalSurfaceStorage::begin_extrapolation_trials(
    const std::vector<SurfaceLayerSelection> &selections) {
  if (selections.empty())
    return;
  if (selections.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX extrapolation trial table exceeds the "
                 "portable device-kernel index range");
  amrex::Gpu::DeviceVector<SurfaceLayerSelection> device_selections(
      selections.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, selections.begin(),
                   selections.end(), device_selections.begin());
  const auto *const selected = device_selections.dataPtr();
  const int selection_count = static_cast<int>(selections.size());
  const auto *const descriptors = descriptors_.dataPtr();
  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const height = height_.dataPtr();
  const auto *const velocity = relaxation_velocity_.dataPtr();
  const auto *const reference = extrapolation_reference_.dataPtr();
  auto *const base = relaxation_base_.dataPtr();
  auto *const mode_displacement =
      surface_extrapolation_displacement_.dataPtr();
  auto *const mode_weight = surface_extrapolation_weight_.dataPtr();
  const int surface_count = local_surface_count_;
  const amrex::Long count = local_point_count_;
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        for (int mode = 0; mode < 4; ++mode) {
          mode_displacement[mode * surface_count + surface] = 0;
          mode_weight[mode * surface_count + surface] = 0;
        }
      });
  amrex::ParallelFor(
      count, [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int ordinal = surface_ordinal[point];
        const auto &surface = descriptors[ordinal];
        if (surface.relaxation_state ==
                static_cast<int>(SurfaceRelaxationState::active) &&
            selected_surface_layer(surface, selected, selection_count)) {
          base[point] = height[point];
          base[count + point] = velocity[point];
          const amrex::Long logical_index = point - surface.offset;
          const int itheta =
              static_cast<int>(logical_index / surface.nphi);
          const int iphi =
              static_cast<int>(logical_index % surface.nphi);
          const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                                    (itheta + amrex::Real(0.5)) /
                                    surface.ntheta;
          const amrex::Real phi = 2 * amrex::Math::pi<amrex::Real>() *
                                  (iphi + amrex::Real(0.5)) /
                                  surface.nphi;
          const AngularGeometry basis = angular_geometry(theta, phi);
          const amrex::Real weight = amrex::Math::sincos(theta).first;
          const amrex::Real displacement =
              height[point] - reference[point];
          amrex::Gpu::Atomic::AddNoRet(
              mode_displacement + ordinal, weight * displacement);
          amrex::Gpu::Atomic::AddNoRet(mode_weight + ordinal, weight);
          for (int d = 0; d < 3; ++d) {
            const amrex::Real radial = basis.radial[d];
            const amrex::Long mode =
                amrex::Long(d + 1) * surface_count + ordinal;
            amrex::Gpu::Atomic::AddNoRet(
                mode_displacement + mode,
                weight * displacement * radial);
            amrex::Gpu::Atomic::AddNoRet(mode_weight + mode,
                                         weight * radial * radial);
          }
        }
      });
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        for (int mode = 0; mode < 4; ++mode) {
          const int component = mode * surface_count + surface;
          if (mode_weight[component] > 0)
            mode_displacement[component] /= mode_weight[component];
        }
      });
  amrex::Gpu::streamSynchronize();
}

void LogicalSurfaceStorage::set_extrapolation_trial(
    const std::vector<SurfaceExtrapolationControl> &controls,
    const amrex::Real eta_times_mass,
    const amrex::Real minimum_radius_factor,
    const amrex::Real maximum_radius_factor,
    const bool reset_accepted_velocity) {
  if (controls.empty())
    return;
  if (controls.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX extrapolation control table exceeds the "
                 "portable device-kernel index range");
  for (std::size_t index = 0; index < controls.size(); ++index) {
    const auto &control = controls[index];
    if (control.stable_id <= 0 || control.angular_level < 0 ||
        !(control.overstep >= 1) ||
        !amrex::Math::isfinite(control.overstep) ||
        (control.mode != RelaxationExtrapolationMode::full_shape &&
         control.mode != RelaxationExtrapolationMode::low_modes) ||
        (index > 0 &&
         (controls[index - 1].stable_id > control.stable_id ||
          (controls[index - 1].stable_id == control.stable_id &&
           controls[index - 1].angular_level >= control.angular_level))))
      amrex::Abort("ParticleAHFinderX extrapolation controls must be finite, "
                   "unique, and ordered");
  }

  amrex::Gpu::DeviceVector<SurfaceExtrapolationControl> device_controls(
      controls.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, controls.begin(), controls.end(),
                   device_controls.begin());
  const auto *const trial = device_controls.dataPtr();
  const int control_count = static_cast<int>(controls.size());
  const auto *const descriptors = descriptors_.dataPtr();
  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  auto *const invalid = surface_invalid_.dataPtr();
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        invalid[surface] = 0;
      });

  const amrex::Long count = local_point_count_;
  const auto *const base = relaxation_base_.dataPtr();
  const auto *const reference = extrapolation_reference_.dataPtr();
  const auto *const mode_displacement =
      surface_extrapolation_displacement_.dataPtr();
  const int surface_count = local_surface_count_;
  amrex::ParallelFor(
      count, [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int ordinal = surface_ordinal[point];
        const auto &surface = descriptors[ordinal];
        if (surface.relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return;
        const int control = surface_extrapolation_control(
            surface, trial, control_count);
        if (control < 0)
          return;
        amrex::Real candidate;
        if (trial[control].mode ==
            RelaxationExtrapolationMode::low_modes) {
          const amrex::Real displacement =
              low_mode_extrapolation_displacement(
                  surface, point, ordinal, surface_count,
                  mode_displacement);
          candidate = relaxation_extrapolated_low_modes(
              base[point], displacement, trial[control].overstep);
        } else {
          candidate = relaxation_extrapolated_height(
              reference[point], base[point], trial[control].overstep);
        }
        const amrex::Real minimum_radius =
            minimum_radius_factor * surface.radius;
        const amrex::Real maximum_radius =
            maximum_radius_factor * surface.radius;
        if (!amrex::Math::isfinite(candidate) ||
            !(candidate >= minimum_radius) ||
            !(candidate <= maximum_radius))
          amrex::Gpu::Atomic::AddNoRet(invalid + ordinal, amrex::Long(1));
      });

  auto *const height = height_.dataPtr();
  auto *const velocity = relaxation_velocity_.dataPtr();
  auto *const expansion = expansion_.dataPtr();
  auto *const geometry_status = geometry_status_.dataPtr();
  auto *const relaxation_status = relaxation_status_.dataPtr();
  amrex::ParallelFor(
      count, [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int ordinal = surface_ordinal[point];
        const auto &surface = descriptors[ordinal];
        if (surface.relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return;
        const int control = surface_extrapolation_control(
            surface, trial, control_count);
        if (control < 0)
          return;
        const amrex::Real overstep =
            invalid[ordinal] == 0 ? trial[control].overstep : amrex::Real(1);
        if (trial[control].mode ==
            RelaxationExtrapolationMode::low_modes) {
          const amrex::Real displacement =
              low_mode_extrapolation_displacement(
                  surface, point, ordinal, surface_count,
                  mode_displacement);
          height[point] = relaxation_extrapolated_low_modes(
              base[point], displacement, overstep);
        } else {
          height[point] = relaxation_extrapolated_height(
              reference[point], base[point], overstep);
        }
        velocity[point] = reset_accepted_velocity && overstep > 1
                              ? eta_times_mass / surface.mass_scale *
                                    height[point]
                              : base[count + point];
        expansion[point] = 0;
        geometry_status[point] = 1;
        relaxation_status[point] = 0;
      });
  amrex::Gpu::streamSynchronize();
}

void LogicalSurfaceStorage::prepare_relaxation_time_steps(
    const amrex::Real cfl_factor) {
  if (!(cfl_factor > 0) || !amrex::Math::isfinite(cfl_factor))
    amrex::Abort("ParticleAHFinderX received an invalid relaxation CFL "
                 "factor");
  const amrex::Real inactive_step =
      std::numeric_limits<amrex::Real>::max();
  auto *const time_step = surface_time_step_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        time_step[surface] = inactive_step;
      });

  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const height = height_.dataPtr();
  const auto *const itheta_values = itheta_.dataPtr();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int ordinal = surface_ordinal[point];
        const auto &surface = descriptors[ordinal];
        if (surface.relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return;
        const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                                  (itheta_values[point] + amrex::Real(0.5)) /
                                  surface.ntheta;
        const amrex::Real dtheta =
            amrex::Math::pi<amrex::Real>() / surface.ntheta;
        const amrex::Real dphi =
            2 * amrex::Math::pi<amrex::Real>() / surface.nphi;
        const amrex::Real sin_theta = amrex::Math::sincos(theta).first;
        const amrex::Real spacing =
            cfl_factor * height[point] *
            amrex::min(dtheta, sin_theta * dphi);
        amrex::Gpu::Atomic::Min(time_step + ordinal, spacing);
      });

}

amrex::Real
LogicalSurfaceStorage::local_minimum_relaxation_time_step() const {
  const amrex::Real inactive_step =
      std::numeric_limits<amrex::Real>::max();
  const auto *const time_step = surface_time_step_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  amrex::ReduceOps<amrex::ReduceOpMin> reduce_ops;
  amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  reduce_ops.eval(
      local_surface_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept -> ReduceTuple {
        if (descriptors[surface].relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return {inactive_step};
        const amrex::Real dt = time_step[surface];
        if (!(dt > 0) || !amrex::Math::isfinite(dt))
          return {amrex::Real(0)};
        return {dt};
      });
  return amrex::get<0>(reduce_data.value());
}

void LogicalSurfaceStorage::accumulate_relaxation_pseudo_time() {
  auto *const pseudo_time = surface_pseudo_time_.dataPtr();
  const auto *const time_step = surface_time_step_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        if (descriptors[surface].relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return;
        const amrex::Real dt = time_step[surface];
        if (dt > 0 && amrex::Math::isfinite(dt))
          pseudo_time[surface] += dt;
      });
}

amrex::Real
LogicalSurfaceStorage::local_minimum_relaxation_pseudo_time() const {
  const amrex::Real inactive_time =
      std::numeric_limits<amrex::Real>::max();
  const auto *const pseudo_time = surface_pseudo_time_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  amrex::ReduceOps<amrex::ReduceOpMin> reduce_ops;
  amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  reduce_ops.eval(
      local_surface_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept -> ReduceTuple {
        if (descriptors[surface].relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return {inactive_time};
        const amrex::Real time = pseudo_time[surface];
        if (!(time >= 0) || !amrex::Math::isfinite(time))
          return {amrex::Real(-1)};
        return {time};
      });
  return amrex::get<0>(reduce_data.value());
}

void LogicalSurfaceStorage::advance_relaxation_stage(
    const RelaxationIntegrator integrator, const int stage,
    const amrex::Real eta_times_mass,
    const amrex::Real dissipation_strength,
    const amrex::Real minimum_radius_factor,
    const amrex::Real maximum_radius_factor) {
  if (stage < 0 || stage >= relaxation_stage_count(integrator))
    amrex::Abort("ParticleAHFinderX received an invalid relaxation stage");
  const amrex::Long count = local_point_count_;
  auto *const height = height_.dataPtr();
  auto *const velocity = relaxation_velocity_.dataPtr();
  const auto *const expansion = expansion_.dataPtr();
  const auto *const geometry_status = geometry_status_.dataPtr();
  auto *const relaxation_status = relaxation_status_.dataPtr();
  const auto *const base = relaxation_base_.dataPtr();
  auto *const rhs = relaxation_rhs_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const time_step = surface_time_step_.dataPtr();
  const bool fused_pointwise_stage = dissipation_strength == 0;
  amrex::ParallelFor(
      count, [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int ordinal = surface_ordinal[point];
        const auto &surface = descriptors[ordinal];
        if (surface.relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return;
        const amrex::Real dt = time_step[ordinal];
        if (geometry_status[point] != 0 || relaxation_status[point] != 0 ||
            !(dt > 0) || !amrex::Math::isfinite(dt)) {
          relaxation_status[point] = 1;
          return;
        }
        const amrex::Real eta = eta_times_mass / surface.mass_scale;
        amrex::Real rhs_height = velocity[point] - eta * height[point];
        amrex::Real rhs_velocity = -expansion[point];
        if (dissipation_strength > 0) {
          const int itheta = static_cast<int>(
              (point - surface.offset) / surface.nphi);
          const int iphi = static_cast<int>(
              point - surface.offset - amrex::Long(itheta) * surface.nphi);
          const auto height_filter = angular_dissipation(
              height, surface.offset, surface.ntheta, surface.nphi,
              itheta, iphi);
          const auto velocity_filter = angular_dissipation(
              velocity, surface.offset, surface.ntheta, surface.nphi,
              itheta, iphi);
          const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                                    (itheta + amrex::Real(0.5)) /
                                    surface.ntheta;
          const amrex::Real sin_theta = amrex::Math::sincos(theta).first;
          const amrex::Real scale = dissipation_strength / height[point];
          rhs_height += scale *
                        (height_filter.theta +
                         height_filter.phi / sin_theta);
          rhs_velocity += scale *
                          (velocity_filter.theta +
                           velocity_filter.phi / sin_theta);
        }
        rhs[amrex::Long(2 * stage) * count + point] = rhs_height;
        rhs[amrex::Long(2 * stage + 1) * count + point] = rhs_velocity;
        if (fused_pointwise_stage) {
          const amrex::Real minimum_radius =
              minimum_radius_factor * surface.radius;
          const amrex::Real maximum_radius =
              maximum_radius_factor * surface.radius;
          const auto update = relaxation_stage_update(
              integrator, stage, count, point, base, rhs,
              dt, rhs_height, rhs_velocity,
              minimum_radius, maximum_radius);
          if (!update.valid) {
            relaxation_status[point] = 1;
            return;
          }
          height[point] = update.height;
          velocity[point] = update.velocity;
        }
      });

  if (fused_pointwise_stage)
    return;

  // Keep neighbor reads and stage writes in separate kernels. This is
  // required when angular dissipation is enabled and also makes each stage a
  // well-defined explicit update on every accelerator backend.
  amrex::ParallelFor(
      count, [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &surface = descriptors[surface_ordinal[point]];
        if (surface.relaxation_state !=
                static_cast<int>(SurfaceRelaxationState::active) ||
            relaxation_status[point] != 0)
          return;
        const amrex::Real rhs_height =
            rhs[amrex::Long(2 * stage) * count + point];
        const amrex::Real rhs_velocity =
            rhs[amrex::Long(2 * stage + 1) * count + point];
        const amrex::Real minimum_radius =
            minimum_radius_factor * surface.radius;
        const amrex::Real maximum_radius =
            maximum_radius_factor * surface.radius;
        const auto update = relaxation_stage_update(
            integrator, stage, count, point, base, rhs,
            time_step[surface_ordinal[point]], rhs_height, rhs_velocity,
            minimum_radius, maximum_radius);
        if (!update.valid) {
          relaxation_status[point] = 1;
          return;
        }
        height[point] = update.height;
        velocity[point] = update.velocity;
      });
}

amrex::Long LogicalSurfaceStorage::local_relaxation_invalid() const {
  amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
  amrex::ReduceData<amrex::Long> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  const int *const status = relaxation_status_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  reduce_ops.eval(
      local_point_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept -> ReduceTuple {
        if (!descriptors[surface_ordinal[point]].due)
          return {amrex::Long(0)};
        return {status[point] != 0};
      });
  return amrex::get<0>(reduce_data.value());
}

void LogicalSurfaceStorage::reject_invalid_relaxation_steps_on_device() {
  auto *const descriptors = descriptors_.dataPtr();
  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const status = relaxation_status_.dataPtr();
  const amrex::Long count = local_point_count_;
  amrex::ParallelFor(
      count, [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        if (!descriptors[surface_ordinal[point]].due)
          return;
        if (status[point] != 0)
          amrex::Gpu::Atomic::Exch(
              &descriptors[surface_ordinal[point]].relaxation_state,
              static_cast<int>(SurfaceRelaxationState::failed));
      });

  auto *const height = height_.dataPtr();
  auto *const velocity = relaxation_velocity_.dataPtr();
  const auto *const committed = committed_state_.dataPtr();
  amrex::ParallelFor(
      count, [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &surface = descriptors[surface_ordinal[point]];
        if (!surface.due)
          return;
        if (surface.relaxation_state ==
            static_cast<int>(SurfaceRelaxationState::failed)) {
          height[point] = committed[point];
          velocity[point] = committed[count + point];
        }
      });
}

amrex::Long LogicalSurfaceStorage::reject_invalid_relaxation_steps() {
  const amrex::Long invalid = local_relaxation_invalid();
  if (invalid == 0)
    return 0;

  reject_invalid_relaxation_steps_on_device();
  return invalid;
}

amrex::Real LogicalSurfaceStorage::local_minimum_angular_spacing() {
  amrex::ReduceOps<amrex::ReduceOpMin> reduce_ops;
  amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  const auto view = device_view();
  reduce_ops.eval(
      local_point_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept -> ReduceTuple {
        const auto &surface =
            view.descriptors[view.surface_ordinal[point]];
        if (surface.relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return {std::numeric_limits<amrex::Real>::max()};
        const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                                  (view.itheta[point] + amrex::Real(0.5)) /
                                  surface.ntheta;
        const amrex::Real dtheta =
            amrex::Math::pi<amrex::Real>() / surface.ntheta;
        const amrex::Real dphi =
            2 * amrex::Math::pi<amrex::Real>() / surface.nphi;
        const amrex::Real sin_theta = amrex::Math::sincos(theta).first;
        return {view.height[point] *
                amrex::min(dtheta, sin_theta * dphi)};
      });
  return amrex::get<0>(reduce_data.value());
}

void LogicalSurfaceStorage::rebuild_positions() {
  const auto view = device_view();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &surface =
            view.descriptors[view.surface_ordinal[point]];
        if (!surface.due)
          return;
        const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                                  (view.itheta[point] + amrex::Real(0.5)) /
                                  surface.ntheta;
        const amrex::Real phi = 2 * amrex::Math::pi<amrex::Real>() *
                                (view.iphi[point] + amrex::Real(0.5)) /
                                surface.nphi;
        const AngularGeometry basis = angular_geometry(theta, phi);
        view.position[0][point] =
            surface.center[0] + view.height[point] * basis.radial[0];
        view.position[1][point] =
            surface.center[1] + view.height[point] * basis.radial[1];
        view.position[2][point] = surface.center[2] +
                                  view.height[point] * basis.radial[2];
      });
}

} // namespace ParticleAHFinderX
