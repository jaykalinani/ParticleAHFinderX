/**
 * \file logical_surface_storage.cxx
 * \brief Batched initialization of persistent logical surfaces on owner GPUs.
 */
#include "logical_surface_storage.hxx"
#include "angular_geometry.hxx"
#include "interpolation.hxx"

#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_Math.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <limits>

namespace ParticleAHFinderX {
namespace {

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
  surface_extrapolation_displacement_.resize(local_surface_count_);
  surface_extrapolation_weight_.resize(local_surface_count_);
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
  surface_time_step_.resize(local_surface_count_);
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
  surface_extrapolation_displacement_.resize(local_surface_count_);
  surface_extrapolation_weight_.resize(local_surface_count_);
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
  surface_time_step_.resize(local_surface_count_);
  source_rank_.resize(count);
  source_index_.resize(count);

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
  surface_time_step_.swap(replacement.surface_time_step_);
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

void LogicalSurfaceStorage::prolong_surface_layers(
    const std::vector<SurfaceLayerSelection> &source_layers) {
  if (source_layers.empty())
    return;
  if (source_layers.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    amrex::Abort("ParticleAHFinderX prolongation table exceeds the portable "
                 "device-kernel index range");
  amrex::Gpu::DeviceVector<SurfaceLayerSelection> device_sources(
      source_layers.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, source_layers.begin(),
                   source_layers.end(), device_sources.begin());
  const auto *const sources = device_sources.dataPtr();
  const int source_count = static_cast<int>(source_layers.size());
  const auto view = device_view();
  amrex::Gpu::DeviceVector<int> device_error(1, 0);
  int *const error = device_error.dataPtr();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int destination_ordinal = view.surface_ordinal[point];
        const auto &destination = view.descriptors[destination_ordinal];
        bool selected = false;
        for (int i = 0; i < source_count; ++i)
          selected |=
              destination.stable_id == sources[i].stable_id &&
              destination.angular_level == sources[i].angular_level + 1;
        if (!selected)
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
        amrex::Real prolonged_height = 0;
        for (int j = 0; j < 4; ++j)
          for (int i = 0; i < 4; ++i)
            prolonged_height +=
                theta_weight[i] * phi_weight[j] *
                angular_scalar(view.height, source.offset, source.ntheta,
                               source.nphi, theta_cell - 1 + i,
                               phi_cell - 1 + j);
        if (!(prolonged_height > 0) ||
            !amrex::Math::isfinite(prolonged_height)) {
          amrex::Gpu::Atomic::Exch(error, 1);
          return;
        }
        view.height[point] = prolonged_height;
        view.relaxation_velocity[point] = 0;
        view.expansion[point] = 0;
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

void LogicalSurfaceStorage::evaluate_angular_derivatives() {
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
      });
}

void LogicalSurfaceStorage::evaluate_expansion() {
  const auto view = device_view();
  amrex::ParallelFor(
      local_point_count_,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &surface =
            view.descriptors[view.surface_ordinal[point]];
        if (!surface.due)
          return;
        ExpansionInput input;
        for (int component = 0; component < 6; ++component) {
          input.gamma[component] = view.gamma(component, point);
          input.curv[component] = view.curv(component, point);
          for (int d = 0; d < 3; ++d)
            input.d_gamma[d][component] =
                view.d_gamma(d, component, point);
        }
        input.height = view.height[point];
        input.h_theta = view.angular_component(0, point);
        input.h_phi = view.angular_component(1, point);
        input.h_theta_theta = view.angular_component(2, point);
        input.h_theta_phi = view.angular_component(3, point);
        input.h_phi_phi = view.angular_component(4, point);
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

ExpansionStats LogicalSurfaceStorage::local_expansion_stats() const {
  amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                   amrex::ReduceOpMax, amrex::ReduceOpSum>
      reduce_ops;
  amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Long>
      reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  const amrex::Real *const expansion = expansion_.dataPtr();
  const amrex::Real *const area_weight = area_weight_.dataPtr();
  const int *const geometry_status = geometry_status_.dataPtr();
  const int *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const descriptors = descriptors_.dataPtr();
  reduce_ops.eval(
      local_point_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept -> ReduceTuple {
        if (!descriptors[surface_ordinal[point]].due)
          return {0, 0, 0, 0};
        if (geometry_status[point] != 0)
          return {0, 0, 0, 1};
        const amrex::Real theta = expansion[point];
        const amrex::Real absolute_theta = theta < 0 ? -theta : theta;
        return {area_weight[point], theta * theta * area_weight[point],
                absolute_theta, 0};
      });
  const auto result = reduce_data.value();
  return {amrex::get<0>(result), amrex::get<1>(result),
          amrex::get<2>(result), amrex::get<3>(result)};
}

SolverConvergenceStats LogicalSurfaceStorage::update_solver_convergence(
    const bool accept_convergence,
    const amrex::Real cutoff_indicator_threshold) {
  if (local_surface_count_ == 0)
    return {};
  auto *const surface_area = surface_area_.dataPtr();
  auto *const surface_theta_squared = surface_theta_squared_.dataPtr();
  auto *const surface_theta_integral = surface_theta_integral_.dataPtr();
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

  amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                   amrex::ReduceOpMax, amrex::ReduceOpMax,
                   amrex::ReduceOpSum>
      reduce_ops;
  amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                    amrex::Long>
      reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  const auto *const descriptors = descriptors_.dataPtr();
  auto *const mutable_descriptors = descriptors_.dataPtr();
  amrex::ParallelFor(
      surface_count,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        auto &descriptor = mutable_descriptors[surface];
        if (descriptor.relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return;
        if (!(surface_area[surface] > 0) ||
            surface_invalid[surface] != 0) {
          descriptor.relaxation_state =
              static_cast<int>(SurfaceRelaxationState::failed);
          return;
        }
        if (accept_convergence && cutoff_indicator_threshold > 0 &&
            descriptor.angular_level + 1 < descriptor.angular_levels &&
            surface_cutoff_indicator[surface] >
                cutoff_indicator_threshold) {
          descriptor.relaxation_state =
              static_cast<int>(SurfaceRelaxationState::refine);
          return;
        }
        const amrex::Real mean_squared =
            surface_theta_squared[surface] / surface_area[surface];
        const amrex::Real l2 =
            mean_squared > 0
                ? mean_squared * amrex::Math::rsqrt(mean_squared)
                : 0;
        const amrex::Real ratio = amrex::max(
            l2 * descriptor.mass_scale / descriptor.theta_l2_tolerance,
            surface_theta_linf[surface] * descriptor.mass_scale /
                descriptor.theta_linf_tolerance);
        if (accept_convergence && ratio <= 1)
          descriptor.relaxation_state =
              static_cast<int>(SurfaceRelaxationState::converged);
      });

  reduce_ops.eval(
      surface_count, reduce_data,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept -> ReduceTuple {
        const auto &descriptor = descriptors[surface];
        if (!descriptor.due)
          return {0, 0, 0, 0, 0};
        if (!(surface_area[surface] > 0) || surface_invalid[surface] != 0)
          return {0, 0, 0, 0, surface_invalid[surface] + 1};
        const amrex::Real mean_squared =
            surface_theta_squared[surface] / surface_area[surface];
        const amrex::Real l2 =
            mean_squared > 0
                ? mean_squared * amrex::Math::rsqrt(mean_squared)
                : 0;
        const amrex::Real l2_times_mass = l2 * descriptor.mass_scale;
        const amrex::Real linf_times_mass =
            surface_theta_linf[surface] * descriptor.mass_scale;
        const amrex::Real ratio = amrex::max(
            l2_times_mass / descriptor.theta_l2_tolerance,
            linf_times_mass / descriptor.theta_linf_tolerance);
        return {ratio, l2_times_mass, linf_times_mass,
                surface_cutoff_indicator[surface],
                surface_invalid[surface]};
      });
  const auto result = reduce_data.value();
  amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                   amrex::ReduceOpSum, amrex::ReduceOpSum>
      state_reduce_ops;
  amrex::ReduceData<amrex::Long, amrex::Long, amrex::Long, amrex::Long>
      state_reduce_data(state_reduce_ops);
  using StateReduceTuple = typename decltype(state_reduce_data)::Type;
  state_reduce_ops.eval(
      surface_count, state_reduce_data,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept -> StateReduceTuple {
        const int state = descriptors[surface].relaxation_state;
        return {
            state == static_cast<int>(SurfaceRelaxationState::active),
            state == static_cast<int>(SurfaceRelaxationState::converged),
            state == static_cast<int>(SurfaceRelaxationState::refine),
            state == static_cast<int>(SurfaceRelaxationState::failed),
        };
      });
  const auto state_result = state_reduce_data.value();
  return {amrex::get<0>(result), amrex::get<1>(result),
          amrex::get<2>(result), amrex::get<3>(result),
          amrex::get<4>(result), amrex::get<0>(state_result),
          amrex::get<1>(state_result), amrex::get<2>(state_result),
          amrex::get<3>(state_result)};
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
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        mutable_descriptors[surface].relaxation_state =
            mutable_descriptors[surface].due
                ? static_cast<int>(SurfaceRelaxationState::active)
                : static_cast<int>(SurfaceRelaxationState::idle);
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

void LogicalSurfaceStorage::begin_relaxation_step() {
  const amrex::Long count = local_point_count_;
  const auto *const height = height_.dataPtr();
  const auto *const velocity = relaxation_velocity_.dataPtr();
  auto *const base = relaxation_base_.dataPtr();
  auto *const status = relaxation_status_.dataPtr();
  amrex::ParallelFor(count,
                     [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
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
  auto *const mean_displacement =
      surface_extrapolation_displacement_.dataPtr();
  auto *const surface_weight = surface_extrapolation_weight_.dataPtr();
  const amrex::Long count = local_point_count_;
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        mean_displacement[surface] = 0;
        surface_weight[surface] = 0;
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
          const int itheta = static_cast<int>(
              (point - surface.offset) / surface.nphi);
          const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                                    (itheta + amrex::Real(0.5)) /
                                    surface.ntheta;
          const amrex::Real weight =
              amrex::Math::sincos(theta).first;
          amrex::Gpu::Atomic::AddNoRet(
              mean_displacement + ordinal,
              weight * (height[point] - reference[point]));
          amrex::Gpu::Atomic::AddNoRet(surface_weight + ordinal, weight);
        }
      });
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        if (surface_weight[surface] > 0)
          mean_displacement[surface] /= surface_weight[surface];
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
         control.mode != RelaxationExtrapolationMode::mean_height) ||
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
  const auto *const mean_displacement =
      surface_extrapolation_displacement_.dataPtr();
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
        const amrex::Real candidate =
            trial[control].mode == RelaxationExtrapolationMode::mean_height
                ? relaxation_extrapolated_mean_height(
                      base[point], mean_displacement[ordinal],
                      trial[control].overstep)
                : relaxation_extrapolated_height(
                      reference[point], base[point],
                      trial[control].overstep);
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
        height[point] =
            trial[control].mode == RelaxationExtrapolationMode::mean_height
                ? relaxation_extrapolated_mean_height(
                      base[point], mean_displacement[ordinal], overstep)
                : relaxation_extrapolated_height(
                      reference[point], base[point], overstep);
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

amrex::Real LogicalSurfaceStorage::prepare_relaxation_time_steps(
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
            height[point] * amrex::min(dtheta, sin_theta * dphi);
        amrex::Gpu::Atomic::Min(time_step + ordinal, spacing);
      });
  amrex::ParallelFor(
      local_surface_count_,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept {
        if (descriptors[surface].relaxation_state ==
            static_cast<int>(SurfaceRelaxationState::active))
          time_step[surface] *= cfl_factor;
      });

  amrex::ReduceOps<amrex::ReduceOpMin> reduce_ops;
  amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  reduce_ops.eval(
      local_surface_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const int surface) noexcept -> ReduceTuple {
        const amrex::Real dt = time_step[surface];
        if (descriptors[surface].relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return {inactive_step};
        if (!(dt > 0) || !amrex::Math::isfinite(dt))
          return {amrex::Real(0)};
        return {dt};
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
  amrex::ParallelFor(
      count, [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const auto &surface = descriptors[surface_ordinal[point]];
        if (surface.relaxation_state !=
            static_cast<int>(SurfaceRelaxationState::active))
          return;
        if (geometry_status[point] != 0 || relaxation_status[point] != 0) {
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
      });

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
        const amrex::Real dt = time_step[surface_ordinal[point]];
        const amrex::Real next_height = relaxation_stage_value(
            integrator, stage, base[point], dt, rhs[point],
            rhs[2 * count + point], rhs[4 * count + point], rhs_height);
        const amrex::Real next_velocity = relaxation_stage_value(
            integrator, stage, base[count + point], dt,
            rhs[count + point], rhs[3 * count + point],
            rhs[5 * count + point], rhs_velocity);
        const amrex::Real minimum_radius =
            minimum_radius_factor * surface.radius;
        const amrex::Real maximum_radius =
            maximum_radius_factor * surface.radius;
        if (!amrex::Math::isfinite(next_height) ||
            !amrex::Math::isfinite(next_velocity) ||
            !(next_height >= minimum_radius) ||
            !(next_height <= maximum_radius)) {
          relaxation_status[point] = 1;
          return;
        }
        height[point] = next_height;
        velocity[point] = next_velocity;
      });
}

amrex::Long LogicalSurfaceStorage::local_relaxation_invalid() const {
  amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
  amrex::ReduceData<amrex::Long> reduce_data(reduce_ops);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  const int *const status = relaxation_status_.dataPtr();
  reduce_ops.eval(
      local_point_count_, reduce_data,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept -> ReduceTuple {
        return {status[point] != 0};
      });
  return amrex::get<0>(reduce_data.value());
}

amrex::Long LogicalSurfaceStorage::reject_invalid_relaxation_steps() {
  const amrex::Long invalid = local_relaxation_invalid();
  if (invalid == 0)
    return 0;

  auto *const descriptors = descriptors_.dataPtr();
  const auto *const surface_ordinal = surface_ordinal_.dataPtr();
  const auto *const status = relaxation_status_.dataPtr();
  const amrex::Long count = local_point_count_;
  amrex::ParallelFor(
      count, [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
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
        if (descriptors[surface_ordinal[point]].relaxation_state ==
            static_cast<int>(SurfaceRelaxationState::failed)) {
          height[point] = committed[point];
          velocity[point] = committed[count + point];
        }
      });
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
