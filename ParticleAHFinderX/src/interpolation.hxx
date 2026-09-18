/**
 * \file interpolation.hxx
 * \brief Portable device interpolation for CarpetX mesh fields.
 *
 * The checked Array4 interface is device-callable and rejects unavailable
 * stencils explicitly. The tensor-product cubic value and analytic-gradient
 * implementation is local to this thorn.
 */
#ifndef PARTICLEAHFINDERX_INTERPOLATION_HXX
#define PARTICLEAHFINDERX_INTERPOLATION_HXX

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

namespace ParticleAHFinderX {

struct FieldCentering {
  // CarpetX centering is converted at the adapter boundary. Here one denotes
  // a nodal direction and zero denotes a cell-centered direction.
  amrex::GpuArray<int, 3> nodal{{0, 0, 0}};
};

enum class InterpolationStatus : int {
  success = 0,
  stencil_unavailable = 1,
  nonfinite = 2
};

struct AdmInterpolationData {
  amrex::GpuArray<amrex::Real, 6> gamma;
  amrex::GpuArray<amrex::GpuArray<amrex::Real, 6>, 3> d_gamma;
  amrex::GpuArray<amrex::Real, 6> curv;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void interpolation_coordinates(
    const amrex::GpuArray<amrex::Real, 3> &position,
    const amrex::GpuArray<amrex::Real, 3> &prob_lo,
    const amrex::GpuArray<amrex::Real, 3> &inv_dx,
    const FieldCentering centering, amrex::GpuArray<int, 3> &cell,
    amrex::GpuArray<amrex::Real, 3> &fraction) noexcept {
  for (int d = 0; d < 3; ++d) {
    const amrex::Real offset = centering.nodal[d] ? amrex::Real(0)
                                                  : amrex::Real(0.5);
    const amrex::Real coordinate =
        (position[d] - prob_lo[d]) * inv_dx[d] - offset;
    cell[d] = static_cast<int>(amrex::Math::floor(coordinate));
    fraction[d] = coordinate - cell[d];
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool stencil_is_available(
    const amrex::GpuArray<int, 3> &lower,
    const amrex::GpuArray<int, 3> &upper,
    const amrex::Box &available) noexcept {
  const auto box_lower = amrex::lbound(available);
  const auto box_upper = amrex::ubound(available);
  return lower[0] >= box_lower.x && lower[1] >= box_lower.y &&
         lower[2] >= box_lower.z && upper[0] <= box_upper.x &&
         upper[1] <= box_upper.y && upper[2] <= box_upper.z;
}

/** Trilinearly gather one scalar component without clamping its stencil. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool trilinear_interpolate(
    const amrex::Array4<const amrex::Real> &field, const int component,
    const amrex::GpuArray<amrex::Real, 3> &position,
    const amrex::GpuArray<amrex::Real, 3> &prob_lo,
    const amrex::GpuArray<amrex::Real, 3> &inv_dx,
    const FieldCentering centering, const amrex::Box &available,
    amrex::Real &value) noexcept {
  amrex::GpuArray<int, 3> cell;
  amrex::GpuArray<amrex::Real, 3> fraction;
  interpolation_coordinates(position, prob_lo, inv_dx, centering, cell,
                            fraction);

  const amrex::GpuArray<int, 3> upper{{cell[0] + 1, cell[1] + 1,
                                       cell[2] + 1}};
  if (!stencil_is_available(cell, upper, available))
    return false;

  value = 0;
  for (int dk = 0; dk <= 1; ++dk) {
    const amrex::Real weight_z =
        dk ? fraction[2] : amrex::Real(1) - fraction[2];
    for (int dj = 0; dj <= 1; ++dj) {
      const amrex::Real weight_y =
          dj ? fraction[1] : amrex::Real(1) - fraction[1];
      for (int di = 0; di <= 1; ++di) {
        const amrex::Real weight_x =
            di ? fraction[0] : amrex::Real(1) - fraction[0];
        value += weight_x * weight_y * weight_z *
                 field(cell[0] + di, cell[1] + dj, cell[2] + dk,
                       component);
      }
    }
  }
  return amrex::Math::isfinite(value);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cubic_lagrange_weights(const amrex::Real fraction,
                       amrex::GpuArray<amrex::Real, 4> &weight,
                       amrex::GpuArray<amrex::Real, 4> &derivative) noexcept {
  const amrex::Real fraction2 = fraction * fraction;
  const amrex::Real fraction3 = fraction2 * fraction;
  weight[0] = -fraction3 / 6 + fraction2 / 2 - fraction / 3;
  weight[1] = fraction3 / 2 - fraction2 - fraction / 2 + 1;
  weight[2] = -fraction3 / 2 + fraction2 / 2 + fraction;
  weight[3] = fraction3 / 6 - fraction / 6;

  derivative[0] = -fraction2 / 2 + fraction - amrex::Real(1) / 3;
  derivative[1] = 3 * fraction2 / 2 - 2 * fraction - amrex::Real(1) / 2;
  derivative[2] = -3 * fraction2 / 2 + fraction + 1;
  derivative[3] = fraction2 / 2 - amrex::Real(1) / 6;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
adm_interpolation_is_finite(const AdmInterpolationData &data) noexcept {
  for (int component = 0; component < 6; ++component) {
    if (!amrex::Math::isfinite(data.gamma[component]) ||
        !amrex::Math::isfinite(data.curv[component]))
      return false;
    for (int d = 0; d < 3; ++d)
      if (!amrex::Math::isfinite(data.d_gamma[d][component]))
        return false;
  }
  return true;
}

/**
 * Fused trilinear ADM gather with analytic derivatives of the trilinear
 * metric polynomial.
 *
 * Metric and curvature are sampled directly at their common native
 * centering. No vertex-to-cell averaging is performed.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE InterpolationStatus
trilinear_interpolate_adm(
    const amrex::Array4<const amrex::Real> &metric,
    const int metric_component,
    const amrex::Array4<const amrex::Real> &curvature,
    const int curvature_component,
    const amrex::GpuArray<amrex::Real, 3> &position,
    const amrex::GpuArray<amrex::Real, 3> &prob_lo,
    const amrex::GpuArray<amrex::Real, 3> &inv_dx,
    const FieldCentering centering, const amrex::Box &metric_available,
    const amrex::Box &curvature_available,
    AdmInterpolationData &data) noexcept {
  amrex::GpuArray<int, 3> cell;
  amrex::GpuArray<amrex::Real, 3> fraction;
  interpolation_coordinates(position, prob_lo, inv_dx, centering, cell,
                            fraction);

  const amrex::GpuArray<int, 3> upper{{cell[0] + 1, cell[1] + 1,
                                       cell[2] + 1}};
  if (!stencil_is_available(cell, upper, metric_available) ||
      !stencil_is_available(cell, upper, curvature_available))
    return InterpolationStatus::stencil_unavailable;

  for (int component = 0; component < 6; ++component) {
    data.gamma[component] = 0;
    data.curv[component] = 0;
    for (int d = 0; d < 3; ++d)
      data.d_gamma[d][component] = 0;
  }

  for (int dk = 0; dk <= 1; ++dk) {
    const amrex::Real weight_z =
        dk ? fraction[2] : amrex::Real(1) - fraction[2];
    const amrex::Real derivative_z = dk ? inv_dx[2] : -inv_dx[2];
    for (int dj = 0; dj <= 1; ++dj) {
      const amrex::Real weight_y =
          dj ? fraction[1] : amrex::Real(1) - fraction[1];
      const amrex::Real derivative_y = dj ? inv_dx[1] : -inv_dx[1];
      for (int di = 0; di <= 1; ++di) {
        const amrex::Real weight_x =
            di ? fraction[0] : amrex::Real(1) - fraction[0];
        const amrex::Real derivative_x = di ? inv_dx[0] : -inv_dx[0];
        const amrex::Real weight = weight_x * weight_y * weight_z;
        const int i = cell[0] + di;
        const int j = cell[1] + dj;
        const int k = cell[2] + dk;
        for (int component = 0; component < 6; ++component) {
          const amrex::Real gamma =
              metric(i, j, k, metric_component + component);
          data.gamma[component] += weight * gamma;
          data.d_gamma[0][component] +=
              derivative_x * weight_y * weight_z * gamma;
          data.d_gamma[1][component] +=
              weight_x * derivative_y * weight_z * gamma;
          data.d_gamma[2][component] +=
              weight_x * weight_y * derivative_z * gamma;
          data.curv[component] +=
              weight * curvature(i, j, k, curvature_component + component);
        }
      }
    }
  }

  return adm_interpolation_is_finite(data)
             ? InterpolationStatus::success
             : InterpolationStatus::nonfinite;
}

/**
 * Fused four-point tensor-product ADM gather.
 *
 * The coordinates and cubic weights are formed once per particle and reused
 * for all six metric values, all eighteen metric derivatives, and all six
 * curvature values. Metric and curvature must share their native centering;
 * the checked CarpetX adapter enforces that contract before this device
 * routine is called.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE InterpolationStatus
tensor_cubic_interpolate_adm(
    const amrex::Array4<const amrex::Real> &metric,
    const int metric_component,
    const amrex::Array4<const amrex::Real> &curvature,
    const int curvature_component,
    const amrex::GpuArray<amrex::Real, 3> &position,
    const amrex::GpuArray<amrex::Real, 3> &prob_lo,
    const amrex::GpuArray<amrex::Real, 3> &inv_dx,
    const FieldCentering centering, const amrex::Box &metric_available,
    const amrex::Box &curvature_available,
    AdmInterpolationData &data) noexcept {
  amrex::GpuArray<int, 3> cell;
  amrex::GpuArray<amrex::Real, 3> fraction;
  interpolation_coordinates(position, prob_lo, inv_dx, centering, cell,
                            fraction);

  const amrex::GpuArray<int, 3> lower{{cell[0] - 1, cell[1] - 1,
                                       cell[2] - 1}};
  const amrex::GpuArray<int, 3> upper{{cell[0] + 2, cell[1] + 2,
                                       cell[2] + 2}};
  if (!stencil_is_available(lower, upper, metric_available) ||
      !stencil_is_available(lower, upper, curvature_available))
    return InterpolationStatus::stencil_unavailable;

  amrex::GpuArray<amrex::Real, 4> weight[3];
  amrex::GpuArray<amrex::Real, 4> derivative[3];
  for (int d = 0; d < 3; ++d)
    cubic_lagrange_weights(fraction[d], weight[d], derivative[d]);

  for (int component = 0; component < 6; ++component) {
    data.gamma[component] = 0;
    data.curv[component] = 0;
    for (int d = 0; d < 3; ++d)
      data.d_gamma[d][component] = 0;
  }

  for (int dk = 0; dk < 4; ++dk)
    for (int dj = 0; dj < 4; ++dj)
      for (int di = 0; di < 4; ++di) {
        const amrex::Real value_weight =
            weight[0][di] * weight[1][dj] * weight[2][dk];
        const int i = lower[0] + di;
        const int j = lower[1] + dj;
        const int k = lower[2] + dk;
        for (int component = 0; component < 6; ++component) {
          const amrex::Real gamma =
              metric(i, j, k, metric_component + component);
          data.gamma[component] += value_weight * gamma;
          data.d_gamma[0][component] +=
              derivative[0][di] * weight[1][dj] * weight[2][dk] * gamma;
          data.d_gamma[1][component] +=
              weight[0][di] * derivative[1][dj] * weight[2][dk] * gamma;
          data.d_gamma[2][component] +=
              weight[0][di] * weight[1][dj] * derivative[2][dk] * gamma;
          data.curv[component] +=
              value_weight *
              curvature(i, j, k, curvature_component + component);
        }
      }
  for (int d = 0; d < 3; ++d)
    for (int component = 0; component < 6; ++component)
      data.d_gamma[d][component] *= inv_dx[d];

  return adm_interpolation_is_finite(data)
             ? InterpolationStatus::success
             : InterpolationStatus::nonfinite;
}

/**
 * Gather a scalar with four-point tensor-product Lagrange interpolation.
 *
 * `gradient` is the analytic derivative of the interpolation polynomial in
 * physical Cartesian coordinates. The complete 4x4x4 stencil must lie in
 * `available`; a missing stencil is reported to the caller and never clamped.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool tensor_cubic_interpolate(
    const amrex::Array4<const amrex::Real> &field, const int component,
    const amrex::GpuArray<amrex::Real, 3> &position,
    const amrex::GpuArray<amrex::Real, 3> &prob_lo,
    const amrex::GpuArray<amrex::Real, 3> &inv_dx,
    const FieldCentering centering, const amrex::Box &available,
    amrex::Real &value,
    amrex::GpuArray<amrex::Real, 3> &gradient) noexcept {
  amrex::GpuArray<int, 3> cell;
  amrex::GpuArray<amrex::Real, 3> fraction;
  interpolation_coordinates(position, prob_lo, inv_dx, centering, cell,
                            fraction);

  const amrex::GpuArray<int, 3> lower{{cell[0] - 1, cell[1] - 1,
                                       cell[2] - 1}};
  const amrex::GpuArray<int, 3> upper{{cell[0] + 2, cell[1] + 2,
                                       cell[2] + 2}};
  if (!stencil_is_available(lower, upper, available))
    return false;

  amrex::GpuArray<amrex::Real, 4> weight[3];
  amrex::GpuArray<amrex::Real, 4> derivative[3];
  for (int d = 0; d < 3; ++d)
    cubic_lagrange_weights(fraction[d], weight[d], derivative[d]);

  value = 0;
  gradient = {{0, 0, 0}};
  for (int dk = 0; dk < 4; ++dk)
    for (int dj = 0; dj < 4; ++dj)
      for (int di = 0; di < 4; ++di) {
        const amrex::Real sample =
            field(lower[0] + di, lower[1] + dj, lower[2] + dk, component);
        value += weight[0][di] * weight[1][dj] * weight[2][dk] * sample;
        gradient[0] += derivative[0][di] * weight[1][dj] * weight[2][dk] *
                       sample;
        gradient[1] += weight[0][di] * derivative[1][dj] * weight[2][dk] *
                       sample;
        gradient[2] += weight[0][di] * weight[1][dj] * derivative[2][dk] *
                       sample;
      }
  for (int d = 0; d < 3; ++d)
    gradient[d] *= inv_dx[d];
  return amrex::Math::isfinite(value) &&
         amrex::Math::isfinite(gradient[0]) &&
         amrex::Math::isfinite(gradient[1]) &&
         amrex::Math::isfinite(gradient[2]);
}

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_INTERPOLATION_HXX
