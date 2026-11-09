/**
 * \file intrinsic_geometry.hxx
 * \brief Polar-regular intrinsic geometry for one logical horizon surface.
 */
#ifndef PARTICLEAHFINDERX_INTRINSIC_GEOMETRY_HXX
#define PARTICLEAHFINDERX_INTRINSIC_GEOMETRY_HXX

#include "angular_derivatives.hxx"
#include "angular_geometry.hxx"
#include "expansion.hxx"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

namespace ParticleAHFinderX {

constexpr int intrinsic_metric_components = 3;
constexpr int intrinsic_connection_components = 6;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
intrinsic_metric_component(int first, int second) noexcept {
  if (first > second) {
    const int temporary = first;
    first = second;
    second = temporary;
  }
  return first == 0 ? second : 2;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
intrinsic_connection_component(const int upper, const int first,
                               const int second) noexcept {
  return 3 * upper + intrinsic_metric_component(first, second);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
intrinsic_index_sign(const int index) noexcept {
  return index == 0 ? -1 : 1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
intrinsic_metric_parity(const int first, const int second) noexcept {
  return intrinsic_index_sign(first) * intrinsic_index_sign(second);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
intrinsic_connection_parity(const int upper, const int first,
                            const int second) noexcept {
  return intrinsic_index_sign(upper) * intrinsic_index_sign(first) *
         intrinsic_index_sign(second);
}

struct IntrinsicGeometryInput {
  amrex::Real gamma[6];
  amrex::Real curv[6];
  amrex::Real normal[3];
  amrex::Real height = 0;
  amrex::Real h_theta = 0;
  amrex::Real h_phi = 0;
  amrex::Real theta = 0;
  amrex::Real phi = 0;
};

struct IntrinsicGeometryOutput {
  amrex::Real tangent[2][3]{{0, 0, 0}, {0, 0, 0}};
  amrex::Real metric[3]{0, 0, 0};
  amrex::Real inverse[3]{0, 0, 0};
  amrex::Real sqrt_determinant = 0;
  amrex::Real normal_connection[2]{0, 0};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
inverse_intrinsic_metric(const amrex::Real metric[3],
                         amrex::Real inverse[3],
                         amrex::Real &determinant) noexcept {
  determinant = metric[0] * metric[2] - metric[1] * metric[1];
  if (!(determinant > 0) || !amrex::Math::isfinite(determinant))
    return false;
  const amrex::Real inverse_determinant = 1 / determinant;
  inverse[0] = metric[2] * inverse_determinant;
  inverse[1] = -metric[1] * inverse_determinant;
  inverse[2] = metric[0] * inverse_determinant;
  return amrex::Math::isfinite(inverse[0]) &&
         amrex::Math::isfinite(inverse[1]) &&
         amrex::Math::isfinite(inverse[2]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
intrinsic_symmetric_value(const amrex::Real values[3], const int first,
                          const int second) noexcept {
  return values[intrinsic_metric_component(first, second)];
}

/** Form q_AB and omega_A=-e_A^i K_ij s^j at one surface point. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
evaluate_intrinsic_geometry(const IntrinsicGeometryInput &input,
                            IntrinsicGeometryOutput &output) noexcept {
  if (!(input.height > 0) || !amrex::Math::isfinite(input.height))
    return false;
  const AngularGeometry basis = angular_geometry(input.theta, input.phi);
  for (int d = 0; d < 3; ++d) {
    output.tangent[0][d] = input.h_theta * basis.radial[d] +
                           input.height * basis.radial_theta[d];
    output.tangent[1][d] = input.h_phi * basis.radial[d] +
                           input.height * basis.radial_phi[d];
  }
  for (int first = 0; first < 2; ++first)
    for (int second = first; second < 2; ++second) {
      amrex::Real value = 0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          value += output.tangent[first][i] *
                   symmetric_value(input.gamma, i, j) *
                   output.tangent[second][j];
      output.metric[intrinsic_metric_component(first, second)] = value;
    }
  amrex::Real determinant = 0;
  if (!inverse_intrinsic_metric(output.metric, output.inverse, determinant))
    return false;
  output.sqrt_determinant =
      determinant * amrex::Math::rsqrt(determinant);

  for (int angular = 0; angular < 2; ++angular) {
    amrex::Real value = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        value -= output.tangent[angular][i] *
                 symmetric_value(input.curv, i, j) * input.normal[j];
    output.normal_connection[angular] = value;
  }
  for (int component = 0; component < 3; ++component)
    if (!amrex::Math::isfinite(output.metric[component]) ||
        !amrex::Math::isfinite(output.inverse[component]))
      return false;
  return amrex::Math::isfinite(output.sqrt_determinant) &&
         amrex::Math::isfinite(output.normal_connection[0]) &&
         amrex::Math::isfinite(output.normal_connection[1]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
unit_sphere_connection(const amrex::Real theta,
                       amrex::Real connection[2][2][2]) noexcept {
  for (int upper = 0; upper < 2; ++upper)
    for (int first = 0; first < 2; ++first)
      for (int second = 0; second < 2; ++second)
        connection[upper][first][second] = 0;
  const auto sincos = amrex::Math::sincos(theta);
  const amrex::Real sine = sincos.first;
  const amrex::Real cosine = sincos.second;
  connection[0][1][1] = -sine * cosine;
  connection[1][0][1] = cosine / sine;
  connection[1][1][0] = cosine / sine;
}

/**
 * Form Delta^A_BC=Gamma(q)^A_BC-Gamma(unit sphere)^A_BC.
 *
 * Delta is a true tensor. Differentiating it, rather than the individually
 * singular latitude/longitude Christoffel symbols, is the polar-regular
 * reference-metric formulation used by the scalar-curvature operator below.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
intrinsic_connection_difference(
    const amrex::Real *const metric, const amrex::Long point_count,
    const amrex::Long surface_offset, const int ntheta, const int nphi,
    const int itheta, const int iphi, const amrex::Real theta,
    amrex::Real difference[2][2][2]) noexcept {
  amrex::Real local_metric[3];
  amrex::Real inverse[3];
  amrex::Real determinant = 0;
  amrex::Real derivative[2][3];
  for (int component = 0; component < 3; ++component) {
    const amrex::Real *const field = metric + component * point_count;
    local_metric[component] =
        field[surface_offset + amrex::Long(itheta) * nphi + iphi];
    const int first = component == 2 ? 1 : 0;
    const int second = component == 0 ? 0 : 1;
    const AngularFirstDerivatives angular = angular_first_derivatives(
        field, surface_offset, ntheta, nphi, itheta, iphi,
        intrinsic_metric_parity(first, second));
    derivative[0][component] = angular.theta;
    derivative[1][component] = angular.phi;
  }
  if (!inverse_intrinsic_metric(local_metric, inverse, determinant))
    return false;
  amrex::Real reference[2][2][2];
  unit_sphere_connection(theta, reference);
  for (int upper = 0; upper < 2; ++upper)
    for (int first = 0; first < 2; ++first)
      for (int second = 0; second < 2; ++second) {
        amrex::Real value = 0;
        for (int contracted = 0; contracted < 2; ++contracted)
          value += amrex::Real(0.5) *
                   intrinsic_symmetric_value(inverse, upper, contracted) *
                   (derivative[first][intrinsic_metric_component(
                        contracted, second)] +
                    derivative[second][intrinsic_metric_component(
                        contracted, first)] -
                    derivative[contracted][intrinsic_metric_component(
                        first, second)]);
        difference[upper][first][second] =
            value - reference[upper][first][second];
        if (!amrex::Math::isfinite(
                difference[upper][first][second]))
          return false;
      }
  return true;
}

/** Evaluate R(q) from the regular reference-connection difference. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
intrinsic_scalar_curvature(
    const amrex::Real *const metric,
    const amrex::Real *const connection_difference,
    const amrex::Long point_count, const amrex::Long surface_offset,
    const int ntheta, const int nphi, const int itheta, const int iphi,
    const amrex::Real theta, amrex::Real &scalar_curvature) noexcept {
  const amrex::Long point =
      surface_offset + amrex::Long(itheta) * nphi + iphi;
  amrex::Real local_metric[3];
  amrex::Real inverse[3];
  amrex::Real determinant = 0;
  for (int component = 0; component < 3; ++component)
    local_metric[component] = metric[component * point_count + point];
  if (!inverse_intrinsic_metric(local_metric, inverse, determinant))
    return false;

  amrex::Real difference[2][2][2];
  amrex::Real partial[2][2][2][2];
  for (int upper = 0; upper < 2; ++upper)
    for (int first = 0; first < 2; ++first)
      for (int second = 0; second < 2; ++second) {
        const int component =
            intrinsic_connection_component(upper, first, second);
        const amrex::Real *const field =
            connection_difference + component * point_count;
        difference[upper][first][second] = field[point];
        const AngularFirstDerivatives angular = angular_first_derivatives(
            field, surface_offset, ntheta, nphi, itheta, iphi,
            intrinsic_connection_parity(upper, first, second));
        partial[0][upper][first][second] = angular.theta;
        partial[1][upper][first][second] = angular.phi;
      }

  amrex::Real reference[2][2][2];
  unit_sphere_connection(theta, reference);
  amrex::Real covariant[2][2][2][2];
  for (int direction = 0; direction < 2; ++direction)
    for (int upper = 0; upper < 2; ++upper)
      for (int first = 0; first < 2; ++first)
        for (int second = 0; second < 2; ++second) {
          amrex::Real value = partial[direction][upper][first][second];
          for (int contracted = 0; contracted < 2; ++contracted) {
            value += reference[upper][direction][contracted] *
                     difference[contracted][first][second];
            value -= reference[contracted][direction][first] *
                     difference[upper][contracted][second];
            value -= reference[contracted][direction][second] *
                     difference[upper][first][contracted];
          }
          covariant[direction][upper][first][second] = value;
        }

  const amrex::Real sine = amrex::Math::sincos(theta).first;
  amrex::Real ricci[2][2]{{1, 0}, {0, sine * sine}};
  for (int first = 0; first < 2; ++first)
    for (int second = 0; second < 2; ++second)
      for (int contracted = 0; contracted < 2; ++contracted) {
        ricci[first][second] +=
            covariant[contracted][contracted][first][second] -
            covariant[second][contracted][first][contracted];
        for (int other = 0; other < 2; ++other)
          ricci[first][second] +=
              difference[contracted][contracted][other] *
                  difference[other][first][second] -
              difference[contracted][second][other] *
                  difference[other][first][contracted];
      }
  scalar_curvature = 0;
  for (int first = 0; first < 2; ++first)
    for (int second = 0; second < 2; ++second)
      scalar_curvature +=
          intrinsic_symmetric_value(inverse, first, second) *
          ricci[first][second];
  return amrex::Math::isfinite(scalar_curvature);
}

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_INTRINSIC_GEOMETRY_HXX
