/**
 * \file expansion.hxx
 * \brief Device evaluation of radial-graph geometry and MOTS expansion.
 *
 * The sign convention is Theta = D_i s^i + K_ij s^i s^j - K, with s^i
 * pointing toward increasing coordinate radius. The surface divergence is
 * evaluated from the second fundamental form of x(theta,phi), avoiding any
 * off-surface extension of the unit normal.
 */
#ifndef PARTICLEAHFINDERX_EXPANSION_HXX
#define PARTICLEAHFINDERX_EXPANSION_HXX

#include "angular_geometry.hxx"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

namespace ParticleAHFinderX {

struct ExpansionInput {
  amrex::Real gamma[6];
  amrex::Real d_gamma[3][6];
  amrex::Real curv[6];
  amrex::Real height;
  amrex::Real h_theta;
  amrex::Real h_phi;
  amrex::Real h_theta_theta;
  amrex::Real h_theta_phi;
  amrex::Real h_phi_phi;
  amrex::Real theta;
  amrex::Real phi;
  amrex::Real dtheta;
  amrex::Real dphi;
};

struct ExpansionOutput {
  amrex::Real theta = 0;
  amrex::Real area_weight = 0;
  amrex::Real normal[3]{0, 0, 0};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
symmetric_component(int i, int j) noexcept {
  if (i > j) {
    const int temporary = i;
    i = j;
    j = temporary;
  }
  if (i == 0)
    return j;
  if (i == 1)
    return j == 1 ? 3 : 4;
  return 5;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
symmetric_value(const amrex::Real values[6], const int i,
                const int j) noexcept {
  return values[symmetric_component(i, j)];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
metric_dot(const amrex::Real metric[6], const amrex::Real lhs[3],
           const amrex::Real rhs[3]) noexcept {
  amrex::Real result = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      result += lhs[i] * symmetric_value(metric, i, j) * rhs[j];
  return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
inverse_metric(const amrex::Real metric[6], amrex::Real inverse[3][3])
    noexcept {
  const amrex::Real a = metric[0];
  const amrex::Real b = metric[1];
  const amrex::Real c = metric[2];
  const amrex::Real d = metric[3];
  const amrex::Real e = metric[4];
  const amrex::Real f = metric[5];
  const amrex::Real determinant =
      a * (d * f - e * e) - b * (b * f - c * e) +
      c * (b * e - c * d);
  if (!(determinant > 0) || !amrex::Math::isfinite(determinant))
    return false;
  const amrex::Real inverse_determinant = 1 / determinant;
  inverse[0][0] = (d * f - e * e) * inverse_determinant;
  inverse[0][1] = inverse[1][0] = (c * e - b * f) * inverse_determinant;
  inverse[0][2] = inverse[2][0] = (b * e - c * d) * inverse_determinant;
  inverse[1][1] = (a * f - c * c) * inverse_determinant;
  inverse[1][2] = inverse[2][1] = (b * c - a * e) * inverse_determinant;
  inverse[2][2] = (a * d - b * b) * inverse_determinant;
  return true;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
evaluate_expansion(const ExpansionInput &input, ExpansionOutput &output)
    noexcept {
  amrex::Real inverse[3][3];
  if (!inverse_metric(input.gamma, inverse) || !(input.height > 0))
    return false;

  const AngularGeometry basis = angular_geometry(input.theta, input.phi);

  amrex::Real tangent_theta[3];
  amrex::Real tangent_phi[3];
  amrex::Real second_theta_theta[3];
  amrex::Real second_theta_phi[3];
  amrex::Real second_phi_phi[3];
  for (int i = 0; i < 3; ++i) {
    tangent_theta[i] =
        input.h_theta * basis.radial[i] +
        input.height * basis.radial_theta[i];
    tangent_phi[i] =
        input.h_phi * basis.radial[i] + input.height * basis.radial_phi[i];
    second_theta_theta[i] =
        input.h_theta_theta * basis.radial[i] +
        2 * input.h_theta * basis.radial_theta[i] +
        input.height * basis.radial_theta_theta[i];
    second_theta_phi[i] =
        input.h_theta_phi * basis.radial[i] +
        input.h_theta * basis.radial_phi[i] +
        input.h_phi * basis.radial_theta[i] +
        input.height * basis.radial_theta_phi[i];
    second_phi_phi[i] =
        input.h_phi_phi * basis.radial[i] +
        2 * input.h_phi * basis.radial_phi[i] +
        input.height * basis.radial_phi_phi[i];
  }

  amrex::Real normal_covector[3]{
      tangent_theta[1] * tangent_phi[2] -
          tangent_theta[2] * tangent_phi[1],
      tangent_theta[2] * tangent_phi[0] -
          tangent_theta[0] * tangent_phi[2],
      tangent_theta[0] * tangent_phi[1] -
          tangent_theta[1] * tangent_phi[0]};
  amrex::Real normal_squared = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      normal_squared += normal_covector[i] * inverse[i][j] *
                        normal_covector[j];
  if (!(normal_squared > 0) || !amrex::Math::isfinite(normal_squared))
    return false;
  const amrex::Real inverse_normal = amrex::Math::rsqrt(normal_squared);
  for (int i = 0; i < 3; ++i)
    normal_covector[i] *= inverse_normal;
  for (int i = 0; i < 3; ++i) {
    output.normal[i] = 0;
    for (int j = 0; j < 3; ++j)
      output.normal[i] += inverse[i][j] * normal_covector[j];
  }

  const amrex::Real induced_theta_theta =
      metric_dot(input.gamma, tangent_theta, tangent_theta);
  const amrex::Real induced_theta_phi =
      metric_dot(input.gamma, tangent_theta, tangent_phi);
  const amrex::Real induced_phi_phi =
      metric_dot(input.gamma, tangent_phi, tangent_phi);
  const amrex::Real induced_determinant =
      induced_theta_theta * induced_phi_phi -
      induced_theta_phi * induced_theta_phi;
  if (!(induced_determinant > 0) ||
      !amrex::Math::isfinite(induced_determinant))
    return false;
  const amrex::Real inverse_induced_determinant = 1 / induced_determinant;
  const amrex::Real induced_inverse_theta_theta =
      induced_phi_phi * inverse_induced_determinant;
  const amrex::Real induced_inverse_theta_phi =
      -induced_theta_phi * inverse_induced_determinant;
  const amrex::Real induced_inverse_phi_phi =
      induced_theta_theta * inverse_induced_determinant;

  amrex::Real christoffel[3][3][3];
  for (int upper = 0; upper < 3; ++upper)
    for (int lower_a = 0; lower_a < 3; ++lower_a)
      for (int lower_b = 0; lower_b < 3; ++lower_b) {
        christoffel[upper][lower_a][lower_b] = 0;
        for (int contracted = 0; contracted < 3; ++contracted) {
          const amrex::Real derivative_a = input.d_gamma[lower_a]
              [symmetric_component(contracted, lower_b)];
          const amrex::Real derivative_b = input.d_gamma[lower_b]
              [symmetric_component(contracted, lower_a)];
          const amrex::Real derivative_contracted =
              input.d_gamma[contracted]
                           [symmetric_component(lower_a, lower_b)];
          christoffel[upper][lower_a][lower_b] +=
              amrex::Real(0.5) * inverse[upper][contracted] *
              (derivative_a + derivative_b - derivative_contracted);
        }
      }

  amrex::Real curvature_theta_theta = 0;
  amrex::Real curvature_theta_phi = 0;
  amrex::Real curvature_phi_phi = 0;
  for (int i = 0; i < 3; ++i) {
    amrex::Real connection_theta_theta = 0;
    amrex::Real connection_theta_phi = 0;
    amrex::Real connection_phi_phi = 0;
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        connection_theta_theta +=
            christoffel[i][j][k] * tangent_theta[j] * tangent_theta[k];
        connection_theta_phi +=
            christoffel[i][j][k] * tangent_theta[j] * tangent_phi[k];
        connection_phi_phi +=
            christoffel[i][j][k] * tangent_phi[j] * tangent_phi[k];
      }
    curvature_theta_theta -= normal_covector[i] *
                             (second_theta_theta[i] +
                              connection_theta_theta);
    curvature_theta_phi -=
        normal_covector[i] *
        (second_theta_phi[i] + connection_theta_phi);
    curvature_phi_phi -= normal_covector[i] *
                         (second_phi_phi[i] + connection_phi_phi);
  }
  const amrex::Real divergence =
      induced_inverse_theta_theta * curvature_theta_theta +
      2 * induced_inverse_theta_phi * curvature_theta_phi +
      induced_inverse_phi_phi * curvature_phi_phi;

  amrex::Real trace_curvature = 0;
  amrex::Real normal_curvature = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const amrex::Real curvature = symmetric_value(input.curv, i, j);
      trace_curvature += inverse[i][j] * curvature;
      normal_curvature += output.normal[i] * curvature * output.normal[j];
    }
  output.theta = divergence + normal_curvature - trace_curvature;
  output.area_weight = induced_determinant *
                       amrex::Math::rsqrt(induced_determinant) *
                       input.dtheta * input.dphi;
  if (!amrex::Math::isfinite(output.theta) ||
      !amrex::Math::isfinite(output.area_weight))
    return false;
  for (int i = 0; i < 3; ++i)
    if (!amrex::Math::isfinite(output.normal[i]))
      return false;
  return true;
}

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_EXPANSION_HXX
