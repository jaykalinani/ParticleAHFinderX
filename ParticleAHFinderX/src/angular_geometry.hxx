/**
 * \file angular_geometry.hxx
 * \brief Portable Cartesian basis for the logical spherical coordinates.
 */
#ifndef PARTICLEAHFINDERX_ANGULAR_GEOMETRY_HXX
#define PARTICLEAHFINDERX_ANGULAR_GEOMETRY_HXX

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

namespace ParticleAHFinderX {

struct AngularGeometry {
  amrex::Real radial[3];
  amrex::Real radial_theta[3];
  amrex::Real radial_phi[3];
  amrex::Real radial_theta_theta[3];
  amrex::Real radial_theta_phi[3];
  amrex::Real radial_phi_phi[3];
};

/**
 * Ratio between the exact latitude-cell solid angle and its midpoint form.
 *
 * A cell centered at theta with width dtheta has exact unit-sphere measure
 *
 *   2 sin(theta) sin(dtheta / 2) dphi,
 *
 * whereas a plain midpoint rule uses sin(theta) dtheta dphi.  Factoring this
 * ratio from a regular radial-graph area Jacobian integrates a round sphere
 * exactly and removes the leading latitude-measure bias without changing the
 * tensor-product angular topology.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
latitude_cell_measure_ratio(const amrex::Real dtheta) noexcept {
  return 2 * amrex::Math::sincos(dtheta / 2).first / dtheta;
}

/** Exact-latitude cell weight from a midpoint intrinsic area density. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
surface_cell_area_weight(const amrex::Real sqrt_determinant,
                         const amrex::Real dtheta,
                         const amrex::Real dphi) noexcept {
  return sqrt_determinant * dtheta * dphi *
         latitude_cell_measure_ratio(dtheta);
}

/**
 * Standard right-handed spherical basis with
 * x=r sin(theta) cos(phi), y=r sin(theta) sin(phi), z=r cos(theta).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE AngularGeometry
angular_geometry(const amrex::Real theta, const amrex::Real phi) noexcept {
  const auto theta_sincos = amrex::Math::sincos(theta);
  const auto phi_sincos = amrex::Math::sincos(phi);
  const amrex::Real sin_theta = theta_sincos.first;
  const amrex::Real cos_theta = theta_sincos.second;
  const amrex::Real sin_phi = phi_sincos.first;
  const amrex::Real cos_phi = phi_sincos.second;
  return {
      {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta},
      {cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta},
      {-sin_theta * sin_phi, sin_theta * cos_phi, 0},
      {-sin_theta * cos_phi, -sin_theta * sin_phi, -cos_theta},
      {-cos_theta * sin_phi, cos_theta * cos_phi, 0},
      {-sin_theta * cos_phi, -sin_theta * sin_phi, 0},
  };
}

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_ANGULAR_GEOMETRY_HXX
