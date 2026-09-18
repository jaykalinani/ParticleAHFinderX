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
