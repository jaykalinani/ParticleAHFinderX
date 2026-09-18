/**
 * \file angular_derivatives.hxx
 * \brief Portable sixth-order operators on a cell-centered sphere.
 *
 * Reflecting theta across either pole shifts phi by pi. The finite-difference
 * coefficients are the standard centered sixth-order stencils.
 */
#ifndef PARTICLEAHFINDERX_ANGULAR_DERIVATIVES_HXX
#define PARTICLEAHFINDERX_ANGULAR_DERIVATIVES_HXX

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

namespace ParticleAHFinderX {

struct AngularDerivatives {
  amrex::Real theta = 0;
  amrex::Real phi = 0;
  amrex::Real theta_theta = 0;
  amrex::Real theta_phi = 0;
  amrex::Real phi_phi = 0;
};

struct AngularDissipation {
  amrex::Real theta = 0;
  amrex::Real phi = 0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
periodic_index(int index, const int period) noexcept {
  index %= period;
  return index < 0 ? index + period : index;
}

/** Read a scalar through the two polar identifications of S^2. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real angular_scalar(
    const amrex::Real *const values, const amrex::Long offset,
    const int ntheta, const int nphi, int itheta, int iphi) noexcept {
  while (itheta < 0 || itheta >= ntheta) {
    if (itheta < 0)
      itheta = -itheta - 1;
    else
      itheta = 2 * ntheta - itheta - 1;
    iphi += nphi / 2;
  }
  iphi = periodic_index(iphi, nphi);
  return values[offset + amrex::Long(itheta) * nphi + iphi];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
first_derivative_coefficient(const int stencil) noexcept {
  constexpr amrex::Real coefficients[7]{
      -amrex::Real(1) / 60, amrex::Real(3) / 20,
      -amrex::Real(3) / 4, 0, amrex::Real(3) / 4,
      -amrex::Real(3) / 20, amrex::Real(1) / 60};
  return coefficients[stencil + 3];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
second_derivative_coefficient(const int stencil) noexcept {
  constexpr amrex::Real coefficients[7]{
      amrex::Real(1) / 90, -amrex::Real(3) / 20,
      amrex::Real(3) / 2, -amrex::Real(49) / 18,
      amrex::Real(3) / 2, -amrex::Real(3) / 20,
      amrex::Real(1) / 90};
  return coefficients[stencil + 3];
}

/** Sixth-order Kreiss-Oliger filter in each angular direction. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE AngularDissipation
angular_dissipation(const amrex::Real *const values,
                    const amrex::Long offset, const int ntheta,
                    const int nphi, const int itheta,
                    const int iphi) noexcept {
  const amrex::Real center = angular_scalar(values, offset, ntheta, nphi,
                                             itheta, iphi);
  AngularDissipation result;
  for (int direction = 0; direction < 2; ++direction) {
    amrex::Real filtered = -amrex::Real(5) / 16 * center;
    for (int stencil = 1; stencil <= 3; ++stencil) {
      const amrex::Real coefficient =
          stencil == 1 ? amrex::Real(15) / 64
          : stencil == 2 ? -amrex::Real(3) / 32
                         : amrex::Real(1) / 64;
      filtered +=
          coefficient *
          (angular_scalar(values, offset, ntheta, nphi,
                          itheta + (direction == 0 ? -stencil : 0),
                          iphi + (direction == 1 ? -stencil : 0)) +
           angular_scalar(values, offset, ntheta, nphi,
                          itheta + (direction == 0 ? stencil : 0),
                          iphi + (direction == 1 ? stencil : 0)));
    }
    if (direction == 0)
      result.theta = filtered * ntheta /
                     amrex::Math::pi<amrex::Real>();
    else
      result.phi = filtered * nphi /
                   (2 * amrex::Math::pi<amrex::Real>());
  }
  return result;
}

/** Evaluate first and second coordinate derivatives of one scalar field. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE AngularDerivatives
angular_derivatives(const amrex::Real *const values,
                    const amrex::Long offset, const int ntheta,
                    const int nphi, const int itheta,
                    const int iphi) noexcept {
  const amrex::Real inverse_dtheta = amrex::Real(ntheta) /
                                      amrex::Math::pi<amrex::Real>();
  const amrex::Real inverse_dphi = amrex::Real(nphi) /
                                    (2 * amrex::Math::pi<amrex::Real>());
  AngularDerivatives result;
  for (int stencil = -3; stencil <= 3; ++stencil) {
    const amrex::Real first = first_derivative_coefficient(stencil);
    const amrex::Real second = second_derivative_coefficient(stencil);
    result.theta += first * angular_scalar(values, offset, ntheta, nphi,
                                            itheta + stencil, iphi);
    result.phi += first * angular_scalar(values, offset, ntheta, nphi,
                                          itheta, iphi + stencil);
    result.theta_theta +=
        second * angular_scalar(values, offset, ntheta, nphi,
                                itheta + stencil, iphi);
    result.phi_phi +=
        second * angular_scalar(values, offset, ntheta, nphi,
                                itheta, iphi + stencil);
    for (int other = -3; other <= 3; ++other)
      result.theta_phi +=
          first * first_derivative_coefficient(other) *
          angular_scalar(values, offset, ntheta, nphi, itheta + stencil,
                         iphi + other);
  }
  result.theta *= inverse_dtheta;
  result.phi *= inverse_dphi;
  result.theta_theta *= inverse_dtheta * inverse_dtheta;
  result.theta_phi *= inverse_dtheta * inverse_dphi;
  result.phi_phi *= inverse_dphi * inverse_dphi;
  return result;
}

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_ANGULAR_DERIVATIVES_HXX
