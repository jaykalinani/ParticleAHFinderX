/**
 * \file multipoles.hxx
 * \brief Portable helpers for invariant horizon multipole moments.
 */
#ifndef PARTICLEAHFINDERX_MULTIPOLES_HXX
#define PARTICLEAHFINDERX_MULTIPOLES_HXX

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

namespace ParticleAHFinderX {

constexpr int maximum_multipole_degree = 8;
constexpr int multipole_component_count = maximum_multipole_degree + 1;

/**
 * Evaluate P_l(zeta) and d P_l / d zeta through l=8.
 *
 * The coupled three-term recurrence is regular at both poles and avoids the
 * division by 1-zeta^2 present in the common derivative identity.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
legendre_values_and_derivatives(
    const amrex::Real zeta,
    amrex::Real (&polynomial)[multipole_component_count],
    amrex::Real (&derivative)[multipole_component_count]) noexcept {
  if (!amrex::Math::isfinite(zeta))
    return false;
  polynomial[0] = 1;
  derivative[0] = 0;
  polynomial[1] = zeta;
  derivative[1] = 1;
  for (int degree = 2; degree <= maximum_multipole_degree; ++degree) {
    const amrex::Real previous_factor = 2 * degree - 1;
    polynomial[degree] =
        (previous_factor * zeta * polynomial[degree - 1] -
         (degree - 1) * polynomial[degree - 2]) /
        degree;
    derivative[degree] =
        (previous_factor *
             (polynomial[degree - 1] +
              zeta * derivative[degree - 1]) -
         (degree - 1) * derivative[degree - 2]) /
        degree;
    if (!amrex::Math::isfinite(polynomial[degree]) ||
        !amrex::Math::isfinite(derivative[degree]))
      return false;
  }
  return true;
}

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_MULTIPOLES_HXX
