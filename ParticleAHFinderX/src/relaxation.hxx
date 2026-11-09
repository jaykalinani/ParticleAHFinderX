/**
 * \file relaxation.hxx
 * \brief Portable explicit stage combinations for surface pseudo-time.
 */
#ifndef PARTICLEAHFINDERX_RELAXATION_HXX
#define PARTICLEAHFINDERX_RELAXATION_HXX

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

namespace ParticleAHFinderX {

enum class RelaxationIntegrator : int {
  ssprk3,
  rk4,
};

enum class RelaxationExtrapolationMode : int {
  full_shape,
  low_modes,
};

/** Alternate stored trajectory references with trials, starting at event 1. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr bool
relaxation_extrapolation_reference_event(const int event) noexcept {
  return event > 0 && event % 2 == 1;
}

/**
 * Extrapolate a recent surface trajectory.  An overstep of one restores the
 * current surface; larger values continue from `previous` through `current`.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
relaxation_extrapolated_height(const amrex::Real previous,
                              const amrex::Real current,
                              const amrex::Real overstep) noexcept {
  return current + (overstep - 1) * (current - previous);
}

/** Continue a supplied low-angular-mode displacement of a recent trajectory. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
relaxation_extrapolated_low_modes(
    const amrex::Real current, const amrex::Real low_mode_displacement,
    const amrex::Real overstep) noexcept {
  return current + (overstep - 1) * low_mode_displacement;
}

/**
 * Use the same dimensionless two-norm objective for extrapolation decisions
 * as for nonlinear convergence.  Either norm can be the limiting one as a
 * surface moves through angular continuation.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
relaxation_residual_ratio(const amrex::Real l2, const amrex::Real linf,
                          const amrex::Real mass_scale,
                          const amrex::Real l2_times_mass_tolerance,
                          const amrex::Real linf_times_mass_tolerance) noexcept {
  const amrex::Real l2_ratio =
      mass_scale * l2 / l2_times_mass_tolerance;
  const amrex::Real linf_ratio =
      mass_scale * linf / linf_times_mass_tolerance;
  return l2_ratio > linf_ratio ? l2_ratio : linf_ratio;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr int
relaxation_stage_count(const RelaxationIntegrator integrator) noexcept {
  return integrator == RelaxationIntegrator::ssprk3 ? 3 : 4;
}

/**
 * Combine one explicit stage from a fixed step base and stored RHS values.
 * `current_rhs` is the RHS evaluated at the current stage state.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
relaxation_stage_value(const RelaxationIntegrator integrator, const int stage,
                       const amrex::Real base, const amrex::Real dt,
                       const amrex::Real rhs_1, const amrex::Real rhs_2,
                       const amrex::Real rhs_3,
                       const amrex::Real current_rhs) noexcept {
  if (integrator == RelaxationIntegrator::ssprk3) {
    if (stage == 0)
      return base + dt * current_rhs;
    if (stage == 1)
      return base + dt * (rhs_1 + current_rhs) / 4;
    return base + dt * (rhs_1 / 6 + rhs_2 / 6 +
                        amrex::Real(2) * current_rhs / 3);
  }

  if (stage < 2)
    return base + dt * current_rhs / 2;
  if (stage == 2)
    return base + dt * current_rhs;
  return base + dt * (rhs_1 + 2 * rhs_2 + 2 * rhs_3 + current_rhs) / 6;
}

struct RelaxationStageUpdate {
  amrex::Real height;
  amrex::Real velocity;
  bool valid;
};

/** Form and validate the point update shared by fused and stencil stages. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE RelaxationStageUpdate
relaxation_stage_update(
    const RelaxationIntegrator integrator, const int stage,
    const amrex::Long point_count, const amrex::Long point,
    const amrex::Real *const base, const amrex::Real *const rhs,
    const amrex::Real dt, const amrex::Real current_rhs_height,
    const amrex::Real current_rhs_velocity,
    const amrex::Real minimum_radius,
    const amrex::Real maximum_radius) noexcept {
  amrex::Real rhs_height_1 = 0;
  amrex::Real rhs_height_2 = 0;
  amrex::Real rhs_height_3 = 0;
  amrex::Real rhs_velocity_1 = 0;
  amrex::Real rhs_velocity_2 = 0;
  amrex::Real rhs_velocity_3 = 0;
  if ((integrator == RelaxationIntegrator::ssprk3 && stage >= 1) ||
      (integrator == RelaxationIntegrator::rk4 && stage == 3)) {
    rhs_height_1 = rhs[point];
    rhs_velocity_1 = rhs[point_count + point];
  }
  if ((integrator == RelaxationIntegrator::ssprk3 && stage >= 2) ||
      (integrator == RelaxationIntegrator::rk4 && stage == 3)) {
    rhs_height_2 = rhs[2 * point_count + point];
    rhs_velocity_2 = rhs[3 * point_count + point];
  }
  if (integrator == RelaxationIntegrator::rk4 && stage == 3) {
    rhs_height_3 = rhs[4 * point_count + point];
    rhs_velocity_3 = rhs[5 * point_count + point];
  }
  RelaxationStageUpdate result;
  result.height = relaxation_stage_value(
      integrator, stage, base[point], dt, rhs_height_1, rhs_height_2,
      rhs_height_3, current_rhs_height);
  result.velocity = relaxation_stage_value(
      integrator, stage, base[point_count + point], dt,
      rhs_velocity_1, rhs_velocity_2, rhs_velocity_3,
      current_rhs_velocity);
  result.valid = amrex::Math::isfinite(result.height) &&
                 amrex::Math::isfinite(result.velocity) &&
                 result.height >= minimum_radius &&
                 result.height <= maximum_radius;
  return result;
}

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_RELAXATION_HXX
