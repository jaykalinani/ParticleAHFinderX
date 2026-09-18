/**
 * \file relaxation_self_test.cxx
 * \brief Optional accelerator test of internal pseudo-time stage updates.
 */
#include "relaxation.hxx"
#include "angular_geometry.hxx"
#include "expansion.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>

#include <algorithm>
#include <cmath>
#include <limits>

namespace ParticleAHFinderX {
namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
isotropic_schwarzschild_expansion(const amrex::Real radius,
                                  const amrex::Real mass) noexcept {
  const amrex::Real conformal_factor = 1 + mass / (2 * radius);
  const amrex::Real radial_derivative = -mass / (2 * radius * radius);
  return (2 / radius + 4 * radial_derivative / conformal_factor) /
         (conformal_factor * conformal_factor);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
absolute_value(const amrex::Real value) noexcept {
  return value < 0 ? -value : value;
}

} // namespace

extern "C" void ParticleAHFinderX_RelaxationSelfTest(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  if (!run_relaxation_self_test)
    return;

  constexpr int num_integrators = 2;
  amrex::Gpu::DeviceVector<amrex::Real> device_results(num_integrators);
  auto *const results = device_results.dataPtr();
  amrex::ParallelFor(
      num_integrators,
      [=] AMREX_GPU_DEVICE(const int index) noexcept {
        const RelaxationIntegrator integrator =
            index == 0 ? RelaxationIntegrator::ssprk3
                       : RelaxationIntegrator::rk4;
        constexpr amrex::Real initial_value = 1;
        constexpr amrex::Real eigenvalue = -2;
        constexpr amrex::Real dt = amrex::Real(0.1);
        amrex::Real rhs[4]{0, 0, 0, 0};
        amrex::Real value = initial_value;
        for (int stage = 0; stage < relaxation_stage_count(integrator);
             ++stage) {
          rhs[stage] = eigenvalue * value;
          value = relaxation_stage_value(integrator, stage, initial_value, dt,
                                         rhs[0], rhs[1], rhs[2], rhs[stage]);
        }
        results[index] = value;
      });

  amrex::Real host_results[num_integrators]{0, 0};
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_results.begin(),
                   device_results.end(), host_results);
  const amrex::Real z = amrex::Real(-0.2);
  const amrex::Real expected_ssprk3 =
      1 + z + z * z / 2 + z * z * z / 6;
  const amrex::Real expected_rk4 =
      expected_ssprk3 + z * z * z * z / 24;
  const amrex::Real stage_error =
      std::max(std::abs(host_results[0] - expected_ssprk3),
               std::abs(host_results[1] - expected_rk4));
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(const int) noexcept {
    results[0] = relaxation_extrapolated_height(
        amrex::Real(0.3), amrex::Real(0.28), amrex::Real(3));
    const amrex::Real mean_height = relaxation_extrapolated_mean_height(
        amrex::Real(0.3), amrex::Real(-0.02), amrex::Real(3));
    const amrex::Real ratio = relaxation_residual_ratio(
        amrex::Real(4), amrex::Real(3), amrex::Real(0.5), amrex::Real(1),
        amrex::Real(2));
    results[1] = absolute_value(mean_height - amrex::Real(0.26)) +
                 absolute_value(ratio - amrex::Real(2));
  });
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_results.begin(),
                   device_results.end(), host_results);
  const amrex::Real extrapolation_error = std::max(
      std::abs(host_results[0] - amrex::Real(0.24)),
      std::abs(host_results[1]));
  const amrex::Real maximum_error =
      std::max(stage_error, extrapolation_error);
  const amrex::Real tolerance =
      100 * std::numeric_limits<amrex::Real>::epsilon();
  if (maximum_error > tolerance)
    CCTK_VERROR("ParticleAHFinderX relaxation integrator self-test failed: "
                "error=%.17g tolerance=%.17g",
                double(maximum_error), double(tolerance));

  // Exercise the complete geometric expansion formula on an independent
  // analytic MOTS. Isotropic Schwarzschild has gamma_ij=psi^4 delta_ij,
  // K_ij=0, and a horizon at coordinate radius M/2.
  constexpr int num_expansion_points = 128;
  amrex::Gpu::DeviceVector<amrex::Real> device_expansion_error(1, 0);
  amrex::Gpu::DeviceVector<int> device_expansion_failure(1, 0);
  amrex::Real *const expansion_error = device_expansion_error.dataPtr();
  int *const expansion_failure = device_expansion_failure.dataPtr();
  amrex::ParallelFor(
      num_expansion_points,
      [=] AMREX_GPU_DEVICE(const int point) noexcept {
        constexpr amrex::Real mass = amrex::Real(0.5);
        const amrex::Real radius =
            point % 2 == 0 ? mass / 2 : amrex::Real(0.3);
        const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                                  (point + amrex::Real(0.5)) /
                                  num_expansion_points;
        const amrex::Real phi = 2 * amrex::Math::pi<amrex::Real>() *
                                (point + amrex::Real(0.5)) /
                                num_expansion_points;
        const AngularGeometry basis = angular_geometry(theta, phi);
        const amrex::Real conformal_factor = 1 + mass / (2 * radius);
        const amrex::Real conformal_metric =
            conformal_factor * conformal_factor * conformal_factor *
            conformal_factor;
        const amrex::Real radial_derivative =
            -mass / (2 * radius * radius);

        ExpansionInput input{};
        input.gamma[0] = conformal_metric;
        input.gamma[3] = conformal_metric;
        input.gamma[5] = conformal_metric;
        for (int d = 0; d < 3; ++d) {
          const amrex::Real derivative =
              4 * conformal_factor * conformal_factor * conformal_factor *
              radial_derivative * basis.radial[d];
          input.d_gamma[d][0] = derivative;
          input.d_gamma[d][3] = derivative;
          input.d_gamma[d][5] = derivative;
        }
        input.height = radius;
        input.theta = theta;
        input.phi = phi;
        input.dtheta = amrex::Math::pi<amrex::Real>() /
                       num_expansion_points;
        input.dphi = 2 * amrex::Math::pi<amrex::Real>() /
                     num_expansion_points;
        ExpansionOutput output;
        if (!evaluate_expansion(input, output)) {
          amrex::Gpu::Atomic::Exch(expansion_failure, 1);
          return;
        }
        const amrex::Real expected =
            isotropic_schwarzschild_expansion(radius, mass);
        amrex::Gpu::Atomic::Max(
            expansion_error, absolute_value(output.theta - expected));
      });

  amrex::Real host_expansion_error = 0;
  int host_expansion_failure = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   device_expansion_error.begin(),
                   device_expansion_error.end(), &host_expansion_error);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   device_expansion_failure.begin(),
                   device_expansion_failure.end(), &host_expansion_failure);
  const amrex::Real expansion_tolerance =
      10000 * std::numeric_limits<amrex::Real>::epsilon();
  if (host_expansion_failure || host_expansion_error > expansion_tolerance)
    CCTK_VERROR("ParticleAHFinderX isotropic-Schwarzschild expansion "
                "self-test failed: evaluation_failure=%d error=%.17g "
                "tolerance=%.17g",
                host_expansion_failure, double(host_expansion_error),
                double(expansion_tolerance));

  // Finally test the coupled h/v sign and damping on that analytic horizon.
  // This is deliberately separate from the stage-polynomial test above: a
  // sign error can integrate a manufactured decaying ODE perfectly while
  // driving a physical surface away from Theta=0.
  amrex::Gpu::DeviceVector<amrex::Real> device_physical_results(
      2 * num_integrators);
  amrex::Real *const physical_results = device_physical_results.dataPtr();
  amrex::ParallelFor(
      num_integrators,
      [=] AMREX_GPU_DEVICE(const int index) noexcept {
        constexpr amrex::Real mass = amrex::Real(0.5);
        constexpr amrex::Real eta_times_mass = amrex::Real(1.6);
        constexpr amrex::Real eta = eta_times_mass / mass;
        constexpr amrex::Real dt = amrex::Real(0.005);
        constexpr int steps = 4000;
        const RelaxationIntegrator integrator =
            index == 0 ? RelaxationIntegrator::ssprk3
                       : RelaxationIntegrator::rk4;
        amrex::Real height = amrex::Real(0.3);
        amrex::Real velocity = eta * height;
        for (int step = 0; step < steps; ++step) {
          const amrex::Real base_height = height;
          const amrex::Real base_velocity = velocity;
          amrex::Real rhs_height[4]{0, 0, 0, 0};
          amrex::Real rhs_velocity[4]{0, 0, 0, 0};
          for (int stage = 0; stage < relaxation_stage_count(integrator);
               ++stage) {
            rhs_height[stage] = velocity - eta * height;
            rhs_velocity[stage] =
                -isotropic_schwarzschild_expansion(height, mass);
            height = relaxation_stage_value(
                integrator, stage, base_height, dt, rhs_height[0],
                rhs_height[1], rhs_height[2], rhs_height[stage]);
            velocity = relaxation_stage_value(
                integrator, stage, base_velocity, dt, rhs_velocity[0],
                rhs_velocity[1], rhs_velocity[2], rhs_velocity[stage]);
          }
        }
        physical_results[2 * index] = height;
        physical_results[2 * index + 1] =
            isotropic_schwarzschild_expansion(height, mass);
      });

  amrex::Real host_physical_results[2 * num_integrators]{0, 0, 0, 0};
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   device_physical_results.begin(),
                   device_physical_results.end(), host_physical_results);
  amrex::Real physical_error = 0;
  for (int index = 0; index < num_integrators; ++index) {
    physical_error = std::max(
        physical_error,
        std::abs(host_physical_results[2 * index] - amrex::Real(0.25)));
    physical_error = std::max(
        physical_error,
        std::abs(host_physical_results[2 * index + 1]));
  }
  const amrex::Real physical_tolerance =
      std::max(amrex::Real(1.0e-11),
               1000 * std::numeric_limits<amrex::Real>::epsilon());
  if (physical_error > physical_tolerance)
    CCTK_VERROR("ParticleAHFinderX analytic physical-relaxation self-test "
                "failed: error=%.17g tolerance=%.17g",
                double(physical_error), double(physical_tolerance));

  CCTK_VINFO(
      "Internal SSPRK3/RK4 relaxation self-test passed on the active "
      "AMReX backend: stage error=%.3g Schwarzschild expansion error=%.3g "
      "physical relaxation error=%.3g extrapolation error=%.3g",
      double(stage_error), double(host_expansion_error),
      double(physical_error), double(extrapolation_error));
}

} // namespace ParticleAHFinderX
