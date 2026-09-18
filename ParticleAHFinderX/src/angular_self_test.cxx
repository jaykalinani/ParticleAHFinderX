/**
 * \file angular_self_test.cxx
 * \brief Optional accelerator test of the spherical angular stencils.
 */
#include "angular_derivatives.hxx"
#include "angular_geometry.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>

#include <algorithm>
#include <limits>

namespace ParticleAHFinderX {
namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
absolute_value(const amrex::Real value) noexcept {
  return value < 0 ? -value : value;
}

} // namespace

extern "C" void ParticleAHFinderX_AngularSelfTest(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  if (!run_angular_self_test)
    return;

  constexpr int test_ntheta = 32;
  constexpr int test_nphi = 64;
  constexpr int count = test_ntheta * test_nphi;
  amrex::Gpu::DeviceVector<amrex::Real> values(count);
  amrex::Real *const field = values.dataPtr();
  amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(const int point) noexcept {
    const int itheta = point / test_nphi;
    const int iphi = point - itheta * test_nphi;
    const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                              (itheta + amrex::Real(0.5)) / test_ntheta;
    const amrex::Real phi = 2 * amrex::Math::pi<amrex::Real>() *
                            (iphi + amrex::Real(0.5)) / test_nphi;
    const auto theta_sincos = amrex::Math::sincos(theta);
    const auto two_phi_sincos = amrex::Math::sincos(2 * phi);
    field[point] = 2 + amrex::Real(0.1) * theta_sincos.first *
                           theta_sincos.first * two_phi_sincos.second +
                   amrex::Real(0.05) * theta_sincos.second;
  });

  amrex::Gpu::DeviceVector<amrex::Real> device_error(1, 0);
  amrex::Gpu::DeviceVector<amrex::Real> constant_values(count, 1);
  amrex::Gpu::DeviceVector<amrex::Real> cutoff_values(count);
  amrex::Real *const maximum_error = device_error.dataPtr();
  const amrex::Real *const constant = constant_values.dataPtr();
  amrex::Real *const cutoff = cutoff_values.dataPtr();
  amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(const int point) noexcept {
    const int iphi = point % test_nphi;
    cutoff[point] = 1 + amrex::Real(0.02) * (iphi % 2 == 0 ? 1 : -1);
  });
  amrex::Gpu::DeviceVector<amrex::Real> device_indicator(2, 0);
  amrex::Real *const maximum_indicator = device_indicator.dataPtr();
  amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(const int point) noexcept {
    const int itheta = point / test_nphi;
    const int iphi = point - itheta * test_nphi;
    const amrex::Real theta = amrex::Math::pi<amrex::Real>() *
                              (itheta + amrex::Real(0.5)) / test_ntheta;
    const amrex::Real phi = 2 * amrex::Math::pi<amrex::Real>() *
                            (iphi + amrex::Real(0.5)) / test_nphi;
    const auto theta_sincos = amrex::Math::sincos(theta);
    const auto two_phi_sincos = amrex::Math::sincos(2 * phi);
    const amrex::Real sin_theta = theta_sincos.first;
    const amrex::Real cos_theta = theta_sincos.second;
    const amrex::Real sin_two_phi = two_phi_sincos.first;
    const amrex::Real cos_two_phi = two_phi_sincos.second;
    const AngularDerivatives numerical =
        angular_derivatives(field, 0, test_ntheta, test_nphi, itheta, iphi);
    const AngularDissipation constant_dissipation =
        angular_dissipation(constant, 0, test_ntheta, test_nphi, itheta,
                            iphi);
    const AngularDissipation smooth_dissipation =
        angular_dissipation(field, 0, test_ntheta, test_nphi, itheta, iphi);
    const AngularDissipation cutoff_dissipation =
        angular_dissipation(cutoff, 0, test_ntheta, test_nphi, itheta, iphi);
    const amrex::Real dtheta =
        amrex::Math::pi<amrex::Real>() / test_ntheta;
    const amrex::Real dphi =
        2 * amrex::Math::pi<amrex::Real>() / test_nphi;
    const amrex::Real smooth_indicator = amrex::max(
        dtheta * absolute_value(smooth_dissipation.theta),
        dphi * absolute_value(smooth_dissipation.phi));
    const amrex::Real cutoff_indicator =
        amrex::max(dtheta * absolute_value(cutoff_dissipation.theta),
                   dphi * absolute_value(cutoff_dissipation.phi)) /
        cutoff[point];
    amrex::Gpu::Atomic::Max(maximum_indicator, smooth_indicator);
    amrex::Gpu::Atomic::Max(maximum_indicator + 1, cutoff_indicator);
    const amrex::Real expected[5]{
        amrex::Real(0.2) * sin_theta * cos_theta * cos_two_phi -
            amrex::Real(0.05) * sin_theta,
        -amrex::Real(0.2) * sin_theta * sin_theta * sin_two_phi,
        amrex::Real(0.2) * (cos_theta * cos_theta -
                            sin_theta * sin_theta) *
                cos_two_phi -
            amrex::Real(0.05) * cos_theta,
        -amrex::Real(0.4) * sin_theta * cos_theta * sin_two_phi,
        -amrex::Real(0.4) * sin_theta * sin_theta * cos_two_phi};
    const amrex::Real observed[5]{
        numerical.theta, numerical.phi, numerical.theta_theta,
        numerical.theta_phi, numerical.phi_phi};
    amrex::Real point_error = 0;
    for (int component = 0; component < 5; ++component)
      point_error =
          amrex::max(point_error,
                     absolute_value(observed[component] - expected[component]));
    point_error = amrex::max(
        point_error, absolute_value(constant_dissipation.theta));
    point_error = amrex::max(
        point_error, absolute_value(constant_dissipation.phi));
    amrex::Gpu::Atomic::Max(maximum_error, point_error);
  });

  amrex::Real host_error = 0;
  amrex::Real host_indicator[2]{0, 0};
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_error.begin(),
                   device_error.end(), &host_error);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_indicator.begin(),
                   device_indicator.end(), host_indicator);
  const AngularGeometry x_axis = angular_geometry(
      amrex::Math::pi<amrex::Real>() / 2, amrex::Real(0));
  const amrex::Real expected_radial[3]{1, 0, 0};
  const amrex::Real expected_theta[3]{0, 0, -1};
  const amrex::Real expected_phi[3]{0, 1, 0};
  for (int d = 0; d < 3; ++d) {
    host_error = amrex::max(
        host_error,
        absolute_value(x_axis.radial[d] - expected_radial[d]));
    host_error = amrex::max(
        host_error,
        absolute_value(x_axis.radial_theta[d] - expected_theta[d]));
    host_error = amrex::max(
        host_error,
        absolute_value(x_axis.radial_phi[d] - expected_phi[d]));
  }
  const amrex::Real tolerance = amrex::max(
      amrex::Real(2.0e-5),
      amrex::Real(10000) * std::numeric_limits<amrex::Real>::epsilon());
  if (host_error > tolerance)
    CCTK_VERROR("ParticleAHFinderX angular self-test failed: error=%.17g "
                "tolerance=%.17g",
                double(host_error), double(tolerance));
  constexpr amrex::Real indicator_threshold = amrex::Real(0.005);
  if (host_indicator[0] >= indicator_threshold ||
      host_indicator[1] <= indicator_threshold)
    CCTK_VERROR("ParticleAHFinderX angular cutoff-indicator self-test failed: "
                "smooth=%.17g cutoff=%.17g threshold=%.17g",
                double(host_indicator[0]), double(host_indicator[1]),
                double(indicator_threshold));
  CCTK_VINFO("Sixth-order angular self-test passed on the active AMReX "
             "backend: maximum error=%.3g smooth/cutoff indicators=%.3g/%.3g",
             double(host_error), double(host_indicator[0]),
             double(host_indicator[1]));
}

} // namespace ParticleAHFinderX
