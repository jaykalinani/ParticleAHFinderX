/**
 * \file intrinsic_geometry_self_test.cxx
 * \brief Accelerator convergence tests for intrinsic surface geometry.
 */
#include "intrinsic_geometry.hxx"
#include "runtime.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>

#include <array>
#include <limits>

namespace ParticleAHFinderX {
namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
absolute_value(const amrex::Real value) noexcept {
  return value < 0 ? -value : value;
}

struct IntrinsicSelfTestResult {
  amrex::Real point_geometry_error = 0;
  amrex::Real curvature_error = 0;
  amrex::Real gauss_bonnet_relative_error = 0;
  amrex::Real area_relative_error = 0;
  amrex::Long invalid = 0;
};

IntrinsicSelfTestResult run_intrinsic_self_test(const int ntheta,
                                                const amrex::Real epsilon) {
  const int nphi = 2 * ntheta;
  const amrex::Long point_count = amrex::Long(ntheta) * nphi;
  constexpr amrex::Real radius = amrex::Real(2.3);
  amrex::Gpu::DeviceVector<amrex::Real> metric(
      intrinsic_metric_components * point_count, 0);
  amrex::Gpu::DeviceVector<amrex::Real> connection(
      intrinsic_connection_components * point_count, 0);
  amrex::Gpu::DeviceVector<amrex::Real> curvature(point_count, 0);
  amrex::Gpu::DeviceVector<amrex::Real> compact(4, 0);
  amrex::Gpu::DeviceVector<amrex::Long> invalid(1, 0);
  amrex::Real *const q = metric.dataPtr();
  amrex::Real *const delta = connection.dataPtr();
  amrex::Real *const scalar_curvature = curvature.dataPtr();
  amrex::Real *const result = compact.dataPtr();
  amrex::Long *const invalid_count = invalid.dataPtr();

  amrex::ParallelFor(
      point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int itheta = static_cast<int>(point / nphi);
        const int iphi = static_cast<int>(point - amrex::Long(itheta) * nphi);
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (itheta + amrex::Real(0.5)) / ntheta;
        const amrex::Real phi =
            2 * amrex::Math::pi<amrex::Real>() *
            (iphi + amrex::Real(0.5)) / nphi;
        const auto sincos = amrex::Math::sincos(theta);
        const amrex::Real sine = sincos.first;
        const amrex::Real cosine = sincos.second;
        const amrex::Real p2 =
            (3 * cosine * cosine - 1) / amrex::Real(2);
        const amrex::Real height = radius * (1 + epsilon * p2);
        const amrex::Real h_theta =
            -3 * radius * epsilon * cosine * sine;
        const amrex::Real expected_metric[3]{
            height * height + h_theta * h_theta, 0,
            height * height * sine * sine};
        const amrex::Real expected_determinant =
            expected_metric[0] * expected_metric[2];

        IntrinsicGeometryInput input;
        input.gamma[0] = input.gamma[3] = input.gamma[5] = 1;
        input.gamma[1] = input.gamma[2] = input.gamma[4] = 0;
        for (int component = 0; component < 6; ++component)
          input.curv[component] = 0;
        const AngularGeometry basis = angular_geometry(theta, phi);
        const amrex::Real inverse_normal =
            amrex::Math::rsqrt(expected_metric[0]);
        for (int d = 0; d < 3; ++d)
          input.normal[d] =
              (height * basis.radial[d] -
               h_theta * basis.radial_theta[d]) *
              inverse_normal;
        input.height = height;
        input.h_theta = h_theta;
        input.h_phi = 0;
        input.theta = theta;
        input.phi = phi;

        IntrinsicGeometryOutput output;
        if (!evaluate_intrinsic_geometry(input, output)) {
          amrex::Gpu::Atomic::AddNoRet(invalid_count, amrex::Long(1));
          return;
        }
        amrex::Real point_error = 0;
        for (int component = 0; component < 3; ++component) {
          q[component * point_count + point] = output.metric[component];
          point_error = amrex::max(
              point_error,
              absolute_value(output.metric[component] -
                             expected_metric[component]));
        }
        point_error = amrex::max(
            point_error,
            absolute_value(output.sqrt_determinant -
                           expected_determinant *
                               amrex::Math::rsqrt(expected_determinant)));
        point_error =
            amrex::max(point_error,
                       absolute_value(output.normal_connection[0]));
        point_error =
            amrex::max(point_error,
                       absolute_value(output.normal_connection[1]));
        amrex::Gpu::Atomic::Max(result, point_error);
      });

  amrex::ParallelFor(
      point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int itheta = static_cast<int>(point / nphi);
        const int iphi = static_cast<int>(point - amrex::Long(itheta) * nphi);
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (itheta + amrex::Real(0.5)) / ntheta;
        amrex::Real local_delta[2][2][2];
        if (!intrinsic_connection_difference(
                q, point_count, 0, ntheta, nphi, itheta, iphi, theta,
                local_delta)) {
          amrex::Gpu::Atomic::AddNoRet(invalid_count, amrex::Long(1));
          return;
        }
        for (int upper = 0; upper < 2; ++upper)
          for (int first = 0; first < 2; ++first)
            for (int second = first; second < 2; ++second)
              delta[intrinsic_connection_component(upper, first, second) *
                        point_count +
                    point] = local_delta[upper][first][second];
      });

  amrex::ParallelFor(
      point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int itheta = static_cast<int>(point / nphi);
        const int iphi = static_cast<int>(point - amrex::Long(itheta) * nphi);
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (itheta + amrex::Real(0.5)) / ntheta;
        const auto sincos = amrex::Math::sincos(theta);
        const amrex::Real sine = sincos.first;
        const amrex::Real cosine = sincos.second;
        const amrex::Real p2 =
            (3 * cosine * cosine - 1) / amrex::Real(2);
        const amrex::Real height = radius * (1 + epsilon * p2);
        const amrex::Real h_theta =
            -3 * radius * epsilon * cosine * sine;
        const amrex::Real h_theta_theta =
            3 * radius * epsilon *
            (sine * sine - cosine * cosine);
        const amrex::Real induced_theta_theta =
            height * height + h_theta * h_theta;
        const amrex::Real sqrt_theta_theta =
            induced_theta_theta *
            amrex::Math::rsqrt(induced_theta_theta);
        const amrex::Real second_theta =
            (height * (h_theta_theta - height) -
             2 * h_theta * h_theta) /
            sqrt_theta_theta;
        const amrex::Real second_phi =
            (-height * height * sine * sine +
             h_theta * height * sine * cosine) /
            sqrt_theta_theta;
        const amrex::Real expected =
            2 * second_theta * second_phi /
            (induced_theta_theta * height * height * sine * sine);
        amrex::Real observed = 0;
        if (!intrinsic_scalar_curvature(q, delta, point_count, 0, ntheta,
                                        nphi, itheta, iphi, theta,
                                        observed)) {
          amrex::Gpu::Atomic::AddNoRet(invalid_count, amrex::Long(1));
          return;
        }
        scalar_curvature[point] = observed;
        const amrex::Real determinant =
            q[point] * q[2 * point_count + point] -
            q[point_count + point] * q[point_count + point];
        const amrex::Real sqrt_determinant =
            determinant * amrex::Math::rsqrt(determinant);
        const amrex::Real dtheta =
            amrex::Math::pi<amrex::Real>() / ntheta;
        const amrex::Real weight = surface_cell_area_weight(
            sqrt_determinant, dtheta,
            2 * amrex::Math::pi<amrex::Real>() / nphi);
        amrex::Gpu::Atomic::Max(
            result + 1, absolute_value(observed - expected));
        amrex::Gpu::Atomic::AddNoRet(result + 2, observed * weight);
        amrex::Gpu::Atomic::AddNoRet(result + 3, weight);
      });

  std::array<amrex::Real, 4> host{};
  amrex::Long host_invalid = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, compact.begin(), compact.end(),
                   host.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, invalid.begin(), invalid.end(),
                   &host_invalid);
  const amrex::Real eight_pi =
      8 * amrex::Math::pi<amrex::Real>();
  const amrex::Real exact_round_area =
      4 * amrex::Math::pi<amrex::Real>() * radius * radius;
  IntrinsicSelfTestResult answer;
  answer.point_geometry_error = host[0];
  answer.curvature_error = host[1];
  answer.gauss_bonnet_relative_error =
      absolute_value(host[2] - eight_pi) / eight_pi;
  answer.area_relative_error =
      epsilon == 0 ? absolute_value(host[3] - exact_round_area) /
                         exact_round_area
                   : 0;
  answer.invalid = host_invalid;
  return answer;
}

} // namespace

extern "C" void ParticleAHFinderX_IntrinsicGeometrySelfTest(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  if (!run_intrinsic_geometry_self_test ||
      state.intrinsic_geometry_self_test_complete)
    return;

  constexpr int resolution_count = 3;
  constexpr int resolutions[resolution_count]{16, 32, 64};
  constexpr amrex::Real distortions[2]{0, amrex::Real(0.12)};
  for (const amrex::Real distortion : distortions) {
    std::array<IntrinsicSelfTestResult, resolution_count> results;
    for (int level = 0; level < resolution_count; ++level) {
      results[level] =
          run_intrinsic_self_test(resolutions[level], distortion);
      CCTK_VINFO(
          "Intrinsic-geometry self-test: distortion=%.3g ntheta=%d "
          "point/curvature/Gauss-Bonnet/area errors="
          "%.3g/%.3g/%.3g/%.3g invalid=%lld",
          double(distortion), resolutions[level],
          double(results[level].point_geometry_error),
          double(results[level].curvature_error),
          double(results[level].gauss_bonnet_relative_error),
          double(results[level].area_relative_error),
          static_cast<long long>(results[level].invalid));
      if (results[level].invalid != 0)
        CCTK_VERROR("ParticleAHFinderX intrinsic-geometry self-test found "
                    "%lld invalid point operation(s)",
                    static_cast<long long>(results[level].invalid));
    }
    for (int level = 1; level < resolution_count; ++level)
      if (!(results[level].curvature_error <
            amrex::Real(0.2) * results[level - 1].curvature_error))
        CCTK_VERROR(
            "ParticleAHFinderX intrinsic curvature did not converge for "
            "distortion %.3g from ntheta=%d to %d: %.17g -> %.17g",
            double(distortion), resolutions[level - 1], resolutions[level],
            double(results[level - 1].curvature_error),
            double(results[level].curvature_error));
    const auto &finest = results.back();
    const amrex::Real point_tolerance = amrex::max(
        amrex::Real(1.0e-11),
        amrex::Real(100000) *
            std::numeric_limits<amrex::Real>::epsilon());
    if (finest.point_geometry_error > point_tolerance ||
        finest.curvature_error > amrex::Real(5.0e-4) ||
        finest.gauss_bonnet_relative_error > amrex::Real(5.0e-4) ||
        finest.area_relative_error > amrex::Real(5.0e-4))
      CCTK_VERROR(
          "ParticleAHFinderX intrinsic-geometry finest-grid guard failed "
          "for distortion %.3g: point/curvature/Gauss-Bonnet/area="
          "%.17g/%.17g/%.17g/%.17g",
          double(distortion), double(finest.point_geometry_error),
          double(finest.curvature_error),
          double(finest.gauss_bonnet_relative_error),
          double(finest.area_relative_error));
  }
  CCTK_INFO("Polar-regular intrinsic-geometry accelerator convergence "
            "self-test passed");
  state.intrinsic_geometry_self_test_complete = true;
}

} // namespace ParticleAHFinderX
