/**
 * \file interpolation_self_test.cxx
 * \brief Optional accelerator test of interpolation values and derivatives.
 */
#include "interpolation.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ParticleAHFinderX {
namespace {

constexpr int num_test_points = 16;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
linear_polynomial(const amrex::Real x, const amrex::Real y,
                  const amrex::Real z) noexcept {
  return 1 + 2 * x - 3 * y + z / 2;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
cubic_polynomial(const amrex::Real x, const amrex::Real y,
                 const amrex::Real z) noexcept {
  return 1 + x / 2 + y / 4 - 3 * z / 4 + x * x / 10 - y * y / 5 +
         z * z / 20 + x * x * x / 100 - 3 * y * y * y / 200 +
         z * z * z / 50 + 3 * x * y / 100 - y * z / 50 + x * z / 25 +
         x * y * z / 200;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, 3>
cubic_gradient(const amrex::Real x, const amrex::Real y,
               const amrex::Real z) noexcept {
  return {{amrex::Real(1) / 2 + x / 5 + 3 * x * x / 100 + 3 * y / 100 +
               z / 25 + y * z / 200,
           amrex::Real(1) / 4 - 2 * y / 5 - 9 * y * y / 200 + 3 * x / 100 -
               z / 50 + x * z / 200,
           -amrex::Real(3) / 4 + z / 10 + 3 * z * z / 50 - y / 50 + x / 25 +
               x * y / 200}};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, 3>
linear_gradient() noexcept {
  return {{2, -3, amrex::Real(1) / 2}};
}

void test_centering(const FieldCentering centering, const bool cubic,
                    amrex::Real &maximum_value_error,
                    amrex::Real &maximum_gradient_error) {
  const amrex::Box box(amrex::IntVect(0, 0, 0), amrex::IntVect(7, 7, 7));
  const amrex::GpuArray<amrex::Real, 3> prob_lo{{-1, -1, -1}};
  const amrex::GpuArray<amrex::Real, 3> dx{{amrex::Real(1) / 4,
                                            amrex::Real(1) / 5,
                                            amrex::Real(1) / 6}};
  const amrex::GpuArray<amrex::Real, 3> inv_dx{{1 / dx[0], 1 / dx[1],
                                                1 / dx[2]}};
  constexpr int num_adm_components = 12;
  amrex::Gpu::DeviceVector<amrex::Real> field_storage(
      num_adm_components * box.numPts());
  amrex::Array4<amrex::Real> field(field_storage.dataPtr(), box,
                                   num_adm_components);
  const auto mutable_field = field;
  amrex::ParallelFor(
      box, [=] AMREX_GPU_DEVICE(const int i, const int j, const int k) noexcept {
        const amrex::Real x =
            prob_lo[0] + (i + (centering.nodal[0] ? 0 : amrex::Real(0.5))) *
                             dx[0];
        const amrex::Real y =
            prob_lo[1] + (j + (centering.nodal[1] ? 0 : amrex::Real(0.5))) *
                             dx[1];
        const amrex::Real z =
            prob_lo[2] + (k + (centering.nodal[2] ? 0 : amrex::Real(0.5))) *
                             dx[2];
        const amrex::Real polynomial =
            cubic ? cubic_polynomial(x, y, z) : linear_polynomial(x, y, z);
        for (int component = 0; component < 6; ++component) {
          const amrex::Real scale = component + 1;
          mutable_field(i, j, k, component) =
              scale * polynomial + amrex::Real(component) / 8;
          mutable_field(i, j, k, 6 + component) =
              -scale * polynomial / 2 + amrex::Real(component) / 10;
        }
      });

  std::vector<amrex::Real> host_position(3 * num_test_points);
  for (int point = 0; point < num_test_points; ++point) {
    const amrex::GpuArray<amrex::Real, 3> logical{{
        amrex::Real(1.1) + amrex::Real(1.1) * (point % 4),
        amrex::Real(1.2) + amrex::Real(1.05) * ((point / 4) % 4),
        amrex::Real(1.3) + amrex::Real(1.0) * ((point * 3) % 4),
    }};
    for (int d = 0; d < 3; ++d)
      host_position[3 * point + d] =
          prob_lo[d] +
          (logical[d] + (centering.nodal[d] ? 0 : amrex::Real(0.5))) * dx[d];
  }

  amrex::Gpu::DeviceVector<amrex::Real> device_position(host_position.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_position.begin(),
                   host_position.end(), device_position.begin());
  amrex::Gpu::DeviceVector<amrex::Real> device_result(4 * num_test_points);
  amrex::Gpu::DeviceVector<int> device_success(num_test_points);
  amrex::Gpu::DeviceVector<amrex::Real> device_adm_result(
      30 * num_test_points);
  amrex::Gpu::DeviceVector<int> device_adm_status(num_test_points);
  const amrex::Array4<const amrex::Real> const_field(field);
  const amrex::Real *const position = device_position.dataPtr();
  amrex::Real *const result = device_result.dataPtr();
  int *const success = device_success.dataPtr();
  amrex::Real *const adm_result = device_adm_result.dataPtr();
  int *const adm_status = device_adm_status.dataPtr();
  amrex::ParallelFor(
      num_test_points,
      [=] AMREX_GPU_DEVICE(const int point) noexcept {
        const amrex::GpuArray<amrex::Real, 3> sample_position{{
            position[3 * point], position[3 * point + 1],
            position[3 * point + 2],
        }};
        amrex::Real value = std::numeric_limits<amrex::Real>::quiet_NaN();
        amrex::GpuArray<amrex::Real, 3> gradient{{0, 0, 0}};
        success[point] =
            cubic ? tensor_cubic_interpolate(
                        const_field, 0, sample_position, prob_lo, inv_dx,
                        centering, box, value, gradient)
                  : trilinear_interpolate(const_field, 0, sample_position,
                                          prob_lo, inv_dx, centering, box,
                                          value);
        result[4 * point] = value;
        for (int d = 0; d < 3; ++d)
          result[4 * point + 1 + d] = gradient[d];

        AdmInterpolationData adm;
        const InterpolationStatus status =
            cubic ? tensor_cubic_interpolate_adm(
                        const_field, 0, const_field, 6, sample_position,
                        prob_lo, inv_dx, centering, box, box, adm)
                  : trilinear_interpolate_adm(
                        const_field, 0, const_field, 6, sample_position,
                        prob_lo, inv_dx, centering, box, box, adm);
        adm_status[point] = static_cast<int>(status);
        for (int component = 0; component < 6; ++component) {
          adm_result[30 * point + component] = adm.gamma[component];
          for (int d = 0; d < 3; ++d)
            adm_result[30 * point + 6 + 6 * d + component] =
                adm.d_gamma[d][component];
          adm_result[30 * point + 24 + component] = adm.curv[component];
        }
      });

  std::vector<amrex::Real> host_result(4 * num_test_points);
  std::vector<int> host_success(num_test_points);
  std::vector<amrex::Real> host_adm_result(30 * num_test_points);
  std::vector<int> host_adm_status(num_test_points);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_result.begin(),
                   device_result.end(), host_result.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_success.begin(),
                   device_success.end(), host_success.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_adm_result.begin(),
                   device_adm_result.end(), host_adm_result.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_adm_status.begin(),
                   device_adm_status.end(), host_adm_status.begin());
  amrex::Gpu::streamSynchronize();

  for (int point = 0; point < num_test_points; ++point) {
    if (!host_success[point])
      CCTK_ERROR("ParticleAHFinderX interpolation self-test missed an "
                 "available stencil");
    const amrex::Real x = host_position[3 * point];
    const amrex::Real y = host_position[3 * point + 1];
    const amrex::Real z = host_position[3 * point + 2];
    const amrex::Real expected =
        cubic ? cubic_polynomial(x, y, z) : linear_polynomial(x, y, z);
    maximum_value_error =
        std::max(maximum_value_error,
                 std::abs(host_result[4 * point] - expected));
    const auto expected_gradient =
        cubic ? cubic_gradient(x, y, z) : linear_gradient();
    if (cubic)
      for (int d = 0; d < 3; ++d)
        maximum_gradient_error =
            std::max(maximum_gradient_error,
                     std::abs(host_result[4 * point + 1 + d] -
                              expected_gradient[d]));

    if (host_adm_status[point] !=
        static_cast<int>(InterpolationStatus::success))
      CCTK_ERROR("ParticleAHFinderX fused ADM interpolation self-test "
                 "missed an available stencil");
    for (int component = 0; component < 6; ++component) {
      const amrex::Real scale = component + 1;
      maximum_value_error = std::max(
          maximum_value_error,
          std::abs(host_adm_result[30 * point + component] -
                   (scale * expected + amrex::Real(component) / 8)));
      maximum_value_error = std::max(
          maximum_value_error,
          std::abs(host_adm_result[30 * point + 24 + component] -
                   (-scale * expected / 2 + amrex::Real(component) / 10)));
      for (int d = 0; d < 3; ++d)
        maximum_gradient_error = std::max(
            maximum_gradient_error,
            std::abs(host_adm_result[30 * point + 6 + 6 * d + component] -
                     scale * expected_gradient[d]));
    }
  }

  if (cubic) {
    amrex::Gpu::DeviceVector<int> device_rejected(1);
    int *const rejected = device_rejected.dataPtr();
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(const int) noexcept {
      const amrex::GpuArray<amrex::Real, 3> position_near_edge{{
          prob_lo[0] + (amrex::Real(0.25) +
                        (centering.nodal[0] ? 0 : amrex::Real(0.5))) *
                           dx[0],
          prob_lo[1] + (amrex::Real(2.25) +
                        (centering.nodal[1] ? 0 : amrex::Real(0.5))) *
                           dx[1],
          prob_lo[2] + (amrex::Real(2.25) +
                        (centering.nodal[2] ? 0 : amrex::Real(0.5))) *
                           dx[2],
      }};
      amrex::Real value = 0;
      amrex::GpuArray<amrex::Real, 3> gradient;
      rejected[0] = !tensor_cubic_interpolate(
          const_field, 0, position_near_edge, prob_lo, inv_dx, centering, box,
          value, gradient);
    });
    int host_rejected = 0;
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_rejected.begin(),
                     device_rejected.end(), &host_rejected);
    amrex::Gpu::streamSynchronize();
    if (!host_rejected)
      CCTK_ERROR("ParticleAHFinderX cubic interpolation self-test silently "
                 "accepted a missing stencil");
  }
}

} // namespace

extern "C" void ParticleAHFinderX_InterpolationSelfTest(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  if (!run_interpolation_self_test)
    return;

  amrex::Real maximum_value_error = 0;
  amrex::Real maximum_gradient_error = 0;
  for (const int nodal : {0, 1}) {
    const FieldCentering centering{{{nodal, nodal, nodal}}};
    test_centering(centering, false, maximum_value_error,
                   maximum_gradient_error);
    test_centering(centering, true, maximum_value_error,
                   maximum_gradient_error);
  }
  const amrex::Real tolerance =
      5000 * std::numeric_limits<amrex::Real>::epsilon();
  if (maximum_value_error > tolerance ||
      maximum_gradient_error > tolerance)
    CCTK_VERROR("ParticleAHFinderX interpolation self-test failed: value "
                "error=%.17g gradient error=%.17g tolerance=%.17g",
                double(maximum_value_error), double(maximum_gradient_error),
                double(tolerance));
  CCTK_VINFO("Interpolation self-test passed on the active AMReX backend: "
             "value error=%.3g gradient error=%.3g",
             double(maximum_value_error), double(maximum_gradient_error));
}

} // namespace ParticleAHFinderX
