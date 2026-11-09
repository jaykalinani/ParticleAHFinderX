/**
 * \file weyl_curvature_self_test.cxx
 * \brief Accelerator tests for the native 3+1 Weyl-curvature helpers.
 */
#include "interpolation.hxx"
#include "runtime.hxx"
#include "weyl_curvature.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>

#include <array>
#include <limits>

namespace ParticleAHFinderX {
namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
weyl_absolute_value(const amrex::Real value) noexcept {
  return value < 0 ? -value : value;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
maximum_error(amrex::Real *const destination,
              const amrex::Real observed,
              const amrex::Real expected = 0) noexcept {
  amrex::Gpu::Atomic::Max(destination,
                         weyl_absolute_value(observed - expected));
}

struct WeylCurvatureSelfTestResult {
  amrex::Real minkowski_error = 0;
  amrex::Real schwarzschild_error = 0;
  amrex::Real projection_error = 0;
  amrex::Real interpolation_coarse_error = 0;
  amrex::Real interpolation_fine_error = 0;
  amrex::Long invalid = 0;
};

struct InterpolatedSchwarzschildResult {
  amrex::Real normalized_error = 0;
  bool valid = false;
};

InterpolatedSchwarzschildResult
interpolated_schwarzschild_error(const amrex::Real spacing) {
  constexpr amrex::Real mass = 1;
  const amrex::GpuArray<amrex::Real, 3> position{{3, amrex::Real(0.7),
                                                  amrex::Real(-0.4)}};
  const amrex::GpuArray<amrex::Real, 3> logical{{amrex::Real(3.37),
                                                 amrex::Real(3.11),
                                                 amrex::Real(2.83)}};
  amrex::GpuArray<amrex::Real, 3> prob_lo;
  amrex::GpuArray<amrex::Real, 3> inv_dx;
  for (int d = 0; d < 3; ++d) {
    prob_lo[d] = position[d] - logical[d] * spacing;
    inv_dx[d] = 1 / spacing;
  }
  const amrex::Box box(amrex::IntVect(0, 0, 0), amrex::IntVect(7, 7, 7));
  constexpr int field_components = 12;
  amrex::Gpu::DeviceVector<amrex::Real> field_storage(
      field_components * box.numPts());
  amrex::Array4<amrex::Real> field(field_storage.dataPtr(), box,
                                   field_components);
  const auto mutable_field = field;
  amrex::ParallelFor(
      box, [=] AMREX_GPU_DEVICE(const int i, const int j,
                               const int k) noexcept {
        const amrex::Real x = prob_lo[0] + i * spacing;
        const amrex::Real y = prob_lo[1] + j * spacing;
        const amrex::Real z = prob_lo[2] + k * spacing;
        const amrex::Real radius_squared = x * x + y * y + z * z;
        const amrex::Real inverse_radius =
            amrex::Math::rsqrt(radius_squared);
        const amrex::Real psi = 1 + mass * inverse_radius / 2;
        const amrex::Real conformal_factor = psi * psi * psi * psi;
        for (int component = 0; component < 6; ++component) {
          mutable_field(i, j, k, component) =
              component == symmetric_component(0, 0) ||
                      component == symmetric_component(1, 1) ||
                      component == symmetric_component(2, 2)
                  ? conformal_factor
                  : 0;
          mutable_field(i, j, k, 6 + component) = 0;
        }
      });

  amrex::Gpu::DeviceVector<amrex::Real> device_error(1, 0);
  amrex::Gpu::DeviceVector<int> device_valid(1, 0);
  amrex::Real *const error = device_error.dataPtr();
  int *const valid = device_valid.dataPtr();
  const amrex::Array4<const amrex::Real> const_field(field);
  const FieldCentering centering{{{1, 1, 1}}};
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(const int) noexcept {
    AdmInterpolationData adm;
    CurvatureDerivativeData derivatives;
    if (tensor_cubic_interpolate_adm(
            const_field, 0, const_field, 6, position, prob_lo, inv_dx,
            centering, box, box, adm) != InterpolationStatus::success ||
        tensor_cubic_interpolate_curvature_derivatives(
            const_field, 0, const_field, 6, position, prob_lo, inv_dx,
            centering, box, box, derivatives) !=
            InterpolationStatus::success)
      return;

    WeylCurvatureInput input;
    for (int component = 0; component < 6; ++component) {
      input.gamma[component] = adm.gamma[component];
      input.curv[component] = adm.curv[component];
      for (int d = 0; d < 3; ++d) {
        input.d_gamma[d][component] = adm.d_gamma[d][component];
        input.d_curv[d][component] = derivatives.d_curv[d][component];
      }
      for (int pair = 0; pair < 6; ++pair)
        input.dd_gamma[pair][component] =
            derivatives.dd_gamma[pair][component];
    }
    ElectricMagneticWeyl curvature;
    if (!evaluate_electric_magnetic_weyl(input, curvature))
      return;

    const amrex::Real radius_squared =
        position[0] * position[0] + position[1] * position[1] +
        position[2] * position[2];
    const amrex::Real inverse_radius =
        amrex::Math::rsqrt(radius_squared);
    const amrex::Real radius = radius_squared * inverse_radius;
    const amrex::Real psi = 1 + mass * inverse_radius / 2;
    const amrex::Real conformal_factor = psi * psi * psi * psi;
    const amrex::Real areal_radius = radius * psi * psi;
    const amrex::Real coulomb =
        mass / (areal_radius * areal_radius * areal_radius);
    const amrex::Real electric_scale = conformal_factor * coulomb;
    amrex::Real maximum_normalized_error =
        weyl_absolute_value(curvature.spatial_ricci_scalar) /
        electric_scale;
    for (int first = 0; first < 3; ++first)
      for (int second = first; second < 3; ++second) {
        const int component = symmetric_component(first, second);
        const amrex::Real radial_first =
            position[first] * inverse_radius;
        const amrex::Real radial_second =
            position[second] * inverse_radius;
        const amrex::Real expected =
            electric_scale *
            ((first == second ? amrex::Real(1) : amrex::Real(0)) -
             3 * radial_first * radial_second);
        maximum_normalized_error = amrex::max(
            maximum_normalized_error,
            weyl_absolute_value(curvature.electric[component] - expected) /
                electric_scale);
        maximum_normalized_error = amrex::max(
            maximum_normalized_error,
            weyl_absolute_value(curvature.magnetic[component]) /
                electric_scale);
      }
    if (!amrex::Math::isfinite(maximum_normalized_error))
      return;
    error[0] = maximum_normalized_error;
    valid[0] = 1;
  });

  amrex::Real host_error = 0;
  int host_valid = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_error.begin(),
                   device_error.end(), &host_error);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_valid.begin(),
                   device_valid.end(), &host_valid);
  amrex::Gpu::streamSynchronize();
  return {host_error, host_valid != 0};
}

WeylCurvatureSelfTestResult execute_weyl_curvature_self_test() {
  amrex::Gpu::DeviceVector<amrex::Real> errors(3, 0);
  amrex::Gpu::DeviceVector<amrex::Long> invalid(1, 0);
  amrex::Real *const error = errors.dataPtr();
  amrex::Long *const invalid_count = invalid.dataPtr();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(const int) noexcept {
    WeylCurvatureInput input;
    input.gamma[0] = input.gamma[3] = input.gamma[5] = 1;
    ElectricMagneticWeyl curvature;
    if (!evaluate_electric_magnetic_weyl(input, curvature)) {
      invalid_count[0] = 1;
      return;
    }
    for (int component = 0; component < 6; ++component) {
      maximum_error(error, curvature.electric[component]);
      maximum_error(error, curvature.magnetic[component]);
    }
    maximum_error(error, curvature.spatial_ricci_scalar);

    const amrex::Real flat_normal[3]{1, 0, 0};
    const amrex::Real flat_theta[3]{0, 0, 1};
    const amrex::Real flat_phi[3]{0, 1, 0};
    SurfaceTriad triad;
    WeylScalars scalars;
    if (!surface_triad(input.gamma, flat_normal, flat_theta, flat_phi,
                       triad) ||
        !weyl_scalars_from_electric_magnetic(curvature, triad, scalars) ||
        scalars.speciality_index_valid ||
        scalars.speciality_difference_valid) {
      invalid_count[0] = 1;
      return;
    }

    constexpr amrex::Real mass = 1;
    constexpr amrex::Real isotropic_radius = 3;
    const amrex::Real psi = 1 + mass / (2 * isotropic_radius);
    const amrex::Real psi_first =
        -mass / (2 * isotropic_radius * isotropic_radius);
    const amrex::Real psi_second =
        mass /
        (isotropic_radius * isotropic_radius * isotropic_radius);
    const amrex::Real conformal_factor = psi * psi * psi * psi;
    const amrex::Real conformal_first =
        4 * psi * psi * psi * psi_first;
    const amrex::Real conformal_second =
        12 * psi * psi * psi_first * psi_first +
        4 * psi * psi * psi * psi_second;
    input = WeylCurvatureInput{};
    input.gamma[0] = input.gamma[3] = input.gamma[5] = conformal_factor;
    for (int d = 0; d < 3; ++d) {
      const int component = symmetric_component(d, d);
      input.d_gamma[0][component] = conformal_first;
      input.dd_gamma[symmetric_component(0, 0)][component] =
          conformal_second;
      input.dd_gamma[symmetric_component(1, 1)][component] =
          conformal_first / isotropic_radius;
      input.dd_gamma[symmetric_component(2, 2)][component] =
          conformal_first / isotropic_radius;
    }
    curvature = ElectricMagneticWeyl{};
    if (!evaluate_electric_magnetic_weyl(input, curvature)) {
      invalid_count[0] = 1;
      return;
    }
    const amrex::Real areal_radius =
        isotropic_radius * psi * psi;
    const amrex::Real coulomb =
        mass / (areal_radius * areal_radius * areal_radius);
    maximum_error(error + 1, curvature.electric[0],
                  -2 * conformal_factor * coulomb);
    maximum_error(error + 1, curvature.electric[3],
                  conformal_factor * coulomb);
    maximum_error(error + 1, curvature.electric[5],
                  conformal_factor * coulomb);
    maximum_error(error + 1, curvature.spatial_ricci_scalar);
    maximum_error(error + 1, curvature.electric[1]);
    maximum_error(error + 1, curvature.electric[2]);
    maximum_error(error + 1, curvature.electric[4]);
    for (int component = 0; component < 6; ++component)
      maximum_error(error + 1, curvature.magnetic[component]);

    const amrex::Real normal[3]{
        amrex::Math::rsqrt(conformal_factor), 0, 0};
    const amrex::Real tangent_theta[3]{0, 0, -1};
    const amrex::Real tangent_phi[3]{0, 1, 0};
    if (!surface_triad(input.gamma, normal, tangent_theta, tangent_phi,
                       triad) ||
        !weyl_scalars_from_electric_magnetic(curvature, triad, scalars)) {
      invalid_count[0] = 1;
      return;
    }
    for (int scalar = 0; scalar < 5; ++scalar)
      if (scalar != 2) {
        maximum_error(error + 1, scalars.psi[scalar].real);
        maximum_error(error + 1, scalars.psi[scalar].imag);
      }
    maximum_error(error + 1, scalars.psi[2].real, -coulomb);
    maximum_error(error + 1, scalars.psi[2].imag);
    maximum_error(error + 1, scalars.invariant_i.real,
                  3 * coulomb * coulomb);
    maximum_error(error + 1, scalars.invariant_i.imag);
    maximum_error(error + 1, scalars.invariant_j.real,
                  coulomb * coulomb * coulomb);
    maximum_error(error + 1, scalars.invariant_j.imag);
    if (!scalars.speciality_index_valid ||
        !scalars.speciality_difference_valid) {
      invalid_count[0] = 1;
      return;
    }
    maximum_error(error + 1, scalars.speciality_index.real, 1);
    maximum_error(error + 1, scalars.speciality_index.imag);
    maximum_error(error + 1, scalars.speciality_difference.real);
    maximum_error(error + 1, scalars.speciality_difference.imag);

    curvature = ElectricMagneticWeyl{};
    constexpr amrex::Real electric_diagonal = amrex::Real(0.7);
    constexpr amrex::Real electric_cross = amrex::Real(-0.2);
    constexpr amrex::Real magnetic_cross = amrex::Real(0.3);
    constexpr amrex::Real magnetic_diagonal = amrex::Real(-0.4);
    curvature.electric[symmetric_component(1, 1)] = electric_diagonal;
    curvature.electric[symmetric_component(2, 2)] = -electric_diagonal;
    curvature.electric[symmetric_component(1, 2)] = electric_cross;
    curvature.magnetic[symmetric_component(1, 1)] = magnetic_diagonal;
    curvature.magnetic[symmetric_component(2, 2)] = -magnetic_diagonal;
    curvature.magnetic[symmetric_component(1, 2)] = magnetic_cross;
    const amrex::Real flat_metric[6]{1, 0, 0, 1, 0, 1};
    if (!surface_triad(flat_metric, flat_normal, flat_theta, flat_phi,
                       triad) ||
        !weyl_scalars_from_electric_magnetic(curvature, triad, scalars)) {
      invalid_count[0] = 1;
      return;
    }
    const ComplexValue expected_psi0{
        electric_diagonal + magnetic_cross,
        electric_cross - magnetic_diagonal};
    const ComplexValue expected_psi4{
        electric_diagonal - magnetic_cross,
        -electric_cross - magnetic_diagonal};
    const ComplexValue expected_i =
        complex_multiply(expected_psi0, expected_psi4);
    maximum_error(error + 2, scalars.psi[0].real, expected_psi0.real);
    maximum_error(error + 2, scalars.psi[0].imag, expected_psi0.imag);
    maximum_error(error + 2, scalars.psi[4].real, expected_psi4.real);
    maximum_error(error + 2, scalars.psi[4].imag, expected_psi4.imag);
    for (int scalar = 1; scalar <= 3; ++scalar) {
      maximum_error(error + 2, scalars.psi[scalar].real);
      maximum_error(error + 2, scalars.psi[scalar].imag);
    }
    maximum_error(error + 2, scalars.invariant_i.real, expected_i.real);
    maximum_error(error + 2, scalars.invariant_i.imag, expected_i.imag);
    maximum_error(error + 2, scalars.invariant_j.real);
    maximum_error(error + 2, scalars.invariant_j.imag);
    if (!scalars.speciality_index_valid ||
        !scalars.speciality_difference_valid) {
      invalid_count[0] = 1;
      return;
    }
    maximum_error(error + 2, scalars.speciality_index.real);
    maximum_error(error + 2, scalars.speciality_index.imag);
    const ComplexValue expected_speciality_difference =
        scalars.speciality_difference;
    for (int scale_case = 0; scale_case < 2; ++scale_case) {
      const amrex::Real scale =
          scale_case == 0 ? amrex::Real(1.0e-90) : amrex::Real(1.0e90);
      ElectricMagneticWeyl scaled_curvature;
      for (int component = 0; component < 6; ++component) {
        scaled_curvature.electric[component] =
            scale * curvature.electric[component];
        scaled_curvature.magnetic[component] =
            scale * curvature.magnetic[component];
      }
      WeylScalars scaled_scalars;
      if (!weyl_scalars_from_electric_magnetic(scaled_curvature, triad,
                                                scaled_scalars) ||
          !scaled_scalars.speciality_index_valid ||
          !scaled_scalars.speciality_difference_valid) {
        invalid_count[0] = 1;
        return;
      }
      maximum_error(error + 2, scaled_scalars.speciality_index.real);
      maximum_error(error + 2, scaled_scalars.speciality_index.imag);
      maximum_error(error + 2,
                    scaled_scalars.speciality_difference.real,
                    expected_speciality_difference.real);
      maximum_error(error + 2,
                    scaled_scalars.speciality_difference.imag,
                    expected_speciality_difference.imag);
    }
  });

  std::array<amrex::Real, 3> host_errors{};
  amrex::Long host_invalid = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, errors.begin(), errors.end(),
                   host_errors.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, invalid.begin(), invalid.end(),
                   &host_invalid);
  const auto coarse = interpolated_schwarzschild_error(amrex::Real(0.2));
  const auto fine = interpolated_schwarzschild_error(amrex::Real(0.1));
  host_invalid += !coarse.valid;
  host_invalid += !fine.valid;
  return {host_errors[0], host_errors[1], host_errors[2],
          coarse.normalized_error, fine.normalized_error, host_invalid};
}

} // namespace

extern "C" void ParticleAHFinderX_WeylCurvatureSelfTest(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  if (!run_weyl_curvature_self_test ||
      state.weyl_curvature_self_test_complete)
    return;

  const WeylCurvatureSelfTestResult result =
      execute_weyl_curvature_self_test();
  CCTK_VINFO("Weyl-curvature self-test: Minkowski/Schwarzschild/projection "
             "errors=%.3g/%.3g/%.3g interpolation coarse/fine=%.3g/%.3g "
             "invalid=%lld",
             double(result.minkowski_error),
             double(result.schwarzschild_error),
             double(result.projection_error),
             double(result.interpolation_coarse_error),
             double(result.interpolation_fine_error),
             static_cast<long long>(result.invalid));
  const amrex::Real tolerance =
      amrex::Real(1000000) * std::numeric_limits<amrex::Real>::epsilon();
  if (result.invalid != 0 || result.minkowski_error > tolerance ||
      result.schwarzschild_error > tolerance ||
      result.projection_error > tolerance ||
      !(result.interpolation_coarse_error > 0) ||
      !(result.interpolation_fine_error <
        amrex::Real(0.35) * result.interpolation_coarse_error) ||
      !(result.interpolation_fine_error < amrex::Real(0.03)))
    CCTK_VERROR(
        "ParticleAHFinderX Weyl-curvature accelerator self-test failed: "
        "Minkowski/Schwarzschild/projection=%.17g/%.17g/%.17g "
        "interpolation coarse/fine=%.17g/%.17g invalid=%lld",
        double(result.minkowski_error),
        double(result.schwarzschild_error),
        double(result.projection_error),
        double(result.interpolation_coarse_error),
        double(result.interpolation_fine_error),
        static_cast<long long>(result.invalid));
  CCTK_INFO("Native 3+1 Weyl-curvature accelerator self-test passed");
  state.weyl_curvature_self_test_complete = true;
}

} // namespace ParticleAHFinderX
