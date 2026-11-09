/**
 * \file four_dimensional_analysis_self_test.cxx
 * \brief Accelerator tests for four-dimensional analysis algebra.
 */
#include "four_dimensional_analysis.hxx"
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

struct FourDimensionalAnalysisSelfTestResult {
  amrex::Real metric_error = 0;
  amrex::Real tetrad_error = 0;
  amrex::Real ricci_error = 0;
  amrex::Real np_error = 0;
  amrex::Real weinberg_error = 0;
  amrex::Long invalid = 0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
record_error(amrex::Real *const destination, const amrex::Real observed,
             const amrex::Real expected = 0) noexcept {
  amrex::Gpu::Atomic::Max(destination,
                         four_absolute_value(observed - expected));
}

FourDimensionalAnalysisSelfTestResult
execute_four_dimensional_analysis_self_test() {
  amrex::Gpu::DeviceVector<amrex::Real> errors(5, 0);
  amrex::Gpu::DeviceVector<amrex::Long> invalid(1, 0);
  amrex::Real *const error = errors.dataPtr();
  amrex::Long *const invalid_count = invalid.dataPtr();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(const int) noexcept {
    FourMetricInput metric_input;
    metric_input.gamma[symmetric_component(0, 0)] = 1;
    metric_input.gamma[symmetric_component(1, 1)] = 1;
    metric_input.gamma[symmetric_component(2, 2)] = 1;
    metric_input.lapse = amrex::Real(1.7);
    metric_input.shift[0] = amrex::Real(0.1);
    metric_input.shift[1] = amrex::Real(-0.2);
    metric_input.shift[2] = amrex::Real(0.3);
    FourMetricData metric;
    if (!evaluate_four_metric(metric_input, metric)) {
      invalid_count[0] = 1;
      return;
    }
    for (int first = 0; first < 4; ++first)
      for (int second = 0; second < 4; ++second) {
        amrex::Real product = 0;
        for (int contracted = 0; contracted < 4; ++contracted)
          product += metric.covariant[first][contracted] *
                     metric.contravariant[contracted][second];
        record_error(error, product, first == second ? 1 : 0);
      }

    SurfaceTriad triad;
    triad.radial[0] = 1;
    triad.polar[1] = 1;
    triad.azimuthal[2] = 1;
    NullTetrad tetrad;
    if (!evaluate_null_tetrad(metric_input, metric, triad, tetrad)) {
      invalid_count[0] = 1;
      return;
    }
    record_error(error + 1, tetrad.normalization_error);

    FourMetricInput derivative_input;
    derivative_input.gamma[symmetric_component(0, 0)] = 1;
    derivative_input.gamma[symmetric_component(1, 1)] = 1;
    derivative_input.gamma[symmetric_component(2, 2)] = 1;
    derivative_input.curv[symmetric_component(0, 0)] =
        amrex::Real(0.3);
    derivative_input.lapse = amrex::Real(1.2);
    derivative_input.dt_lapse = amrex::Real(0.07);
    FourMetricData derivative_metric;
    if (!evaluate_four_metric(derivative_input, derivative_metric)) {
      invalid_count[0] = 1;
      return;
    }
    record_error(error,
                 derivative_metric.derivative[0][1][1],
                 amrex::Real(-0.72));
    record_error(error,
                 derivative_metric.derivative[0][0][0],
                 amrex::Real(-0.168));

    // Verify all four derivative directions against an independent centered
    // difference of the ADM four-metric construction.  The step is close to
    // the cube root of double precision epsilon, balancing cancellation and
    // the second-order finite-difference remainder.
    FourMetricInput manufactured;
    manufactured.gamma[symmetric_component(0, 0)] = amrex::Real(1.4);
    manufactured.gamma[symmetric_component(0, 1)] = amrex::Real(0.08);
    manufactured.gamma[symmetric_component(0, 2)] = amrex::Real(-0.04);
    manufactured.gamma[symmetric_component(1, 1)] = amrex::Real(1.2);
    manufactured.gamma[symmetric_component(1, 2)] = amrex::Real(0.03);
    manufactured.gamma[symmetric_component(2, 2)] = amrex::Real(0.9);
    manufactured.curv[symmetric_component(0, 0)] = amrex::Real(0.07);
    manufactured.curv[symmetric_component(0, 1)] = amrex::Real(-0.02);
    manufactured.curv[symmetric_component(0, 2)] = amrex::Real(0.01);
    manufactured.curv[symmetric_component(1, 1)] = amrex::Real(-0.04);
    manufactured.curv[symmetric_component(1, 2)] = amrex::Real(0.015);
    manufactured.curv[symmetric_component(2, 2)] = amrex::Real(0.03);
    manufactured.lapse = amrex::Real(1.15);
    manufactured.dt_lapse = amrex::Real(-0.06);
    for (int d = 0; d < 3; ++d) {
      manufactured.shift[d] = amrex::Real(0.04) * (d + 1);
      manufactured.d_lapse[d] = amrex::Real(0.013) * (d + 1);
      manufactured.dt_shift[d] = amrex::Real(-0.009) * (d + 1);
      for (int component = 0; component < 6; ++component)
        manufactured.d_gamma[d][component] =
            amrex::Real(0.002) * (d + 1) * (component + 1);
      for (int component = 0; component < 3; ++component)
        manufactured.d_shift[d][component] =
            amrex::Real(0.004) * (d + 1) * (component + 1);
    }
    FourMetricData manufactured_metric;
    if (!evaluate_four_metric(manufactured, manufactured_metric)) {
      invalid_count[0] = 1;
      return;
    }
    amrex::Real dt_gamma[3][3]{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        dt_gamma[i][j] =
            -2 * manufactured.lapse *
            symmetric_value(manufactured.curv, i, j);
        for (int k = 0; k < 3; ++k)
          dt_gamma[i][j] +=
              manufactured.shift[k] *
                  manufactured.d_gamma[k][symmetric_component(i, j)] +
              symmetric_value(manufactured.gamma, i, k) *
                  manufactured.d_shift[j][k] +
              symmetric_value(manufactured.gamma, k, j) *
                  manufactured.d_shift[i][k];
      }
    const amrex::Real derivative_step = amrex::Real(6.0e-6);
    for (int derivative = 0; derivative < 4; ++derivative) {
      FourMetricInput plus = manufactured;
      FourMetricInput minus = manufactured;
      const amrex::Real lapse_derivative =
          derivative == 0 ? manufactured.dt_lapse
                          : manufactured.d_lapse[derivative - 1];
      plus.lapse += derivative_step * lapse_derivative;
      minus.lapse -= derivative_step * lapse_derivative;
      for (int i = 0; i < 3; ++i) {
        const amrex::Real shift_derivative =
            derivative == 0
                ? manufactured.dt_shift[i]
                : manufactured.d_shift[derivative - 1][i];
        plus.shift[i] += derivative_step * shift_derivative;
        minus.shift[i] -= derivative_step * shift_derivative;
        for (int j = i; j < 3; ++j) {
          const int component = symmetric_component(i, j);
          const amrex::Real gamma_derivative =
              derivative == 0
                  ? dt_gamma[i][j]
                  : manufactured.d_gamma[derivative - 1][component];
          plus.gamma[component] += derivative_step * gamma_derivative;
          minus.gamma[component] -= derivative_step * gamma_derivative;
        }
      }
      FourMetricData plus_metric;
      FourMetricData minus_metric;
      if (!evaluate_four_metric(plus, plus_metric) ||
          !evaluate_four_metric(minus, minus_metric)) {
        invalid_count[0] = 1;
        return;
      }
      for (int first = 0; first < 4; ++first)
        for (int second = 0; second < 4; ++second) {
          const amrex::Real finite_difference =
              (plus_metric.covariant[first][second] -
               minus_metric.covariant[first][second]) /
              (2 * derivative_step);
          record_error(error,
                       manufactured_metric.derivative[derivative][first]
                                                       [second],
                       finite_difference);
        }
    }

    FourMetricInput flat_input;
    flat_input.gamma[symmetric_component(0, 0)] = 1;
    flat_input.gamma[symmetric_component(1, 1)] = 1;
    flat_input.gamma[symmetric_component(2, 2)] = 1;
    FourMetricData flat_metric;
    NullTetrad flat_tetrad;
    if (!evaluate_four_metric(flat_input, flat_metric) ||
        !evaluate_null_tetrad(flat_input, flat_metric, triad,
                              flat_tetrad)) {
      invalid_count[0] = 1;
      return;
    }

    constexpr amrex::Real density = 2;
    constexpr amrex::Real pressure_x = amrex::Real(0.3);
    constexpr amrex::Real pressure_y = amrex::Real(0.5);
    constexpr amrex::Real pressure_z = amrex::Real(0.7);
    MatterRicciInput matter;
    matter.stress_energy[four_symmetric_component(0, 0)] = density;
    matter.stress_energy[four_symmetric_component(1, 1)] = pressure_x;
    matter.stress_energy[four_symmetric_component(2, 2)] = pressure_y;
    matter.stress_energy[four_symmetric_component(3, 3)] = pressure_z;
    MatterRicciTensor ricci;
    RicciNPScalars projections;
    if (!evaluate_matter_ricci(matter, flat_metric, ricci) ||
        !project_matter_ricci(ricci, flat_tetrad, projections)) {
      invalid_count[0] = 1;
      return;
    }
    const amrex::Real pi = amrex::Math::pi<amrex::Real>();
    const amrex::Real stress_trace =
        -density + pressure_x + pressure_y + pressure_z;
    const amrex::Real ricci00 =
        8 * pi * (density + amrex::Real(0.5) * stress_trace);
    const amrex::Real riccixx =
        8 * pi * (pressure_x - amrex::Real(0.5) * stress_trace);
    const amrex::Real ricciyy =
        8 * pi * (pressure_y - amrex::Real(0.5) * stress_trace);
    const amrex::Real riccizz =
        8 * pi * (pressure_z - amrex::Real(0.5) * stress_trace);
    record_error(error + 2, projections.phi00,
                 -amrex::Real(0.25) * (ricci00 + riccixx));
    record_error(error + 2, projections.phi22,
                 -amrex::Real(0.25) * (ricci00 + riccixx));
    record_error(error + 2, projections.phi11,
                 -amrex::Real(0.125) *
                     (ricci00 - riccixx + ricciyy + riccizz));
    record_error(error + 2, projections.phi02.real,
                 -amrex::Real(0.25) * (riccizz - ricciyy));
    record_error(error + 2, projections.phi02.imag);
    record_error(error + 2, projections.phi01.real);
    record_error(error + 2, projections.phi01.imag);
    record_error(error + 2, projections.phi12.real);
    record_error(error + 2, projections.phi12.imag);
    record_error(error + 2, projections.lambda,
                 -8 * pi * stress_trace / 24);

    TetradCovariantDerivatives tetrad_derivatives;
    NPSpinCoefficients coefficients;
    tetrad_derivatives.outgoing[3][0] = 2;
    if (!evaluate_np_spin_coefficients(flat_tetrad, tetrad_derivatives,
                                       coefficients)) {
      invalid_count[0] = 1;
      return;
    }
    record_error(error + 3, coefficients.value[np_kappa].real, -1);
    record_error(error + 3, coefficients.value[np_tau].real, -1);
    for (int coefficient = 0; coefficient < np_spin_coefficient_count;
         ++coefficient) {
      if (coefficient != np_kappa && coefficient != np_tau)
        record_error(error + 3, coefficients.value[coefficient].real);
      record_error(error + 3, coefficients.value[coefficient].imag);
    }

    tetrad_derivatives = {};
    tetrad_derivatives.outgoing[3][3] = 2;
    tetrad_derivatives.outgoing[1][0] = 2;
    tetrad_derivatives.outgoing[1][3] = 2;
    if (!evaluate_np_spin_coefficients(flat_tetrad, tetrad_derivatives,
                                       coefficients)) {
      invalid_count[0] = 1;
      return;
    }
    record_error(error + 3, coefficients.value[np_sigma].real, -1);
    record_error(error + 3, coefficients.value[np_rho].real, -1);
    record_error(error + 3, coefficients.value[np_epsilon].real,
                 amrex::Real(0.5));
    record_error(error + 3, coefficients.value[np_gamma].real,
                 amrex::Real(0.5));
    record_error(error + 3, coefficients.value[np_beta].real,
                 amrex::Real(0.5));
    record_error(error + 3, coefficients.value[np_alpha].real,
                 amrex::Real(0.5));

    tetrad_derivatives = {};
    tetrad_derivatives.inward[3][0] = 2;
    tetrad_derivatives.inward[3][3] = 2;
    if (!evaluate_np_spin_coefficients(flat_tetrad, tetrad_derivatives,
                                       coefficients)) {
      invalid_count[0] = 1;
      return;
    }
    record_error(error + 3, coefficients.value[np_pi].real, 1);
    record_error(error + 3, coefficients.value[np_nu].real, 1);
    record_error(error + 3, coefficients.value[np_mu].real, 1);
    record_error(error + 3, coefficients.value[np_lambda].real, 1);

    tetrad_derivatives = {};
    tetrad_derivatives.m[3][0] = {2, 4};
    tetrad_derivatives.m[3][3] = {2, 4};
    if (!evaluate_np_spin_coefficients(flat_tetrad, tetrad_derivatives,
                                       coefficients)) {
      invalid_count[0] = 1;
      return;
    }
    constexpr int complex_coefficients[4]{np_epsilon, np_gamma, np_beta,
                                           np_alpha};
    for (const int coefficient : complex_coefficients) {
      record_error(error + 3, coefficients.value[coefficient].real,
                   amrex::Real(0.5));
      record_error(error + 3, coefficients.value[coefficient].imag, 1);
    }

    const amrex::Real position[3]{amrex::Real(0.2), amrex::Real(-0.3),
                                  amrex::Real(0.4)};
    const amrex::Real normal[3]{1, 0, 0};
    WeinbergFluxIntegrand weinberg;
    if (!evaluate_weinberg_flux_integrand(flat_metric, position, normal,
                                          weinberg)) {
      invalid_count[0] = 1;
      return;
    }
    record_error(error + 4, weinberg.energy);
    for (int component = 0; component < 3; ++component) {
      record_error(error + 4, weinberg.momentum[component]);
      record_error(error + 4, weinberg.angular_momentum[component]);
    }
    FourMetricData energy_metric = flat_metric;
    energy_metric.derivative[1][2][2] = 4;
    if (!evaluate_weinberg_flux_integrand(energy_metric, position, normal,
                                          weinberg)) {
      invalid_count[0] = 1;
      return;
    }
    record_error(error + 4, weinberg.energy, -1 / (4 * pi));
    for (int component = 0; component < 3; ++component) {
      record_error(error + 4, weinberg.momentum[component]);
      record_error(error + 4, weinberg.angular_momentum[component]);
    }

    FourMetricData momentum_metric = flat_metric;
    momentum_metric.derivative[0][2][2] = 2;
    if (!evaluate_weinberg_flux_integrand(momentum_metric, position, normal,
                                          weinberg)) {
      invalid_count[0] = 1;
      return;
    }
    record_error(error + 4, weinberg.energy);
    record_error(error + 4, weinberg.momentum[0], 1 / (8 * pi));
    record_error(error + 4, weinberg.momentum[1]);
    record_error(error + 4, weinberg.momentum[2]);

    FourMetricData angular_metric = flat_metric;
    angular_metric.derivative[1][0][2] = 5;
    angular_metric.derivative[1][2][0] = 5;
    if (!evaluate_weinberg_flux_integrand(angular_metric, position, normal,
                                          weinberg)) {
      invalid_count[0] = 1;
      return;
    }
    record_error(error + 4, weinberg.energy);
    record_error(error + 4, weinberg.momentum[0]);
    record_error(error + 4, weinberg.momentum[1], 5 / (16 * pi));
    record_error(error + 4, weinberg.momentum[2]);
    // The manufactured y-momentum density is evaluated away from the global
    // Cartesian origin. Its orbital contribution is
    // (x cross P)_x = -z P_y = -0.4 * 5 / (16 pi).
    record_error(error + 4, weinberg.angular_momentum[0], -1 / (8 * pi));
    record_error(error + 4, weinberg.angular_momentum[1]);
    record_error(error + 4, weinberg.angular_momentum[2], 1 / (16 * pi));

    FourMetricInput invalid_metric = flat_input;
    invalid_metric.lapse = -1;
    FourMetricData ignored;
    if (evaluate_four_metric(invalid_metric, ignored))
      invalid_count[0] = 1;
  });

  std::array<amrex::Real, 5> host_errors{};
  amrex::Long host_invalid = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, errors.begin(), errors.end(),
                   host_errors.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, invalid.begin(), invalid.end(),
                   &host_invalid);
  return {host_errors[0], host_errors[1], host_errors[2], host_errors[3],
          host_errors[4], host_invalid};
}

} // namespace

extern "C" void
ParticleAHFinderX_FourDimensionalAnalysisSelfTest(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  if (!run_four_dimensional_analysis_self_test ||
      state.four_dimensional_analysis_self_test_complete)
    return;

  const FourDimensionalAnalysisSelfTestResult result =
      execute_four_dimensional_analysis_self_test();
  CCTK_VINFO("Four-dimensional analysis self-test: metric/tetrad/Ricci/NP/"
             "Weinberg errors=%.3g/%.3g/%.3g/%.3g/%.3g invalid=%lld",
             double(result.metric_error), double(result.tetrad_error),
             double(result.ricci_error), double(result.np_error),
             double(result.weinberg_error),
             static_cast<long long>(result.invalid));
  const amrex::Real tolerance =
      amrex::Real(1000000) * std::numeric_limits<amrex::Real>::epsilon();
  if (result.invalid != 0 || result.metric_error > tolerance ||
      result.tetrad_error > tolerance || result.ricci_error > tolerance ||
      result.np_error > tolerance || result.weinberg_error > tolerance)
    CCTK_VERROR(
        "ParticleAHFinderX four-dimensional analysis accelerator self-test "
        "failed: metric/tetrad/Ricci/NP/Weinberg="
        "%.17g/%.17g/%.17g/%.17g/%.17g invalid=%lld",
        double(result.metric_error), double(result.tetrad_error),
        double(result.ricci_error), double(result.np_error),
        double(result.weinberg_error),
        static_cast<long long>(result.invalid));
  CCTK_INFO("Four-dimensional horizon-analysis accelerator self-test passed");
  state.four_dimensional_analysis_self_test_complete = true;
}

} // namespace ParticleAHFinderX
