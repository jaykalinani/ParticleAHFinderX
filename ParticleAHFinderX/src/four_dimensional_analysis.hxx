/**
 * \file four_dimensional_analysis.hxx
 * \brief Portable four-dimensional horizon-analysis algebra.
 */
#ifndef PARTICLEAHFINDERX_FOUR_DIMENSIONAL_ANALYSIS_HXX
#define PARTICLEAHFINDERX_FOUR_DIMENSIONAL_ANALYSIS_HXX

#include "weyl_curvature.hxx"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <limits>

namespace ParticleAHFinderX {

/** Transient gauge payload produced by one fused mesh-owner gather. */
struct FourGaugeLayout {
  enum : int {
    lapse = 0,
    d_lapse = 1,
    dt_lapse = 4,
    shift = 5,
    d_shift = 8,
    dt_shift = 17,
    component_count = 20,
  };
};

/** Covariant T_ab payload in the native TmunuBaseX component order. */
struct FourMatterLayout {
  enum : int {
    tt = 0,
    tx,
    ty,
    tz,
    xx,
    xy,
    xz,
    yy,
    yz,
    zz,
    component_count,
  };
};

struct FourMetricInput {
  amrex::Real gamma[6]{};
  amrex::Real d_gamma[3][6]{};
  amrex::Real curv[6]{};
  amrex::Real lapse = 1;
  amrex::Real d_lapse[3]{};
  amrex::Real dt_lapse = 0;
  amrex::Real shift[3]{};
  // First index is the Cartesian derivative, second is the vector component.
  amrex::Real d_shift[3][3]{};
  amrex::Real dt_shift[3]{};
};

struct FourMetricData {
  amrex::Real covariant[4][4]{};
  amrex::Real contravariant[4][4]{};
  // derivative[coordinate][first tensor index][second tensor index]
  amrex::Real derivative[4][4][4]{};
};

struct NullTetrad {
  amrex::Real outgoing[4]{};
  amrex::Real inward[4]{};
  amrex::Real m_real[4]{};
  amrex::Real m_imag[4]{};
  amrex::Real normalization_error = 0;
};

struct MatterRicciInput {
  // 00, 01, 02, 03, 11, 12, 13, 22, 23, 33.
  amrex::Real stress_energy[10]{};
};

struct MatterRicciTensor {
  amrex::Real covariant[10]{};
  amrex::Real scalar = 0;
};

struct RicciNPScalars {
  amrex::Real phi00 = 0;
  amrex::Real phi11 = 0;
  amrex::Real phi22 = 0;
  ComplexValue phi01;
  ComplexValue phi12;
  ComplexValue phi02;
  amrex::Real lambda = 0;
};

/** Real component layout used by the compact surface Ricci transaction. */
struct RicciIntegralLayout {
  enum : int {
    phi00 = 0,
    phi01_real,
    phi01_imag,
    phi02_real,
    phi02_imag,
    phi11,
    phi12_real,
    phi12_imag,
    phi22,
    lambda,
    component_count,
  };
};

struct TetradCovariantDerivatives {
  // First index is lowered tetrad-vector index; second is derivative index.
  amrex::Real outgoing[4][4]{};
  amrex::Real inward[4][4]{};
  ComplexValue m[4][4]{};
};

enum NPSpinCoefficientIndex : int {
  np_kappa = 0,
  np_tau,
  np_sigma,
  np_rho,
  np_epsilon,
  np_gamma,
  np_beta,
  np_alpha,
  np_pi,
  np_nu,
  np_mu,
  np_lambda,
  np_spin_coefficient_count
};

struct NPSpinCoefficients {
  ComplexValue value[np_spin_coefficient_count]{};
};

struct WeinbergFluxIntegrand {
  amrex::Real energy = 0;
  amrex::Real momentum[3]{};
  amrex::Real angular_momentum[3]{};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
four_absolute_value(const amrex::Real value) noexcept {
  return value < 0 ? -value : value;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
four_maximum(const amrex::Real first, const amrex::Real second) noexcept {
  return first > second ? first : second;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
four_symmetric_component(int first, int second) noexcept {
  if (first > second) {
    const int temporary = first;
    first = second;
    second = temporary;
  }
  if (first == 0)
    return second;
  if (first == 1)
    return 3 + second;
  if (first == 2)
    return 5 + second;
  return 9;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
four_symmetric_value(const amrex::Real values[10], const int first,
                     const int second) noexcept {
  return values[four_symmetric_component(first, second)];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
evaluate_four_metric(const FourMetricInput &input,
                     FourMetricData &output) noexcept {
  if (!(input.lapse > 0) || !amrex::Math::isfinite(input.lapse))
    return false;
  amrex::Real inverse_gamma[3][3];
  if (!inverse_metric(input.gamma, inverse_gamma))
    return false;

  amrex::Real shift_lower[3]{};
  amrex::Real shift_squared = 0;
  for (int i = 0; i < 3; ++i) {
    if (!amrex::Math::isfinite(input.shift[i]) ||
        !amrex::Math::isfinite(input.dt_shift[i]) ||
        !amrex::Math::isfinite(input.d_lapse[i]))
      return false;
    for (int j = 0; j < 3; ++j)
      shift_lower[i] +=
          symmetric_value(input.gamma, i, j) * input.shift[j];
    shift_squared += shift_lower[i] * input.shift[i];
  }
  if (!amrex::Math::isfinite(input.dt_lapse) ||
      !amrex::Math::isfinite(shift_squared))
    return false;

  output.covariant[0][0] =
      -input.lapse * input.lapse + shift_squared;
  output.contravariant[0][0] =
      -1 / (input.lapse * input.lapse);
  for (int i = 0; i < 3; ++i) {
    output.covariant[0][i + 1] = shift_lower[i];
    output.covariant[i + 1][0] = shift_lower[i];
    output.contravariant[0][i + 1] =
        input.shift[i] / (input.lapse * input.lapse);
    output.contravariant[i + 1][0] =
        output.contravariant[0][i + 1];
    for (int j = 0; j < 3; ++j) {
      output.covariant[i + 1][j + 1] =
          symmetric_value(input.gamma, i, j);
      output.contravariant[i + 1][j + 1] =
          inverse_gamma[i][j] - input.shift[i] * input.shift[j] /
                                    (input.lapse * input.lapse);
    }
  }

  amrex::Real dt_gamma[3][3]{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      amrex::Real value =
          -2 * input.lapse * symmetric_value(input.curv, i, j);
      for (int k = 0; k < 3; ++k)
        value += input.shift[k] *
                     input.d_gamma[k][symmetric_component(i, j)] +
                 symmetric_value(input.gamma, i, k) *
                     input.d_shift[j][k] +
                 symmetric_value(input.gamma, k, j) *
                     input.d_shift[i][k];
      dt_gamma[i][j] = value;
    }

  for (int derivative = 0; derivative < 4; ++derivative) {
    const amrex::Real d_lapse =
        derivative == 0 ? input.dt_lapse : input.d_lapse[derivative - 1];
    amrex::Real d_shift[3];
    for (int i = 0; i < 3; ++i)
      d_shift[i] = derivative == 0
                       ? input.dt_shift[i]
                       : input.d_shift[derivative - 1][i];

    amrex::Real d_shift_squared = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        const amrex::Real d_gamma =
            derivative == 0
                ? dt_gamma[i][j]
                : input.d_gamma[derivative - 1]
                               [symmetric_component(i, j)];
        d_shift_squared += d_gamma * input.shift[i] * input.shift[j] +
                           2 * symmetric_value(input.gamma, i, j) *
                               input.shift[i] * d_shift[j];
      }
    output.derivative[derivative][0][0] =
        -2 * input.lapse * d_lapse + d_shift_squared;
    for (int i = 0; i < 3; ++i) {
      amrex::Real d_shift_lower = 0;
      for (int j = 0; j < 3; ++j) {
        const amrex::Real d_gamma =
            derivative == 0
                ? dt_gamma[i][j]
                : input.d_gamma[derivative - 1]
                               [symmetric_component(i, j)];
        d_shift_lower += d_gamma * input.shift[j] +
                         symmetric_value(input.gamma, i, j) * d_shift[j];
      }
      output.derivative[derivative][0][i + 1] = d_shift_lower;
      output.derivative[derivative][i + 1][0] = d_shift_lower;
      for (int j = 0; j < 3; ++j)
        output.derivative[derivative][i + 1][j + 1] =
            derivative == 0
                ? dt_gamma[i][j]
                : input.d_gamma[derivative - 1]
                               [symmetric_component(i, j)];
    }
  }

  for (int first = 0; first < 4; ++first)
    for (int second = 0; second < 4; ++second) {
      if (!amrex::Math::isfinite(output.covariant[first][second]) ||
          !amrex::Math::isfinite(output.contravariant[first][second]))
        return false;
      for (int derivative = 0; derivative < 4; ++derivative)
        if (!amrex::Math::isfinite(
                output.derivative[derivative][first][second]))
          return false;
    }
  return true;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
four_metric_dot(const amrex::Real metric[4][4], const amrex::Real first[4],
                const amrex::Real second[4]) noexcept {
  amrex::Real value = 0;
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b)
      value += metric[a][b] * first[a] * second[b];
  return value;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
evaluate_null_tetrad(const FourMetricInput &input,
                     const FourMetricData &metric,
                     const SurfaceTriad &triad,
                     NullTetrad &output) noexcept {
  if (!(input.lapse > 0) || !amrex::Math::isfinite(input.lapse))
    return false;
  const amrex::Real inverse_sqrt_two =
      amrex::Math::rsqrt(amrex::Real(2));
  amrex::Real slice_normal[4]{1 / input.lapse, 0, 0, 0};
  amrex::Real surface_normal[4]{};
  output.m_real[0] = 0;
  output.m_imag[0] = 0;
  for (int i = 0; i < 3; ++i) {
    slice_normal[i + 1] = -input.shift[i] / input.lapse;
    surface_normal[i + 1] = triad.radial[i];
    output.m_real[i + 1] =
        inverse_sqrt_two * triad.azimuthal[i];
    output.m_imag[i + 1] = inverse_sqrt_two * triad.polar[i];
  }
  for (int a = 0; a < 4; ++a) {
    output.outgoing[a] =
        inverse_sqrt_two * (slice_normal[a] + surface_normal[a]);
    output.inward[a] =
        inverse_sqrt_two * (slice_normal[a] - surface_normal[a]);
  }

  amrex::Real error = 0;
  error = four_maximum(error, four_absolute_value(four_metric_dot(
                                  metric.covariant, output.outgoing,
                                  output.outgoing)));
  error = four_maximum(error, four_absolute_value(four_metric_dot(
                                  metric.covariant, output.inward,
                                  output.inward)));
  error = four_maximum(
      error, four_absolute_value(four_metric_dot(metric.covariant,
                                                 output.outgoing,
                                                 output.inward) +
                                 1));
  const amrex::Real m_real_norm = four_metric_dot(
      metric.covariant, output.m_real, output.m_real);
  const amrex::Real m_imag_norm = four_metric_dot(
      metric.covariant, output.m_imag, output.m_imag);
  const amrex::Real m_cross = four_metric_dot(
      metric.covariant, output.m_real, output.m_imag);
  error = four_maximum(
      error, four_absolute_value(m_real_norm + m_imag_norm - 1));
  error = four_maximum(error,
                       four_absolute_value(m_real_norm - m_imag_norm));
  error = four_maximum(error, four_absolute_value(2 * m_cross));
  const amrex::Real *const null_vectors[2]{output.outgoing, output.inward};
  for (int null_index = 0; null_index < 2; ++null_index) {
    error = four_maximum(
        error, four_absolute_value(four_metric_dot(
                   metric.covariant, null_vectors[null_index],
                   output.m_real)));
    error = four_maximum(
        error, four_absolute_value(four_metric_dot(
                   metric.covariant, null_vectors[null_index],
                   output.m_imag)));
  }
  output.normalization_error = error;
  const amrex::Real tolerance =
      amrex::Real(4096) * std::numeric_limits<amrex::Real>::epsilon();
  return amrex::Math::isfinite(error) && error <= tolerance;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
evaluate_matter_ricci(const MatterRicciInput &input,
                      const FourMetricData &metric,
                      MatterRicciTensor &output) noexcept {
  amrex::Real trace = 0;
  for (int first = 0; first < 4; ++first)
    for (int second = 0; second < 4; ++second) {
      const amrex::Real stress =
          four_symmetric_value(input.stress_energy, first, second);
      if (!amrex::Math::isfinite(stress))
        return false;
      trace += metric.contravariant[first][second] * stress;
    }
  if (!amrex::Math::isfinite(trace))
    return false;
  const amrex::Real eight_pi =
      8 * amrex::Math::pi<amrex::Real>();
  for (int first = 0; first < 4; ++first)
    for (int second = first; second < 4; ++second) {
      const int component = four_symmetric_component(first, second);
      output.covariant[component] =
          eight_pi *
          (input.stress_energy[component] -
           amrex::Real(0.5) * metric.covariant[first][second] * trace);
      if (!amrex::Math::isfinite(output.covariant[component]))
        return false;
    }
  output.scalar = -eight_pi * trace;
  return amrex::Math::isfinite(output.scalar);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
four_tensor_projection(const amrex::Real tensor[10],
                       const amrex::Real first[4],
                       const amrex::Real second[4]) noexcept {
  amrex::Real value = 0;
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b)
      value += first[a] * four_symmetric_value(tensor, a, b) * second[b];
  return value;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
project_matter_ricci(const MatterRicciTensor &ricci,
                     const NullTetrad &tetrad,
                     RicciNPScalars &output) noexcept {
  output.phi00 = -amrex::Real(0.5) *
                 four_tensor_projection(ricci.covariant, tetrad.outgoing,
                                        tetrad.outgoing);
  output.phi22 = -amrex::Real(0.5) *
                 four_tensor_projection(ricci.covariant, tetrad.inward,
                                        tetrad.inward);
  const amrex::Real outgoing_inward = four_tensor_projection(
      ricci.covariant, tetrad.outgoing, tetrad.inward);
  const amrex::Real m_mbar =
      four_tensor_projection(ricci.covariant, tetrad.m_real,
                             tetrad.m_real) +
      four_tensor_projection(ricci.covariant, tetrad.m_imag,
                             tetrad.m_imag);
  output.phi11 = -amrex::Real(0.25) * (outgoing_inward + m_mbar);
  output.phi01 = {
      -amrex::Real(0.5) * four_tensor_projection(
                              ricci.covariant, tetrad.outgoing,
                              tetrad.m_real),
      -amrex::Real(0.5) * four_tensor_projection(
                              ricci.covariant, tetrad.outgoing,
                              tetrad.m_imag)};
  output.phi12 = {
      -amrex::Real(0.5) * four_tensor_projection(
                              ricci.covariant, tetrad.inward,
                              tetrad.m_real),
      -amrex::Real(0.5) * four_tensor_projection(
                              ricci.covariant, tetrad.inward,
                              tetrad.m_imag)};
  output.phi02 = {
      -amrex::Real(0.5) *
          (four_tensor_projection(ricci.covariant, tetrad.m_real,
                                  tetrad.m_real) -
           four_tensor_projection(ricci.covariant, tetrad.m_imag,
                                  tetrad.m_imag)),
      -four_tensor_projection(ricci.covariant, tetrad.m_real,
                              tetrad.m_imag)};
  output.lambda = ricci.scalar / 24;
  return amrex::Math::isfinite(output.phi00) &&
         amrex::Math::isfinite(output.phi11) &&
         amrex::Math::isfinite(output.phi22) &&
         complex_is_finite(output.phi01) &&
         complex_is_finite(output.phi12) &&
         complex_is_finite(output.phi02) &&
         amrex::Math::isfinite(output.lambda);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ComplexValue
four_complex_conjugate(const ComplexValue value) noexcept {
  return {value.real, -value.imag};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ComplexValue
complex_derivative_projection(const ComplexValue derivative[4][4],
                              const ComplexValue first[4],
                              const ComplexValue direction[4]) noexcept {
  ComplexValue value;
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b)
      value = complex_add(
          value, complex_multiply(
                     complex_multiply(first[a], direction[b]),
                     derivative[a][b]));
  return value;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
evaluate_np_spin_coefficients(
    const NullTetrad &tetrad,
    const TetradCovariantDerivatives &derivatives,
    NPSpinCoefficients &output) noexcept {
  ComplexValue outgoing[4];
  ComplexValue inward[4];
  ComplexValue m[4];
  ComplexValue mbar[4];
  ComplexValue nabla_outgoing[4][4];
  ComplexValue nabla_inward[4][4];
  for (int a = 0; a < 4; ++a) {
    outgoing[a] = {tetrad.outgoing[a], 0};
    inward[a] = {tetrad.inward[a], 0};
    m[a] = {tetrad.m_real[a], tetrad.m_imag[a]};
    mbar[a] = four_complex_conjugate(m[a]);
    for (int b = 0; b < 4; ++b) {
      nabla_outgoing[a][b] = {derivatives.outgoing[a][b], 0};
      nabla_inward[a][b] = {derivatives.inward[a][b], 0};
    }
  }

  output.value[np_kappa] = complex_scale(complex_derivative_projection(
                                             nabla_outgoing, m, outgoing),
                                         -1);
  output.value[np_tau] = complex_scale(complex_derivative_projection(
                                           nabla_outgoing, m, inward),
                                       -1);
  output.value[np_sigma] = complex_scale(complex_derivative_projection(
                                             nabla_outgoing, m, m),
                                         -1);
  output.value[np_rho] = complex_scale(complex_derivative_projection(
                                           nabla_outgoing, m, mbar),
                                       -1);
  output.value[np_epsilon] = complex_scale(
      complex_subtract(
          complex_derivative_projection(nabla_outgoing, inward, outgoing),
          complex_derivative_projection(derivatives.m, mbar, outgoing)),
      -amrex::Real(0.5));
  output.value[np_gamma] = complex_scale(
      complex_subtract(
          complex_derivative_projection(nabla_outgoing, inward, inward),
          complex_derivative_projection(derivatives.m, mbar, inward)),
      -amrex::Real(0.5));
  output.value[np_beta] = complex_scale(
      complex_subtract(
          complex_derivative_projection(nabla_outgoing, inward, m),
          complex_derivative_projection(derivatives.m, mbar, m)),
      -amrex::Real(0.5));
  output.value[np_alpha] = complex_scale(
      complex_subtract(
          complex_derivative_projection(nabla_outgoing, inward, mbar),
          complex_derivative_projection(derivatives.m, mbar, mbar)),
      -amrex::Real(0.5));
  output.value[np_pi] =
      complex_derivative_projection(nabla_inward, mbar, outgoing);
  output.value[np_nu] =
      complex_derivative_projection(nabla_inward, mbar, inward);
  output.value[np_mu] =
      complex_derivative_projection(nabla_inward, mbar, m);
  output.value[np_lambda] =
      complex_derivative_projection(nabla_inward, mbar, mbar);
  for (int coefficient = 0; coefficient < np_spin_coefficient_count;
       ++coefficient)
    if (!complex_is_finite(output.value[coefficient]))
      return false;
  return true;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
evaluate_lie_inward_outgoing_expansion(
    const ComplexValue sigma, const ComplexValue lambda,
    const ComplexValue psi2, const amrex::Real ricci_lambda,
    amrex::Real &value) noexcept {
  value = 2 * complex_multiply(sigma, lambda).real + 2 * psi2.real +
          4 * ricci_lambda;
  return amrex::Math::isfinite(value);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
evaluate_weinberg_flux_integrand(const FourMetricData &metric,
                                 const amrex::Real position[3],
                                 const amrex::Real normal[3],
                                 WeinbergFluxIntegrand &output) noexcept {
  const amrex::Real normalization =
      -1 / (16 * amrex::Math::pi<amrex::Real>());
  amrex::Real energy = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      energy +=
          (metric.derivative[i + 1][j + 1][j + 1] -
           metric.derivative[j + 1][i + 1][j + 1]) *
          normal[i];
  output.energy = normalization * energy;

  for (int component = 0; component < 3; ++component) {
    amrex::Real momentum = 0;
    for (int contracted = 0; contracted < 3; ++contracted) {
      momentum +=
          (-metric.derivative[0][contracted + 1][contracted + 1] +
           metric.derivative[contracted + 1][contracted + 1][0]) *
          normal[component];
      momentum +=
          (-metric.derivative[contracted + 1][component + 1][0] +
           metric.derivative[0][contracted + 1][component + 1]) *
          normal[contracted];
    }
    output.momentum[component] = normalization * momentum;
  }

  amrex::Real angular_tensor[3][3]{};
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      for (int contracted = 0; contracted < 3; ++contracted)
        angular_tensor[first][second] += normal[contracted] *
            (-position[first] *
                 metric.derivative[contracted + 1][0][second + 1] +
             position[second] *
                 metric.derivative[contracted + 1][0][first + 1] +
             position[first] *
                 metric.derivative[0][second + 1][contracted + 1] -
             position[second] *
                 metric.derivative[0][first + 1][contracted + 1] +
             metric.covariant[0][second + 1] *
                 (contracted == first ? 1 : 0) -
             metric.covariant[0][first + 1] *
                 (contracted == second ? 1 : 0));
  output.angular_momentum[0] = normalization * angular_tensor[1][2];
  output.angular_momentum[1] = normalization * angular_tensor[2][0];
  output.angular_momentum[2] = normalization * angular_tensor[0][1];

  if (!amrex::Math::isfinite(output.energy))
    return false;
  for (int component = 0; component < 3; ++component)
    if (!amrex::Math::isfinite(output.momentum[component]) ||
        !amrex::Math::isfinite(output.angular_momentum[component]))
      return false;
  return true;
}

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_FOUR_DIMENSIONAL_ANALYSIS_HXX
