/**
 * \file weyl_curvature.hxx
 * \brief Portable 3+1 Weyl curvature at a horizon point.
 *
 * The sign conventions are
 *
 *   K_ij = -(1/2) L_n gamma_ij,
 *   E_ij = C_{i a j b} n^a n^b,
 *   B_ij = (1/2) epsilon_{i a k l} C^{k l}_{j b} n^a n^b,
 *
 * with an outward spatial unit normal s^i and
 * m^a=(e_phi^a+i e_theta^a)/sqrt(2).  The Newman--Penrose scalars use the
 * direct contractions C_abcd l^a m^b l^c m^d, ..., without an additional
 * leading minus sign.
 */
#ifndef PARTICLEAHFINDERX_WEYL_CURVATURE_HXX
#define PARTICLEAHFINDERX_WEYL_CURVATURE_HXX

#include "expansion.hxx"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

namespace ParticleAHFinderX {

struct WeylCurvatureInput {
  amrex::Real gamma[6]{};
  amrex::Real d_gamma[3][6]{};
  // The first index is the symmetric Cartesian derivative pair.
  amrex::Real dd_gamma[6][6]{};
  amrex::Real curv[6]{};
  amrex::Real d_curv[3][6]{};
  amrex::Real spatial_stress[6]{};
  bool include_matter = false;
};

struct ElectricMagneticWeyl {
  amrex::Real electric[6]{};
  amrex::Real magnetic[6]{};
  amrex::Real spatial_ricci_scalar = 0;
};

struct SurfaceTriad {
  amrex::Real radial[3]{};
  amrex::Real polar[3]{};
  amrex::Real azimuthal[3]{};
};

struct ComplexValue {
  amrex::Real real = 0;
  amrex::Real imag = 0;
};

struct WeylScalars {
  ComplexValue psi[5];
  ComplexValue invariant_i;
  ComplexValue invariant_j;
  ComplexValue speciality_index;
  ComplexValue speciality_difference;
  bool speciality_index_valid = false;
  bool speciality_difference_valid = false;
};

/** Real/imaginary pairs in the compact area-integral record. */
struct WeylIntegralLayout {
  static constexpr int psi = 0;
  static constexpr int invariant_i = 10;
  static constexpr int invariant_j = 12;
  static constexpr int speciality_index = 14;
  static constexpr int speciality_difference = 16;
  static constexpr int component_count = 18;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ComplexValue
complex_add(const ComplexValue first, const ComplexValue second) noexcept {
  return {first.real + second.real, first.imag + second.imag};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ComplexValue
complex_subtract(const ComplexValue first,
                 const ComplexValue second) noexcept {
  return {first.real - second.real, first.imag - second.imag};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ComplexValue
complex_scale(const ComplexValue value, const amrex::Real scale) noexcept {
  return {scale * value.real, scale * value.imag};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ComplexValue
complex_multiply(const ComplexValue first,
                 const ComplexValue second) noexcept {
  return {first.real * second.real - first.imag * second.imag,
          first.real * second.imag + first.imag * second.real};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
complex_norm_squared(const ComplexValue value) noexcept {
  return value.real * value.real + value.imag * value.imag;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
complex_maximum_absolute_component(const ComplexValue value) noexcept {
  const amrex::Real real = value.real < 0 ? -value.real : value.real;
  const amrex::Real imag = value.imag < 0 ? -value.imag : value.imag;
  return real > imag ? real : imag;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
complex_is_finite(const ComplexValue value) noexcept {
  return amrex::Math::isfinite(value.real) &&
         amrex::Math::isfinite(value.imag);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
complex_divide(const ComplexValue numerator, const ComplexValue denominator,
               ComplexValue &result) noexcept {
  if (!complex_is_finite(numerator) || !complex_is_finite(denominator))
    return false;
  const amrex::Real real = denominator.real < 0 ? -denominator.real
                                                : denominator.real;
  const amrex::Real imag = denominator.imag < 0 ? -denominator.imag
                                                : denominator.imag;
  if (real >= imag) {
    if (!(real > 0))
      return false;
    const amrex::Real ratio = denominator.imag / denominator.real;
    const amrex::Real divisor = denominator.real + denominator.imag * ratio;
    result = {(numerator.real + numerator.imag * ratio) / divisor,
              (numerator.imag - numerator.real * ratio) / divisor};
  } else {
    const amrex::Real ratio = denominator.real / denominator.imag;
    const amrex::Real divisor = denominator.imag + denominator.real * ratio;
    result = {(numerator.real * ratio + numerator.imag) / divisor,
              (numerator.imag * ratio - numerator.real) / divisor};
  }
  return complex_is_finite(result);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
weyl_invariants(const ComplexValue psi[5], ComplexValue &invariant_i,
                ComplexValue &invariant_j) noexcept {
  const ComplexValue psi2_squared = complex_multiply(psi[2], psi[2]);
  invariant_i = complex_add(
      complex_subtract(
          complex_scale(psi2_squared, 3),
          complex_scale(complex_multiply(psi[1], psi[3]), 4)),
      complex_multiply(psi[0], psi[4]));
  invariant_j = complex_subtract(
      complex_add(
          complex_add(
              complex_scale(complex_multiply(psi2_squared, psi[2]), -1),
              complex_multiply(complex_multiply(psi[0], psi[2]), psi[4])),
          complex_scale(
              complex_multiply(complex_multiply(psi[1], psi[2]), psi[3]),
              2)),
      complex_add(
          complex_multiply(complex_multiply(psi[1], psi[1]), psi[4]),
          complex_multiply(psi[0], complex_multiply(psi[3], psi[3]))));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
alternating_symbol(const int first, const int second,
                   const int third) noexcept {
  if (first == second || first == third || second == third)
    return 0;
  return (first == 0 && second == 1 && third == 2) ||
                 (first == 1 && second == 2 && third == 0) ||
                 (first == 2 && second == 0 && third == 1)
             ? 1
             : -1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
second_metric_derivative(const WeylCurvatureInput &input,
                         const int first_derivative,
                         const int second_derivative, const int first_metric,
                         const int second_metric) noexcept {
  return input.dd_gamma[symmetric_component(first_derivative,
                                             second_derivative)]
                       [symmetric_component(first_metric, second_metric)];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
evaluate_electric_magnetic_weyl(const WeylCurvatureInput &input,
                                ElectricMagneticWeyl &output) noexcept {
  amrex::Real inverse[3][3];
  const amrex::Real determinant = metric_determinant(input.gamma);
  if (!inverse_metric(input.gamma, inverse) || !(determinant > 0) ||
      !amrex::Math::isfinite(determinant))
    return false;

  amrex::Real inverse_derivative[3][3][3];
  for (int derivative = 0; derivative < 3; ++derivative)
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second) {
        amrex::Real value = 0;
        for (int lower_first = 0; lower_first < 3; ++lower_first)
          for (int lower_second = 0; lower_second < 3; ++lower_second)
            value -= inverse[first][lower_first] *
                     inverse[second][lower_second] *
                     input.d_gamma[derivative][symmetric_component(
                         lower_first, lower_second)];
        inverse_derivative[derivative][first][second] = value;
      }

  amrex::Real connection[3][3][3];
  amrex::Real connection_derivative[3][3][3][3];
  for (int upper = 0; upper < 3; ++upper)
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second) {
        amrex::Real value = 0;
        for (int contracted = 0; contracted < 3; ++contracted)
          value += amrex::Real(0.5) * inverse[upper][contracted] *
                   (input.d_gamma[first][symmetric_component(contracted,
                                                               second)] +
                    input.d_gamma[second][symmetric_component(contracted,
                                                                first)] -
                    input.d_gamma[contracted]
                                 [symmetric_component(first, second)]);
        connection[upper][first][second] = value;

        for (int derivative = 0; derivative < 3; ++derivative) {
          amrex::Real derivative_value = 0;
          for (int contracted = 0; contracted < 3; ++contracted) {
            const amrex::Real first_derivative =
                input.d_gamma[first]
                             [symmetric_component(contracted, second)] +
                input.d_gamma[second]
                             [symmetric_component(contracted, first)] -
                input.d_gamma[contracted]
                             [symmetric_component(first, second)];
            const amrex::Real second_derivative =
                second_metric_derivative(input, derivative, first,
                                         contracted, second) +
                second_metric_derivative(input, derivative, second,
                                         contracted, first) -
                second_metric_derivative(input, derivative, contracted,
                                         first, second);
            derivative_value +=
                amrex::Real(0.5) *
                (inverse_derivative[derivative][upper][contracted] *
                     first_derivative +
                 inverse[upper][contracted] * second_derivative);
          }
          connection_derivative[derivative][upper][first][second] =
              derivative_value;
        }
      }

  amrex::Real ricci[3][3];
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second) {
      amrex::Real value = 0;
      for (int contracted = 0; contracted < 3; ++contracted) {
        value += connection_derivative[contracted][contracted][first]
                                      [second] -
                 connection_derivative[second][contracted][first]
                                      [contracted];
        for (int other = 0; other < 3; ++other)
          value += connection[contracted][first][second] *
                       connection[other][contracted][other] -
                   connection[contracted][first][other] *
                       connection[other][second][contracted];
      }
      ricci[first][second] = value;
    }

  amrex::Real trace_curvature = 0;
  amrex::Real ricci_scalar = 0;
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second) {
      trace_curvature +=
          inverse[first][second] * symmetric_value(input.curv, first, second);
      ricci_scalar += inverse[first][second] * ricci[first][second];
    }
  output.spatial_ricci_scalar = ricci_scalar;

  amrex::Real electric[3][3];
  amrex::Real covariant_curvature_derivative[3][3][3];
  for (int derivative = 0; derivative < 3; ++derivative)
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second) {
        amrex::Real value =
            input.d_curv[derivative][symmetric_component(first, second)];
        for (int contracted = 0; contracted < 3; ++contracted)
          value -= connection[contracted][derivative][first] *
                       symmetric_value(input.curv, contracted, second) +
                   connection[contracted][derivative][second] *
                       symmetric_value(input.curv, first, contracted);
        covariant_curvature_derivative[derivative][first][second] = value;
      }

  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second) {
      amrex::Real curvature_product = 0;
      for (int lower_first = 0; lower_first < 3; ++lower_first)
        for (int lower_second = 0; lower_second < 3; ++lower_second)
          curvature_product +=
              symmetric_value(input.curv, first, lower_first) *
              inverse[lower_first][lower_second] *
              symmetric_value(input.curv, lower_second, second);
      electric[first][second] =
          ricci[first][second] +
          trace_curvature * symmetric_value(input.curv, first, second) -
          curvature_product;
      if (input.include_matter)
        electric[first][second] -=
            4 * amrex::Math::pi<amrex::Real>() *
            symmetric_value(input.spatial_stress, first, second);
    }

  const amrex::Real inverse_sqrt_determinant =
      amrex::Math::rsqrt(determinant);
  amrex::Real magnetic[3][3];
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second) {
      amrex::Real value = 0;
      for (int raised = 0; raised < 3; ++raised)
        for (int derivative = 0; derivative < 3; ++derivative)
          for (int lower = 0; lower < 3; ++lower)
            value += symmetric_value(input.gamma, first, raised) *
                     alternating_symbol(raised, derivative, lower) *
                     inverse_sqrt_determinant *
                     covariant_curvature_derivative[derivative][lower]
                                                   [second];
      magnetic[first][second] = value;
    }

  amrex::Real electric_trace = 0;
  amrex::Real magnetic_trace = 0;
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second) {
      electric_trace += inverse[first][second] * electric[first][second];
      magnetic_trace +=
          inverse[first][second] *
          amrex::Real(0.5) *
          (magnetic[first][second] + magnetic[second][first]);
    }

  for (int first = 0; first < 3; ++first)
    for (int second = first; second < 3; ++second) {
      const int component = symmetric_component(first, second);
      output.electric[component] =
          amrex::Real(0.5) *
              (electric[first][second] + electric[second][first]) -
          input.gamma[component] * electric_trace / 3;
      output.magnetic[component] =
          amrex::Real(0.5) *
              (magnetic[first][second] + magnetic[second][first]) -
          input.gamma[component] * magnetic_trace / 3;
      if (!amrex::Math::isfinite(output.electric[component]) ||
          !amrex::Math::isfinite(output.magnetic[component]))
        return false;
    }
  return amrex::Math::isfinite(output.spatial_ricci_scalar);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
normalize_metric_vector(const amrex::Real metric[6],
                        amrex::Real vector[3]) noexcept {
  const amrex::Real norm_squared = metric_dot(metric, vector, vector);
  if (!(norm_squared > 0) || !amrex::Math::isfinite(norm_squared))
    return false;
  const amrex::Real inverse_norm = amrex::Math::rsqrt(norm_squared);
  for (int d = 0; d < 3; ++d)
    vector[d] *= inverse_norm;
  return true;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
surface_triad(const amrex::Real metric[6], const amrex::Real normal[3],
              const amrex::Real tangent_theta[3],
              const amrex::Real tangent_phi[3], SurfaceTriad &triad) noexcept {
  for (int d = 0; d < 3; ++d) {
    triad.radial[d] = normal[d];
    triad.azimuthal[d] = tangent_phi[d];
    triad.polar[d] = tangent_theta[d];
  }
  if (!normalize_metric_vector(metric, triad.radial))
    return false;

  const amrex::Real radial_phi =
      metric_dot(metric, triad.radial, triad.azimuthal);
  for (int d = 0; d < 3; ++d)
    triad.azimuthal[d] -= radial_phi * triad.radial[d];
  if (!normalize_metric_vector(metric, triad.azimuthal))
    return false;

  const amrex::Real radial_theta =
      metric_dot(metric, triad.radial, triad.polar);
  const amrex::Real phi_theta =
      metric_dot(metric, triad.azimuthal, triad.polar);
  for (int d = 0; d < 3; ++d)
    triad.polar[d] -= radial_theta * triad.radial[d] +
                      phi_theta * triad.azimuthal[d];
  return normalize_metric_vector(metric, triad.polar);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
tensor_projection(const amrex::Real tensor[6], const amrex::Real first[3],
                  const amrex::Real second[3]) noexcept {
  amrex::Real value = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      value += first[i] * symmetric_value(tensor, i, j) * second[j];
  return value;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
weyl_scalars_from_electric_magnetic(const ElectricMagneticWeyl &curvature,
                                    const SurfaceTriad &triad,
                                    WeylScalars &output) noexcept {
  const amrex::Real ess = tensor_projection(
      curvature.electric, triad.radial, triad.radial);
  const amrex::Real esp = tensor_projection(
      curvature.electric, triad.radial, triad.azimuthal);
  const amrex::Real esq =
      tensor_projection(curvature.electric, triad.radial, triad.polar);
  const amrex::Real epp = tensor_projection(
      curvature.electric, triad.azimuthal, triad.azimuthal);
  const amrex::Real epq = tensor_projection(
      curvature.electric, triad.azimuthal, triad.polar);
  const amrex::Real eqq =
      tensor_projection(curvature.electric, triad.polar, triad.polar);
  const amrex::Real bss = tensor_projection(
      curvature.magnetic, triad.radial, triad.radial);
  const amrex::Real bsp = tensor_projection(
      curvature.magnetic, triad.radial, triad.azimuthal);
  const amrex::Real bsq =
      tensor_projection(curvature.magnetic, triad.radial, triad.polar);
  const amrex::Real bpp = tensor_projection(
      curvature.magnetic, triad.azimuthal, triad.azimuthal);
  const amrex::Real bpq = tensor_projection(
      curvature.magnetic, triad.azimuthal, triad.polar);
  const amrex::Real bqq =
      tensor_projection(curvature.magnetic, triad.polar, triad.polar);

  output.psi[0] = {amrex::Real(0.5) * (epp - eqq + 2 * bpq),
                   amrex::Real(0.5) * (-bpp + bqq + 2 * epq)};
  output.psi[1] = {-amrex::Real(0.5) * (esp + bsq),
                   amrex::Real(0.5) * (bsp - esq)};
  output.psi[2] = {amrex::Real(0.5) * ess,
                   -amrex::Real(0.5) * bss};
  output.psi[3] = {amrex::Real(0.5) * (esp - bsq),
                   -amrex::Real(0.5) * (bsp + esq)};
  output.psi[4] = {amrex::Real(0.5) * (epp - eqq - 2 * bpq),
                   amrex::Real(0.5) * (-bpp + bqq - 2 * epq)};

  weyl_invariants(output.psi, output.invariant_i, output.invariant_j);

  amrex::Real weyl_scale = 0;
  for (int scalar = 0; scalar < 5; ++scalar) {
    const amrex::Real scale =
        complex_maximum_absolute_component(output.psi[scalar]);
    weyl_scale = scale > weyl_scale ? scale : weyl_scale;
  }
  ComplexValue normalized_psi[5]{};
  amrex::Real mean_weyl_norm = 0;
  if (weyl_scale > 0 && amrex::Math::isfinite(weyl_scale))
    for (int scalar = 0; scalar < 5; ++scalar) {
      normalized_psi[scalar] =
          complex_scale(output.psi[scalar], 1 / weyl_scale);
      mean_weyl_norm += complex_norm_squared(normalized_psi[scalar]) / 5;
    }
  ComplexValue normalized_i;
  ComplexValue normalized_j;
  weyl_invariants(normalized_psi, normalized_i, normalized_j);
  const ComplexValue invariant_j_squared =
      complex_multiply(normalized_j, normalized_j);
  const ComplexValue invariant_i_cubed =
      complex_multiply(complex_multiply(normalized_i, normalized_i),
                       normalized_i);
  const ComplexValue speciality_numerator = complex_subtract(
      complex_scale(invariant_j_squared, 27), invariant_i_cubed);
  output.speciality_index_valid =
      weyl_scale > 0 &&
      complex_divide(complex_scale(invariant_j_squared, 27),
                     invariant_i_cubed, output.speciality_index);
  output.speciality_difference_valid =
      weyl_scale > 0 && mean_weyl_norm > 0 &&
      amrex::Math::isfinite(mean_weyl_norm);
  if (output.speciality_difference_valid) {
    const amrex::Real normalization =
        mean_weyl_norm * mean_weyl_norm *
        amrex::Math::rsqrt(mean_weyl_norm);
    output.speciality_difference =
        complex_scale(speciality_numerator, 1 / normalization);
    output.speciality_difference_valid =
        complex_is_finite(output.speciality_difference);
  }

  for (int scalar = 0; scalar < 5; ++scalar)
    if (!complex_is_finite(output.psi[scalar]))
      return false;
  return complex_is_finite(output.invariant_i) &&
         complex_is_finite(output.invariant_j);
}

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_WEYL_CURVATURE_HXX
