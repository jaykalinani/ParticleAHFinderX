/**
 * \file axial_symmetry.hxx
 * \brief Portable device helpers for variational axial-symmetry analysis.
 */
#ifndef PARTICLEAHFINDERX_AXIAL_SYMMETRY_HXX
#define PARTICLEAHFINDERX_AXIAL_SYMMETRY_HXX

#include <AMReX_Algorithm.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <limits>

namespace ParticleAHFinderX {

constexpr int maximum_axial_harmonic_degree = 6;
// Scale, axis (3), eigenvalue/residuals (4), raw potential mean,
// dimensionless invariant-coordinate extrema (2), and inverse R^2.
constexpr int axial_surface_real_components = 12;

enum class AxialSymmetryStatus : int {
  valid = 0,
  not_requested = 1,
  invalid_intrinsic_geometry = 2,
  invalid_basis = 3,
  singular_mass_matrix = 4,
  eigensolver_failed = 5,
  invalid_axis = 6,
  invalid_normalization = 7,
  invalid_spin_integral = 8,
};

struct AxialSymmetryControl {
  bool enabled = false;
  bool compute_multipoles = false;
  int harmonic_degree = 4;
  int maximum_jacobi_sweeps = 64;
  amrex::Real eigen_tolerance = amrex::Real(1.0e-12);
  amrex::Real minimum_axis_separation = amrex::Real(0.1);
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
axial_absolute_value(const amrex::Real value) noexcept {
  return value < 0 ? -value : value;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
axial_square_root(const amrex::Real value) noexcept {
  return value > 0 ? value * amrex::Math::rsqrt(value) : 0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
axial_basis_count(const int harmonic_degree) noexcept {
  return (harmonic_degree + 1) * (harmonic_degree + 1) - 1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
axial_matrix_index(const int row, const int column,
                   const int dimension) noexcept {
  return row * dimension + column;
}

/** Decode the real spherical-harmonic order l=1,...,L and m=-l,...,l. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
axial_basis_mode(const int mode, const int harmonic_degree, int &ell,
                 int &order) noexcept {
  int current = 0;
  for (ell = 1; ell <= harmonic_degree; ++ell)
    for (order = -ell; order <= ell; ++order, ++current)
      if (current == mode)
        return true;
  ell = 0;
  order = 0;
  return false;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
associated_legendre(const int ell, const int order,
                    const amrex::Real cosine) noexcept {
  amrex::Real pmm = 1;
  if (order > 0) {
    const amrex::Real sine_squared =
        amrex::max(amrex::Real(0), 1 - cosine * cosine);
    const amrex::Real sine =
        sine_squared > 0
            ? sine_squared * amrex::Math::rsqrt(sine_squared)
            : 0;
    amrex::Real factor = 1;
    for (int index = 1; index <= order; ++index) {
      pmm *= -factor * sine;
      factor += 2;
    }
  }
  if (ell == order)
    return pmm;
  amrex::Real previous = pmm;
  amrex::Real current = (2 * order + 1) * cosine * pmm;
  if (ell == order + 1)
    return current;
  for (int degree = order + 2; degree <= ell; ++degree) {
    const amrex::Real next =
        ((2 * degree - 1) * cosine * current -
         (degree + order - 1) * previous) /
        (degree - order);
    previous = current;
    current = next;
  }
  return current;
}

/** Orthonormal real spherical harmonic, excluding the constant mode. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
axial_basis_value(const int mode, const int harmonic_degree,
                  const amrex::Real theta, const amrex::Real phi,
                  amrex::Real &value) noexcept {
  int ell = 0;
  int signed_order = 0;
  if (!axial_basis_mode(mode, harmonic_degree, ell, signed_order))
    return false;
  const int order = signed_order < 0 ? -signed_order : signed_order;
  const amrex::Real cosine = amrex::Math::sincos(theta).second;
  amrex::Real factorial_ratio = 1;
  for (int index = ell - order + 1; index <= ell + order; ++index)
    factorial_ratio /= index;
  amrex::Real normalization =
      (2 * ell + 1) * factorial_ratio /
      (4 * amrex::Math::pi<amrex::Real>());
  normalization = normalization > 0
                      ? normalization * amrex::Math::rsqrt(normalization)
                      : 0;
  if (order != 0)
    normalization *= axial_square_root(amrex::Real(2));
  const amrex::Real legendre =
      associated_legendre(ell, order, cosine);
  if (signed_order == 0) {
    value = normalization * legendre;
  } else {
    const auto sincos = amrex::Math::sincos(order * phi);
    value = normalization * legendre *
            (signed_order < 0 ? sincos.first : sincos.second);
  }
  return amrex::Math::isfinite(value);
}

/**
 * Solve K x = lambda M x for the smallest Rayleigh--Ritz mode.
 *
 * One accelerator thread owns each small dense problem. All matrices are in
 * device global memory, which avoids backend-specific solver libraries and
 * keeps the angular mode coefficients off the host.
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE AxialSymmetryStatus
smallest_generalized_symmetric_mode(
    const amrex::Real *const stiffness, const amrex::Real *const mass,
    amrex::Real *const factor, amrex::Real *const work,
    amrex::Real *const eigenvectors, amrex::Real *const coefficients,
    amrex::Real &eigenvalue, amrex::Real &eigen_residual,
    int &completed_sweeps, const int dimension,
    const int maximum_sweeps, const amrex::Real tolerance) noexcept {
  if (dimension <= 0 || maximum_sweeps <= 0 || !(tolerance > 0))
    return AxialSymmetryStatus::eigensolver_failed;

  amrex::Real mass_scale = 0;
  for (int row = 0; row < dimension; ++row) {
    mass_scale = amrex::max(mass_scale,
                            axial_absolute_value(mass[axial_matrix_index(
                                row, row, dimension)]));
    for (int column = 0; column < dimension; ++column) {
      const int index = axial_matrix_index(row, column, dimension);
      factor[index] = 0;
      work[index] = 0;
      eigenvectors[index] = 0;
      if (!amrex::Math::isfinite(stiffness[index]) ||
          !amrex::Math::isfinite(mass[index]))
        return AxialSymmetryStatus::invalid_basis;
    }
  }
  if (!(mass_scale > 0) || !amrex::Math::isfinite(mass_scale))
    return AxialSymmetryStatus::singular_mass_matrix;

  const amrex::Real pivot_floor =
      amrex::max(amrex::Real(64) * tolerance,
                 amrex::Real(1024) *
                     std::numeric_limits<amrex::Real>::epsilon()) *
      mass_scale;
  for (int row = 0; row < dimension; ++row)
    for (int column = 0; column <= row; ++column) {
      amrex::Real value =
          mass[axial_matrix_index(row, column, dimension)];
      for (int contracted = 0; contracted < column; ++contracted)
        value -= factor[axial_matrix_index(row, contracted, dimension)] *
                 factor[axial_matrix_index(column, contracted, dimension)];
      if (row == column) {
        if (!(value > pivot_floor) || !amrex::Math::isfinite(value))
          return AxialSymmetryStatus::singular_mass_matrix;
        factor[axial_matrix_index(row, column, dimension)] =
            axial_square_root(value);
      } else {
        factor[axial_matrix_index(row, column, dimension)] =
            value /
            factor[axial_matrix_index(column, column, dimension)];
      }
    }

  // work = L^{-1} K
  for (int column = 0; column < dimension; ++column)
    for (int row = 0; row < dimension; ++row) {
      amrex::Real value =
          stiffness[axial_matrix_index(row, column, dimension)];
      for (int contracted = 0; contracted < row; ++contracted)
        value -= factor[axial_matrix_index(row, contracted, dimension)] *
                 work[axial_matrix_index(contracted, column, dimension)];
      work[axial_matrix_index(row, column, dimension)] =
          value / factor[axial_matrix_index(row, row, dimension)];
    }

  // eigenvectors temporarily holds B = L^{-1} K L^{-T}. Each row solves
  // L B_row^T = work_row^T while factor still contains the Cholesky factor.
  for (int row = 0; row < dimension; ++row)
    for (int column = 0; column < dimension; ++column) {
      amrex::Real value = work[axial_matrix_index(row, column, dimension)];
      for (int contracted = 0; contracted < column; ++contracted)
        value -= factor[axial_matrix_index(column, contracted, dimension)] *
                 eigenvectors[axial_matrix_index(row, contracted,
                                                  dimension)];
      eigenvectors[axial_matrix_index(row, column, dimension)] =
          value / factor[axial_matrix_index(column, column, dimension)];
    }
  for (int row = 0; row < dimension; ++row)
    for (int column = row + 1; column < dimension; ++column) {
      const amrex::Real symmetric =
          amrex::Real(0.5) *
          (eigenvectors[axial_matrix_index(row, column, dimension)] +
           eigenvectors[axial_matrix_index(column, row, dimension)]);
      eigenvectors[axial_matrix_index(row, column, dimension)] = symmetric;
      eigenvectors[axial_matrix_index(column, row, dimension)] = symmetric;
    }
  for (int row = 0; row < dimension; ++row)
    for (int column = 0; column < dimension; ++column) {
      factor[axial_matrix_index(row, column, dimension)] =
          eigenvectors[axial_matrix_index(row, column, dimension)];
      eigenvectors[axial_matrix_index(row, column, dimension)] =
          row == column ? 1 : 0;
    }

  completed_sweeps = 0;
  bool converged = false;
  const amrex::Real relative_tolerance =
      amrex::max(tolerance,
                 amrex::Real(64) *
                     std::numeric_limits<amrex::Real>::epsilon());
  for (int sweep = 0; sweep < maximum_sweeps; ++sweep) {
    amrex::Real diagonal_scale = 0;
    amrex::Real maximum_off_diagonal = 0;
    for (int row = 0; row < dimension; ++row) {
      diagonal_scale = amrex::max(
          diagonal_scale,
          axial_absolute_value(
              factor[axial_matrix_index(row, row, dimension)]));
      for (int column = row + 1; column < dimension; ++column)
        maximum_off_diagonal = amrex::max(
            maximum_off_diagonal,
            axial_absolute_value(
                factor[axial_matrix_index(row, column, dimension)]));
    }
    const amrex::Real matrix_scale =
        amrex::max(diagonal_scale, maximum_off_diagonal);
    if (!(matrix_scale > 0) ||
        maximum_off_diagonal <= relative_tolerance * matrix_scale) {
      converged = true;
      completed_sweeps = sweep;
      break;
    }
    for (int first = 0; first < dimension; ++first)
      for (int second = first + 1; second < dimension; ++second) {
        const amrex::Real off =
            factor[axial_matrix_index(first, second, dimension)];
        const amrex::Real first_diagonal =
            factor[axial_matrix_index(first, first, dimension)];
        const amrex::Real second_diagonal =
            factor[axial_matrix_index(second, second, dimension)];
        if (axial_absolute_value(off) <=
            relative_tolerance * matrix_scale)
          continue;
        const amrex::Real tau =
            (second_diagonal - first_diagonal) / (2 * off);
        const amrex::Real tangent =
            (tau < 0 ? -1 : 1) /
            (axial_absolute_value(tau) +
             axial_square_root(1 + tau * tau));
        const amrex::Real cosine = amrex::Math::rsqrt(1 + tangent * tangent);
        const amrex::Real sine = tangent * cosine;
        for (int index = 0; index < dimension; ++index) {
          if (index == first || index == second)
            continue;
          const amrex::Real first_value =
              factor[axial_matrix_index(index, first, dimension)];
          const amrex::Real second_value =
              factor[axial_matrix_index(index, second, dimension)];
          factor[axial_matrix_index(index, first, dimension)] =
              factor[axial_matrix_index(first, index, dimension)] =
                  cosine * first_value - sine * second_value;
          factor[axial_matrix_index(index, second, dimension)] =
              factor[axial_matrix_index(second, index, dimension)] =
                  sine * first_value + cosine * second_value;
        }
        factor[axial_matrix_index(first, first, dimension)] =
            cosine * cosine * first_diagonal -
            2 * sine * cosine * off +
            sine * sine * second_diagonal;
        factor[axial_matrix_index(second, second, dimension)] =
            sine * sine * first_diagonal +
            2 * sine * cosine * off +
            cosine * cosine * second_diagonal;
        factor[axial_matrix_index(first, second, dimension)] = 0;
        factor[axial_matrix_index(second, first, dimension)] = 0;
        for (int row = 0; row < dimension; ++row) {
          const amrex::Real first_vector =
              eigenvectors[axial_matrix_index(row, first, dimension)];
          const amrex::Real second_vector =
              eigenvectors[axial_matrix_index(row, second, dimension)];
          eigenvectors[axial_matrix_index(row, first, dimension)] =
              cosine * first_vector - sine * second_vector;
          eigenvectors[axial_matrix_index(row, second, dimension)] =
              sine * first_vector + cosine * second_vector;
        }
      }
    completed_sweeps = sweep + 1;
  }
  if (!converged) {
    amrex::Real diagonal_scale = 0;
    amrex::Real maximum_off_diagonal = 0;
    for (int row = 0; row < dimension; ++row) {
      diagonal_scale = amrex::max(
          diagonal_scale,
          axial_absolute_value(
              factor[axial_matrix_index(row, row, dimension)]));
      for (int column = row + 1; column < dimension; ++column)
        maximum_off_diagonal = amrex::max(
            maximum_off_diagonal,
            axial_absolute_value(
                factor[axial_matrix_index(row, column, dimension)]));
    }
    const amrex::Real matrix_scale =
        amrex::max(diagonal_scale, maximum_off_diagonal);
    converged = !(matrix_scale > 0) ||
                maximum_off_diagonal <=
                    relative_tolerance * matrix_scale;
  }
  if (!converged)
    return AxialSymmetryStatus::eigensolver_failed;

  int selected = 0;
  for (int mode = 1; mode < dimension; ++mode)
    if (factor[axial_matrix_index(mode, mode, dimension)] <
        factor[axial_matrix_index(selected, selected, dimension)])
      selected = mode;
  eigenvalue = factor[axial_matrix_index(selected, selected, dimension)];
  if (!amrex::Math::isfinite(eigenvalue))
    return AxialSymmetryStatus::eigensolver_failed;

  // Recover L in work and form coefficients = L^{-T} y.
  for (int row = 0; row < dimension; ++row)
    for (int column = 0; column < dimension; ++column)
      work[axial_matrix_index(row, column, dimension)] = 0;
  for (int row = 0; row < dimension; ++row)
    for (int column = 0; column <= row; ++column) {
      amrex::Real value =
          mass[axial_matrix_index(row, column, dimension)];
      for (int contracted = 0; contracted < column; ++contracted)
        value -= work[axial_matrix_index(row, contracted, dimension)] *
                 work[axial_matrix_index(column, contracted, dimension)];
      work[axial_matrix_index(row, column, dimension)] =
          row == column
              ? axial_square_root(value)
              : value /
                    work[axial_matrix_index(column, column, dimension)];
    }
  for (int row = dimension - 1; row >= 0; --row) {
    amrex::Real value =
        eigenvectors[axial_matrix_index(row, selected, dimension)];
    for (int contracted = row + 1; contracted < dimension; ++contracted)
      value -= work[axial_matrix_index(contracted, row, dimension)] *
               coefficients[contracted];
    coefficients[row] =
        value / work[axial_matrix_index(row, row, dimension)];
  }

  amrex::Real mass_norm = 0;
  for (int row = 0; row < dimension; ++row)
    for (int column = 0; column < dimension; ++column)
      mass_norm += coefficients[row] *
                   mass[axial_matrix_index(row, column, dimension)] *
                   coefficients[column];
  if (!(mass_norm > 0) || !amrex::Math::isfinite(mass_norm))
    return AxialSymmetryStatus::eigensolver_failed;
  const amrex::Real inverse_norm = amrex::Math::rsqrt(mass_norm);
  for (int mode = 0; mode < dimension; ++mode)
    coefficients[mode] *= inverse_norm;

  amrex::Real stiffness_norm = 0;
  amrex::Real mass_action_norm = 0;
  amrex::Real residual_norm = 0;
  for (int row = 0; row < dimension; ++row) {
    amrex::Real stiffness_action = 0;
    amrex::Real mass_action = 0;
    for (int column = 0; column < dimension; ++column) {
      stiffness_action +=
          stiffness[axial_matrix_index(row, column, dimension)] *
          coefficients[column];
      mass_action += mass[axial_matrix_index(row, column, dimension)] *
                     coefficients[column];
    }
    const amrex::Real residual =
        stiffness_action - eigenvalue * mass_action;
    stiffness_norm += stiffness_action * stiffness_action;
    mass_action_norm += mass_action * mass_action;
    residual_norm += residual * residual;
  }
  const amrex::Real denominator =
      axial_square_root(stiffness_norm) +
      axial_absolute_value(eigenvalue) *
          axial_square_root(mass_action_norm);
  eigen_residual = denominator > 0
                       ? axial_square_root(residual_norm) / denominator
                       : axial_square_root(residual_norm);
  if (!amrex::Math::isfinite(eigen_residual))
    return AxialSymmetryStatus::eigensolver_failed;
  return AxialSymmetryStatus::valid;
}

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_AXIAL_SYMMETRY_HXX
