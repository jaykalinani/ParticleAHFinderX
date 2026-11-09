/**
 * \file coordinate_fluxes.hxx
 * \brief Portable finite-radius ADM and conformal-ADM surface fluxes.
 */
#ifndef PARTICLEAHFINDERX_COORDINATE_FLUXES_HXX
#define PARTICLEAHFINDERX_COORDINATE_FLUXES_HXX

#include "expansion.hxx"

#include <AMReX_Algorithm.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace ParticleAHFinderX {

struct CoordinateFluxInput {
  amrex::Real gamma[6];
  amrex::Real d_gamma[3][6];
  amrex::Real curv[6];
  amrex::Real normal[3];
  amrex::Real position[3];
  amrex::Real center[3];
  amrex::Real physical_area_weight = 0;
};

struct CoordinateFluxOutput {
  amrex::Real adm_energy = 0;
  amrex::Real adm_momentum[3]{0, 0, 0};
  amrex::Real adm_angular_momentum[3]{0, 0, 0};
  amrex::Real conformal_adm_energy = 0;
  amrex::Real conformal_adm_momentum[3]{0, 0, 0};
  amrex::Real conformal_adm_angular_momentum[3]{0, 0, 0};
};

/** Offsets of each group in the compact fourteen-real analysis record. */
struct CoordinateFluxLayout {
  static constexpr int adm_energy = 0;
  static constexpr int adm_momentum = 1;
  static constexpr int adm_angular_momentum = 4;
  static constexpr int conformal_adm_energy = 7;
  static constexpr int conformal_adm_momentum = 8;
  static constexpr int conformal_adm_angular_momentum = 11;
  static constexpr int component_count = 14;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
levi_civita(const int first, const int second, const int third) noexcept {
  if (first == second || first == third || second == third)
    return 0;
  return (first == 0 && second == 1 && third == 2) ||
                 (first == 1 && second == 2 && third == 0) ||
                 (first == 2 && second == 0 && third == 1)
             ? 1
             : -1;
}

/**
 * Evaluate one cell's already weighted coordinate flux contributions.
 *
 * The physical-normal form uses s^i dA. The conformal form reconstructs the
 * covariant coordinate surface element as gamma_ij s^j dA/sqrt(det(gamma)).
 * Angular summation of these outputs therefore completes each surface
 * integral without any additional quadrature factor.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
evaluate_coordinate_fluxes(const CoordinateFluxInput &input,
                           const amrex::Real inverse[3][3],
                           const amrex::Real trace_curvature,
                           CoordinateFluxOutput &output) noexcept {
  output = CoordinateFluxOutput{};
  if (!(input.physical_area_weight > 0) ||
      !amrex::Math::isfinite(input.physical_area_weight))
    return false;
  for (int component = 0; component < 6; ++component) {
    if (!amrex::Math::isfinite(input.gamma[component]) ||
        !amrex::Math::isfinite(input.curv[component]))
      return false;
    for (int derivative = 0; derivative < 3; ++derivative)
      if (!amrex::Math::isfinite(
              input.d_gamma[derivative][component]))
        return false;
  }
  for (int d = 0; d < 3; ++d)
    if (!amrex::Math::isfinite(input.normal[d]) ||
        !amrex::Math::isfinite(input.position[d]) ||
        !amrex::Math::isfinite(input.center[d]))
      return false;

  const amrex::Real determinant = metric_determinant(input.gamma);
  if (!(determinant > 0) || !amrex::Math::isfinite(determinant) ||
      !amrex::Math::isfinite(trace_curvature))
    return false;
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      if (!amrex::Math::isfinite(inverse[first][second]))
        return false;
  const amrex::Real sqrt_determinant =
      determinant * amrex::Math::rsqrt(determinant);

  amrex::Real inverse_derivative[3][3][3]{};
  for (int derivative = 0; derivative < 3; ++derivative)
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second)
        for (int left = 0; left < 3; ++left)
          for (int right = 0; right < 3; ++right)
            inverse_derivative[derivative][first][second] -=
                inverse[first][left] * inverse[second][right] *
                input.d_gamma[derivative]
                             [symmetric_component(left, right)];

  const amrex::Real inverse_eight_pi =
      amrex::Real(1) /
      (amrex::Real(8) * amrex::Math::pi<amrex::Real>());
  amrex::Real energy_density = 0;
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      for (int normal = 0; normal < 3; ++normal) {
        const amrex::Real delta = first == second ? 1 : 0;
        const amrex::Real metric_perturbation =
            symmetric_value(input.gamma, first, second) - delta;
        energy_density +=
            (delta * input.d_gamma[second]
                                  [symmetric_component(first, normal)] -
             inverse[first][second] *
                 input.d_gamma[normal]
                              [symmetric_component(first, second)] -
             metric_perturbation *
                 inverse_derivative[normal][first][second]) *
            input.normal[normal];
      }
  output.adm_energy = amrex::Real(0.5) * inverse_eight_pi *
                      energy_density * input.physical_area_weight;

  amrex::Real mixed_curvature[3][3]{};
  amrex::Real adm_flux[3]{};
  for (int upper = 0; upper < 3; ++upper)
    for (int lower = 0; lower < 3; ++lower) {
      for (int contracted = 0; contracted < 3; ++contracted)
        mixed_curvature[upper][lower] +=
            inverse[upper][contracted] *
            symmetric_value(input.curv, contracted, lower);
      adm_flux[upper] +=
          (mixed_curvature[upper][lower] -
           (upper == lower ? trace_curvature : 0)) *
          input.normal[lower];
    }
  for (int component = 0; component < 3; ++component) {
    output.adm_momentum[component] =
        inverse_eight_pi * adm_flux[component] *
        input.physical_area_weight;
    for (int position = 0; position < 3; ++position)
      for (int flux = 0; flux < 3; ++flux)
        output.adm_angular_momentum[component] +=
            inverse_eight_pi *
            amrex::Real(levi_civita(component, position, flux)) *
            input.position[position] * adm_flux[flux] *
            input.physical_area_weight;
  }

  amrex::Real normal_covector[3]{};
  amrex::Real coordinate_surface_element[3]{};
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second)
      normal_covector[first] +=
          symmetric_value(input.gamma, first, second) *
          input.normal[second];
    coordinate_surface_element[first] =
        normal_covector[first] * input.physical_area_weight /
        sqrt_determinant;
  }

  const amrex::Real chi_raw = amrex::Real(1) / std::cbrt(determinant);
  const amrex::Real chi = amrex::max(chi_raw, amrex::Real(1.0e-4));
  if (!(chi_raw > 0) || !amrex::Math::isfinite(chi_raw))
    return false;
  amrex::Real conformal_inverse[3][3];
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      conformal_inverse[first][second] = inverse[first][second] / chi_raw;

  amrex::Real dchi[3]{};
  amrex::Real dconformal[3][3][3]{};
  for (int derivative = 0; derivative < 3; ++derivative) {
    amrex::Real trace_derivative = 0;
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second)
        trace_derivative +=
            inverse[first][second] *
            input.d_gamma[derivative]
                         [symmetric_component(first, second)];
    dchi[derivative] = -chi_raw * trace_derivative / amrex::Real(3);
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second)
        dconformal[derivative][first][second] =
            dchi[derivative] *
                symmetric_value(input.gamma, first, second) +
            chi_raw * input.d_gamma[derivative]
                                      [symmetric_component(first, second)];
  }

  amrex::Real contracted_christoffel[3]{};
  for (int upper = 0; upper < 3; ++upper)
    for (int first = 0; first < 3; ++first)
      for (int second = 0; second < 3; ++second)
        for (int contracted = 0; contracted < 3; ++contracted)
          contracted_christoffel[upper] +=
              amrex::Real(0.5) * conformal_inverse[first][second] *
              conformal_inverse[upper][contracted] *
              (dconformal[first][contracted][second] +
               dconformal[second][contracted][first] -
               dconformal[contracted][first][second]);

  const amrex::Real inverse_sqrt_chi = amrex::Math::rsqrt(chi);
  const amrex::Real inverse_chi_three_halves = inverse_sqrt_chi / chi;
  amrex::Real conformal_mass_flux[3]{};
  for (int first = 0; first < 3; ++first) {
    amrex::Real raised_dchi = 0;
    for (int second = 0; second < 3; ++second)
      raised_dchi += conformal_inverse[first][second] * dchi[second];
    conformal_mass_flux[first] =
        amrex::Real(0.5) * inverse_eight_pi * inverse_sqrt_chi *
        (contracted_christoffel[first] +
         amrex::Real(2) * raised_dchi / chi);
    output.conformal_adm_energy +=
        conformal_mass_flux[first] * coordinate_surface_element[first];
  }

  amrex::Real conformal_tracefree[3][3];
  for (int upper = 0; upper < 3; ++upper)
    for (int lower = 0; lower < 3; ++lower)
      conformal_tracefree[upper][lower] =
          mixed_curvature[upper][lower] -
          (upper == lower ? trace_curvature / amrex::Real(3) : 0);
  for (int component = 0; component < 3; ++component) {
    for (int surface = 0; surface < 3; ++surface)
      output.conformal_adm_momentum[component] +=
          inverse_eight_pi * inverse_chi_three_halves *
          (conformal_tracefree[surface][component] -
           (surface == component
                ? amrex::Real(2) * trace_curvature / amrex::Real(3)
                : 0)) *
          coordinate_surface_element[surface];
    for (int surface = 0; surface < 3; ++surface)
      for (int position = 0; position < 3; ++position)
        for (int flux = 0; flux < 3; ++flux)
          output.conformal_adm_angular_momentum[component] +=
              inverse_eight_pi * inverse_chi_three_halves *
              amrex::Real(levi_civita(component, position, flux)) *
              (input.position[position] - input.center[position]) *
              conformal_tracefree[surface][flux] *
              coordinate_surface_element[surface];
  }

  if (!amrex::Math::isfinite(output.adm_energy) ||
      !amrex::Math::isfinite(output.conformal_adm_energy))
    return false;
  for (int component = 0; component < 3; ++component)
    if (!amrex::Math::isfinite(output.adm_momentum[component]) ||
        !amrex::Math::isfinite(output.adm_angular_momentum[component]) ||
        !amrex::Math::isfinite(output.conformal_adm_momentum[component]) ||
        !amrex::Math::isfinite(
            output.conformal_adm_angular_momentum[component]))
      return false;
  return true;
}

/** Convenience entry point for isolated tests and other pointwise callers. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
evaluate_coordinate_fluxes(const CoordinateFluxInput &input,
                           CoordinateFluxOutput &output) noexcept {
  for (int component = 0; component < 6; ++component)
    if (!amrex::Math::isfinite(input.gamma[component]) ||
        !amrex::Math::isfinite(input.curv[component])) {
      output = CoordinateFluxOutput{};
      return false;
    }
  amrex::Real inverse[3][3];
  if (!inverse_metric(input.gamma, inverse)) {
    output = CoordinateFluxOutput{};
    return false;
  }
  amrex::Real trace_curvature = 0;
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      trace_curvature += inverse[first][second] *
                         symmetric_value(input.curv, first, second);
  return evaluate_coordinate_fluxes(input, inverse, trace_curvature, output);
}

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_COORDINATE_FLUXES_HXX
