/**
 * \file coordinate_flux_self_test.cxx
 * \brief Accelerator tests for finite-radius coordinate fluxes.
 */
#include "angular_geometry.hxx"
#include "coordinate_fluxes.hxx"
#include "runtime.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_Array.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>

#include <array>
#include <limits>

namespace ParticleAHFinderX {
namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
pack_coordinate_fluxes(
    const CoordinateFluxOutput &output,
    amrex::Real values[CoordinateFluxLayout::component_count]) noexcept {
  values[CoordinateFluxLayout::adm_energy] = output.adm_energy;
  values[CoordinateFluxLayout::conformal_adm_energy] =
      output.conformal_adm_energy;
  for (int d = 0; d < 3; ++d) {
    values[CoordinateFluxLayout::adm_momentum + d] =
        output.adm_momentum[d];
    values[CoordinateFluxLayout::adm_angular_momentum + d] =
        output.adm_angular_momentum[d];
    values[CoordinateFluxLayout::conformal_adm_momentum + d] =
        output.conformal_adm_momentum[d];
    values[CoordinateFluxLayout::conformal_adm_angular_momentum + d] =
        output.conformal_adm_angular_momentum[d];
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
absolute_value(const amrex::Real value) noexcept {
  return value < 0 ? -value : value;
}

struct CoordinateFluxSelfTestResult {
  amrex::Real minkowski_error = 0;
  amrex::Real schwarzschild_error = 0;
  amrex::Real manufactured_error = 0;
  amrex::Long invalid = 0;
};

CoordinateFluxSelfTestResult run_coordinate_flux_test(const int ntheta) {
  const int nphi = 2 * ntheta;
  const amrex::Long point_count = amrex::Long(ntheta) * nphi;
  constexpr amrex::Real radius = 2;
  constexpr amrex::Real schwarzschild_radius = 10;
  constexpr amrex::Real mass = 1;
  const amrex::GpuArray<amrex::Real, 3> boost{
      {amrex::Real(0.6), amrex::Real(-0.4), amrex::Real(0.3)}};
  const amrex::GpuArray<amrex::Real, 3> center{
      {amrex::Real(0.7), amrex::Real(0.2), amrex::Real(-0.5)}};
  const amrex::GpuArray<amrex::Real, 3> spin{
      {amrex::Real(0.15), amrex::Real(-0.22), amrex::Real(0.31)}};
  amrex::Gpu::DeviceVector<amrex::Real> integrals(
      2 * CoordinateFluxLayout::component_count, 0);
  amrex::Gpu::DeviceVector<amrex::Real> minkowski_error(1, 0);
  amrex::Gpu::DeviceVector<amrex::Long> invalid(1, 0);
  auto *const sums = integrals.dataPtr();
  auto *const flat_error = minkowski_error.dataPtr();
  auto *const invalid_count = invalid.dataPtr();

  amrex::ParallelFor(
      point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
        const int itheta = static_cast<int>(point / nphi);
        const int iphi = static_cast<int>(point - amrex::Long(itheta) * nphi);
        const amrex::Real dtheta =
            amrex::Math::pi<amrex::Real>() / ntheta;
        const amrex::Real dphi =
            2 * amrex::Math::pi<amrex::Real>() / nphi;
        const amrex::Real theta = (itheta + amrex::Real(0.5)) * dtheta;
        const amrex::Real phi = (iphi + amrex::Real(0.5)) * dphi;
        const AngularGeometry basis = angular_geometry(theta, phi);
        const amrex::Real sine = amrex::Math::sincos(theta).first;

        CoordinateFluxInput input{};
        input.gamma[0] = input.gamma[3] = input.gamma[5] = 1;
        for (int d = 0; d < 3; ++d) {
          input.normal[d] = basis.radial[d];
          input.position[d] = radius * basis.radial[d];
          input.center[d] = 0;
        }
        input.physical_area_weight = surface_cell_area_weight(
            radius * radius * sine, dtheta, dphi);
        CoordinateFluxOutput output;
        if (!evaluate_coordinate_fluxes(input, output)) {
          amrex::Gpu::Atomic::AddNoRet(invalid_count, amrex::Long(1));
          return;
        }
        amrex::Real values[CoordinateFluxLayout::component_count];
        pack_coordinate_fluxes(output, values);
        for (int component = 0;
             component < CoordinateFluxLayout::component_count;
             ++component)
          amrex::Gpu::Atomic::Max(flat_error,
                                  absolute_value(values[component]));

        const amrex::Real psi =
            1 + mass / (2 * schwarzschild_radius);
        const amrex::Real conformal_factor =
            psi * psi * psi * psi;
        const amrex::Real radial_metric_derivative =
            -2 * mass * psi * psi * psi /
            (schwarzschild_radius * schwarzschild_radius);
        input = CoordinateFluxInput{};
        input.gamma[0] = input.gamma[3] = input.gamma[5] =
            conformal_factor;
        for (int derivative = 0; derivative < 3; ++derivative) {
          const amrex::Real value =
              radial_metric_derivative * basis.radial[derivative];
          input.d_gamma[derivative][0] = value;
          input.d_gamma[derivative][3] = value;
          input.d_gamma[derivative][5] = value;
        }
        for (int d = 0; d < 3; ++d) {
          input.normal[d] = basis.radial[d] / (psi * psi);
          input.position[d] = schwarzschild_radius * basis.radial[d];
          input.center[d] = 0;
        }
        input.physical_area_weight = surface_cell_area_weight(
            conformal_factor * schwarzschild_radius *
                schwarzschild_radius * sine,
            dtheta, dphi);
        output = CoordinateFluxOutput{};
        if (!evaluate_coordinate_fluxes(input, output)) {
          amrex::Gpu::Atomic::AddNoRet(invalid_count, amrex::Long(1));
          return;
        }
        pack_coordinate_fluxes(output, values);
        for (int component = 0;
             component < CoordinateFluxLayout::component_count;
             ++component)
          amrex::Gpu::Atomic::AddNoRet(sums + component, values[component]);

        input = CoordinateFluxInput{};
        input.gamma[0] = input.gamma[3] = input.gamma[5] = 1;
        const amrex::Real spin_cross_normal[3]{
            spin[1] * basis.radial[2] - spin[2] * basis.radial[1],
            spin[2] * basis.radial[0] - spin[0] * basis.radial[2],
            spin[0] * basis.radial[1] - spin[1] * basis.radial[0]};
        const amrex::Real spin_scale = 3 / (radius * radius * radius);
        for (int first = 0; first < 3; ++first)
          for (int second = first; second < 3; ++second)
            input.curv[symmetric_component(first, second)] =
                boost[first] * basis.radial[second] +
                basis.radial[first] * boost[second] +
                spin_scale *
                    (spin_cross_normal[first] * basis.radial[second] +
                     basis.radial[first] * spin_cross_normal[second]);
        for (int d = 0; d < 3; ++d) {
          input.normal[d] = basis.radial[d];
          input.position[d] = center[d] + radius * basis.radial[d];
          input.center[d] = center[d];
        }
        input.physical_area_weight = surface_cell_area_weight(
            radius * radius * sine, dtheta, dphi);
        output = CoordinateFluxOutput{};
        if (!evaluate_coordinate_fluxes(input, output)) {
          amrex::Gpu::Atomic::AddNoRet(invalid_count, amrex::Long(1));
          return;
        }
        pack_coordinate_fluxes(output, values);
        for (int component = 0;
             component < CoordinateFluxLayout::component_count;
             ++component)
          amrex::Gpu::Atomic::AddNoRet(
              sums + CoordinateFluxLayout::component_count + component,
              values[component]);
      });

  std::array<amrex::Real, 2 * CoordinateFluxLayout::component_count>
      host_sums{};
  amrex::Real host_minkowski_error = 0;
  amrex::Long host_invalid = 0;
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, integrals.begin(),
                   integrals.end(), host_sums.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, minkowski_error.begin(),
                   minkowski_error.end(), &host_minkowski_error);
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, invalid.begin(), invalid.end(),
                   &host_invalid);

  const amrex::Real psi = 1 + mass / (2 * schwarzschild_radius);
  std::array<amrex::Real, CoordinateFluxLayout::component_count>
      schwarzschild_exact{};
  schwarzschild_exact[CoordinateFluxLayout::adm_energy] =
      mass * (3 / (psi * psi * psi) - psi * psi * psi * psi * psi) / 2;
  schwarzschild_exact[CoordinateFluxLayout::conformal_adm_energy] =
      mass * psi;
  std::array<amrex::Real, CoordinateFluxLayout::component_count>
      manufactured_exact{};
  for (int d = 0; d < 3; ++d) {
    manufactured_exact[CoordinateFluxLayout::adm_momentum + d] =
        radius * radius * boost[d] / 3;
    manufactured_exact[CoordinateFluxLayout::conformal_adm_momentum + d] =
        manufactured_exact[CoordinateFluxLayout::adm_momentum + d];
    manufactured_exact[
        CoordinateFluxLayout::conformal_adm_angular_momentum + d] = spin[d];
  }
  const int adm_momentum = CoordinateFluxLayout::adm_momentum;
  const int adm_angular_momentum =
      CoordinateFluxLayout::adm_angular_momentum;
  manufactured_exact[adm_angular_momentum] =
      center[1] * manufactured_exact[adm_momentum + 2] -
      center[2] * manufactured_exact[adm_momentum + 1] + spin[0];
  manufactured_exact[adm_angular_momentum + 1] =
      center[2] * manufactured_exact[adm_momentum] -
      center[0] * manufactured_exact[adm_momentum + 2] + spin[1];
  manufactured_exact[adm_angular_momentum + 2] =
      center[0] * manufactured_exact[adm_momentum + 1] -
      center[1] * manufactured_exact[adm_momentum] + spin[2];

  CoordinateFluxSelfTestResult result;
  result.minkowski_error = host_minkowski_error;
  result.invalid = host_invalid;
  for (int component = 0;
       component < CoordinateFluxLayout::component_count;
       ++component) {
    result.schwarzschild_error = amrex::max(
        result.schwarzschild_error,
        absolute_value(host_sums[component] -
                       schwarzschild_exact[component]));
    result.manufactured_error = amrex::max(
        result.manufactured_error,
        absolute_value(
            host_sums[CoordinateFluxLayout::component_count + component] -
            manufactured_exact[component]));
  }
  return result;
}

} // namespace

extern "C" void ParticleAHFinderX_CoordinateFluxSelfTest(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  if (!run_coordinate_flux_self_test ||
      state.coordinate_flux_self_test_complete)
    return;

  constexpr std::array<int, 3> resolutions{{16, 32, 64}};
  std::array<CoordinateFluxSelfTestResult, resolutions.size()> results;
  for (std::size_t level = 0; level < resolutions.size(); ++level) {
    results[level] = run_coordinate_flux_test(resolutions[level]);
    CCTK_VINFO(
        "Coordinate-flux self-test: ntheta=%d Minkowski/Schwarzschild/"
        "manufactured errors=%.3g/%.3g/%.3g invalid=%lld",
        resolutions[level], double(results[level].minkowski_error),
        double(results[level].schwarzschild_error),
        double(results[level].manufactured_error),
        static_cast<long long>(results[level].invalid));
    if (results[level].invalid != 0)
      CCTK_VERROR("ParticleAHFinderX coordinate-flux self-test found %lld "
                  "invalid point evaluation(s)",
                  static_cast<long long>(results[level].invalid));
  }
  for (std::size_t level = 1; level < resolutions.size(); ++level)
    if (!(results[level].manufactured_error <
          amrex::Real(0.3) * results[level - 1].manufactured_error))
      CCTK_VERROR(
          "ParticleAHFinderX manufactured coordinate fluxes did not "
          "converge from ntheta=%d to %d: %.17g -> %.17g",
          resolutions[level - 1], resolutions[level],
          double(results[level - 1].manufactured_error),
          double(results[level].manufactured_error));
  const amrex::Real roundoff =
      amrex::Real(100000) * std::numeric_limits<amrex::Real>::epsilon();
  const auto &finest = results.back();
  if (finest.minkowski_error > roundoff ||
      finest.schwarzschild_error > roundoff ||
      finest.manufactured_error > amrex::Real(1.0e-4))
    CCTK_VERROR(
        "ParticleAHFinderX coordinate-flux finest-grid guard failed: "
        "Minkowski/Schwarzschild/manufactured=%.17g/%.17g/%.17g",
        double(finest.minkowski_error),
        double(finest.schwarzschild_error),
        double(finest.manufactured_error));
  CCTK_INFO("Finite-radius coordinate-flux accelerator self-test passed");
  state.coordinate_flux_self_test_complete = true;
}

} // namespace ParticleAHFinderX
