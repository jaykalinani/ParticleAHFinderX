/**
 * \file axial_symmetry_self_test.cxx
 * \brief Accelerator tests for axial trial fields and the projected solver.
 */
#include "axial_symmetry.hxx"
#include "angular_derivatives.hxx"
#include "multipoles.hxx"
#include "runtime.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>

#include <array>
#include <cmath>

namespace ParticleAHFinderX {
namespace {

struct AxialSelfTestResult {
  amrex::Real maximum_mass_error = 0;
  amrex::Real maximum_stiffness = 0;
  amrex::Real dense_eigenvalue_error = 0;
  amrex::Real dense_eigenvector_error = 0;
  amrex::Real dense_residual = 0;
  amrex::Real maximum_legendre_error = 0;
  amrex::Real maximum_legendre_derivative_error = 0;
  int dense_status = -1;
};

AxialSelfTestResult run_axial_self_test() {
  constexpr int ntheta = 32;
  constexpr int nphi = 2 * ntheta;
  constexpr int modes = 3;
  constexpr amrex::Long point_count = amrex::Long(ntheta) * nphi;
  amrex::Gpu::DeviceVector<amrex::Real> basis(modes * point_count, 0);
  amrex::Gpu::DeviceVector<amrex::Real> gradient(
      2 * modes * point_count, 0);
  amrex::Gpu::DeviceVector<amrex::Real> covector(
      2 * modes * point_count, 0);
  amrex::Gpu::DeviceVector<amrex::Real> deformation(
      3 * modes * point_count, 0);
  amrex::Gpu::DeviceVector<amrex::Real> gram(2 * modes * modes, 0);
  auto *const values = basis.dataPtr();
  auto *const derivatives = gradient.dataPtr();
  auto *const one_form = covector.dataPtr();
  auto *const strain = deformation.dataPtr();
  auto *const matrices = gram.dataPtr();

  amrex::ParallelFor(
      amrex::Long(modes) * point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long index) noexcept {
        const int mode = static_cast<int>(index / point_count);
        const amrex::Long point = index - amrex::Long(mode) * point_count;
        const int itheta = static_cast<int>(point / nphi);
        const int iphi = static_cast<int>(point - amrex::Long(itheta) * nphi);
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (itheta + amrex::Real(0.5)) / ntheta;
        const amrex::Real phi =
            2 * amrex::Math::pi<amrex::Real>() *
            (iphi + amrex::Real(0.5)) / nphi;
        amrex::Real value = 0;
        axial_basis_value(mode, 1, theta, phi, value);
        values[index] = value;
      });
  amrex::ParallelFor(
      amrex::Long(modes) * point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long index) noexcept {
        const int mode = static_cast<int>(index / point_count);
        const amrex::Long point = index - amrex::Long(mode) * point_count;
        const int itheta = static_cast<int>(point / nphi);
        const int iphi = static_cast<int>(point - amrex::Long(itheta) * nphi);
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (itheta + amrex::Real(0.5)) / ntheta;
        const amrex::Real sine = amrex::Math::sincos(theta).first;
        const AngularFirstDerivatives angular = angular_first_derivatives(
            values + amrex::Long(mode) * point_count, 0, ntheta, nphi,
            itheta, iphi, 1);
        derivatives[amrex::Long(mode) * point_count + point] = angular.theta;
        derivatives[(modes + mode) * point_count + point] = angular.phi;
        one_form[amrex::Long(mode) * point_count + point] =
            angular.phi / sine;
        one_form[(modes + mode) * point_count + point] =
            -angular.theta * sine;
      });
  amrex::ParallelFor(
      amrex::Long(modes) * point_count,
      [=] AMREX_GPU_DEVICE(const amrex::Long index) noexcept {
        const int mode = static_cast<int>(index / point_count);
        const amrex::Long point = index - amrex::Long(mode) * point_count;
        const int itheta = static_cast<int>(point / nphi);
        const int iphi = static_cast<int>(point - amrex::Long(itheta) * nphi);
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (itheta + amrex::Real(0.5)) / ntheta;
        const auto sincos = amrex::Math::sincos(theta);
        const amrex::Real sine = sincos.first;
        const amrex::Real cosine = sincos.second;
        const AngularFirstDerivatives first = angular_first_derivatives(
            one_form + amrex::Long(mode) * point_count, 0, ntheta, nphi,
            itheta, iphi, -1);
        const AngularFirstDerivatives second = angular_first_derivatives(
            one_form + amrex::Long(modes + mode) * point_count, 0, ntheta,
            nphi, itheta, iphi, 1);
        const amrex::Real phi_theta =
            one_form[amrex::Long(mode) * point_count + point];
        const amrex::Real phi_phi =
            one_form[amrex::Long(modes + mode) * point_count + point];
        const amrex::Real cotangent = cosine / sine;
        const amrex::Real covariant[2][2]{
            {first.theta, second.theta - cotangent * phi_phi},
            {first.phi - cotangent * phi_phi,
             second.phi + sine * cosine * phi_theta}};
        const amrex::Real divergence =
            covariant[0][0] + covariant[1][1] / (sine * sine);
        strain[amrex::Long(mode) * point_count + point] =
            covariant[0][0] - amrex::Real(0.5) * divergence;
        strain[amrex::Long(modes + mode) * point_count + point] =
            amrex::Real(0.5) *
            (covariant[0][1] + covariant[1][0]);
        strain[amrex::Long(2 * modes + mode) * point_count + point] =
            covariant[1][1] -
            amrex::Real(0.5) * sine * sine * divergence;
      });
  amrex::ParallelFor(
      point_count * modes * modes,
      [=] AMREX_GPU_DEVICE(const amrex::Long index) noexcept {
        const amrex::Long point = index / (modes * modes);
        const int pair = static_cast<int>(index - point * modes * modes);
        const int first = pair / modes;
        const int second = pair - first * modes;
        const int itheta = static_cast<int>(point / nphi);
        const amrex::Real theta =
            amrex::Math::pi<amrex::Real>() *
            (itheta + amrex::Real(0.5)) / ntheta;
        const amrex::Real sine = amrex::Math::sincos(theta).first;
        const amrex::Real inverse_sine_squared = 1 / (sine * sine);
        const amrex::Real weight =
            sine * amrex::Math::pi<amrex::Real>() / ntheta *
            (2 * amrex::Math::pi<amrex::Real>() / nphi);
        const amrex::Real first_theta =
            derivatives[amrex::Long(first) * point_count + point];
        const amrex::Real first_phi =
            derivatives[amrex::Long(modes + first) * point_count + point];
        const amrex::Real second_theta =
            derivatives[amrex::Long(second) * point_count + point];
        const amrex::Real second_phi =
            derivatives[amrex::Long(modes + second) * point_count + point];
        const amrex::Real mass_value =
            weight * (first_theta * second_theta +
                      inverse_sine_squared * first_phi * second_phi);
        const amrex::Real first_tt =
            strain[amrex::Long(first) * point_count + point];
        const amrex::Real first_tp =
            strain[amrex::Long(modes + first) * point_count + point];
        const amrex::Real first_pp =
            strain[amrex::Long(2 * modes + first) * point_count + point];
        const amrex::Real second_tt =
            strain[amrex::Long(second) * point_count + point];
        const amrex::Real second_tp =
            strain[amrex::Long(modes + second) * point_count + point];
        const amrex::Real second_pp =
            strain[amrex::Long(2 * modes + second) * point_count + point];
        const amrex::Real stiffness_value =
            weight *
            (first_tt * second_tt +
             2 * inverse_sine_squared * first_tp * second_tp +
             inverse_sine_squared * inverse_sine_squared * first_pp *
                 second_pp);
        amrex::Gpu::Atomic::AddNoRet(matrices + first * modes + second,
                                     mass_value);
        amrex::Gpu::Atomic::AddNoRet(
            matrices + modes * modes + first * modes + second,
            stiffness_value);
      });

  constexpr int dense_dimension = 3;
  amrex::Gpu::DeviceVector<amrex::Real> dense_stiffness{
      12, 6, 3, 6, 4, 1, 3, 1, amrex::Real(5.5)};
  amrex::Gpu::DeviceVector<amrex::Real> dense_mass{
      4, 2, 1, 2, 2, 0, 1, 0, amrex::Real(2.75)};
  amrex::Gpu::DeviceVector<amrex::Real> dense_factor(9, 0);
  amrex::Gpu::DeviceVector<amrex::Real> dense_work(9, 0);
  amrex::Gpu::DeviceVector<amrex::Real> dense_vectors(9, 0);
  amrex::Gpu::DeviceVector<amrex::Real> dense_coefficients(3, 0);
  amrex::Gpu::DeviceVector<amrex::Real> dense_result(5, 0);
  amrex::Gpu::DeviceVector<int> dense_status(2, 0);
  amrex::Gpu::DeviceVector<amrex::Real> legendre_result(
      2 * multipole_component_count, 0);
  const auto *const dense_k = dense_stiffness.dataPtr();
  const auto *const dense_m = dense_mass.dataPtr();
  auto *const dense_l = dense_factor.dataPtr();
  auto *const dense_w = dense_work.dataPtr();
  auto *const dense_v = dense_vectors.dataPtr();
  auto *const dense_c = dense_coefficients.dataPtr();
  auto *const dense_output = dense_result.dataPtr();
  auto *const status = dense_status.dataPtr();
  auto *const legendre = legendre_result.dataPtr();
  amrex::ParallelFor(
      1, [=] AMREX_GPU_DEVICE(const int) noexcept {
        amrex::Real eigenvalue = 0;
        amrex::Real residual = 0;
        int sweeps = 0;
        const auto result = smallest_generalized_symmetric_mode(
            dense_k, dense_m, dense_l, dense_w, dense_v, dense_c,
            eigenvalue, residual, sweeps, dense_dimension, 32,
            amrex::Real(1.0e-13));
        dense_output[0] = eigenvalue;
        dense_output[1] = residual;
        dense_output[2] = dense_c[0];
        dense_output[3] = dense_c[1];
        dense_output[4] = dense_c[2];
        status[0] = static_cast<int>(result);
        status[1] = sweeps;
      });
  amrex::ParallelFor(
      1, [=] AMREX_GPU_DEVICE(const int) noexcept {
        amrex::Real polynomial[multipole_component_count];
        amrex::Real derivative[multipole_component_count];
        if (!legendre_values_and_derivatives(
                amrex::Real(0.25), polynomial, derivative))
          return;
        for (int degree = 0; degree <= maximum_multipole_degree; ++degree) {
          legendre[degree] = polynomial[degree];
          legendre[multipole_component_count + degree] =
              derivative[degree];
        }
      });

  std::array<amrex::Real, 18> host_gram{};
  std::array<amrex::Real, 5> host_dense{};
  std::array<int, 2> host_status{};
  std::array<amrex::Real, 2 * multipole_component_count> host_legendre{};
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, gram.begin(), gram.end(),
                   host_gram.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, dense_result.begin(),
                   dense_result.end(), host_dense.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, dense_status.begin(),
                   dense_status.end(), host_status.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost, legendre_result.begin(),
                   legendre_result.end(), host_legendre.begin());

  AxialSelfTestResult result;
  for (int first = 0; first < modes; ++first)
    for (int second = 0; second < modes; ++second) {
      const amrex::Real expected = first == second ? 2 : 0;
      result.maximum_mass_error = amrex::max(
          result.maximum_mass_error,
          axial_absolute_value(host_gram[first * modes + second] -
                               expected));
      result.maximum_stiffness = amrex::max(
          result.maximum_stiffness,
          axial_absolute_value(
              host_gram[modes * modes + first * modes + second]));
    }
  result.dense_eigenvalue_error =
      axial_absolute_value(host_dense[0] - 1);
  const amrex::Real positive_error = amrex::max(
      axial_absolute_value(host_dense[2] + amrex::Real(0.5)),
      amrex::max(axial_absolute_value(host_dense[3] - 1),
                 axial_absolute_value(host_dense[4])));
  const amrex::Real negative_error = amrex::max(
      axial_absolute_value(host_dense[2] - amrex::Real(0.5)),
      amrex::max(axial_absolute_value(host_dense[3] + 1),
                 axial_absolute_value(host_dense[4])));
  result.dense_eigenvector_error =
      amrex::min(positive_error, negative_error);
  result.dense_residual = host_dense[1];
  result.dense_status = host_status[0];
  constexpr amrex::Real expected_polynomial[multipole_component_count]{
      1,
      amrex::Real(0.25),
      amrex::Real(-0.40625),
      amrex::Real(-0.3359375),
      amrex::Real(0.15771484375),
      amrex::Real(0.3397216796875),
      amrex::Real(0.0242767333984375),
      amrex::Real(-0.279918670654296875),
      amrex::Real(-0.15245401859283447265625)};
  constexpr amrex::Real expected_derivative[multipole_component_count]{
      0,
      1,
      amrex::Real(0.75),
      amrex::Real(-1.03125),
      amrex::Real(-1.6015625),
      amrex::Real(0.38818359375),
      amrex::Real(2.1353759765625),
      amrex::Real(0.7037811279296875),
      amrex::Real(-2.063404083251953125)};
  for (int degree = 0; degree <= maximum_multipole_degree; ++degree) {
    result.maximum_legendre_error = amrex::max(
        result.maximum_legendre_error,
        axial_absolute_value(host_legendre[degree] -
                             expected_polynomial[degree]));
    result.maximum_legendre_derivative_error = amrex::max(
        result.maximum_legendre_derivative_error,
        axial_absolute_value(
            host_legendre[multipole_component_count + degree] -
            expected_derivative[degree]));
  }
  return result;
}

} // namespace

extern "C" void ParticleAHFinderX_AxialSymmetrySelfTest(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  if (!run_axial_symmetry_self_test ||
      state.axial_symmetry_self_test_complete)
    return;
  const AxialSelfTestResult result = run_axial_self_test();
  CCTK_VINFO(
      "Axial-symmetry accelerator self-test: mass/stiffness/dense-eigen/"
      "dense-vector/dense-residual/Legendre/Legendre-derivative "
      "errors=%.3g/%.3g/%.3g/%.3g/%.3g/%.3g/%.3g status=%d",
      double(result.maximum_mass_error), double(result.maximum_stiffness),
      double(result.dense_eigenvalue_error),
      double(result.dense_eigenvector_error), double(result.dense_residual),
      double(result.maximum_legendre_error),
      double(result.maximum_legendre_derivative_error),
      result.dense_status);
  if (result.maximum_mass_error > amrex::Real(2.0e-3) ||
      result.maximum_stiffness > amrex::Real(1.0e-10) ||
      result.dense_eigenvalue_error > amrex::Real(1.0e-12) ||
      result.dense_eigenvector_error > amrex::Real(1.0e-12) ||
      result.dense_residual > amrex::Real(1.0e-12) ||
      result.maximum_legendre_error > amrex::Real(1.0e-14) ||
      result.maximum_legendre_derivative_error > amrex::Real(1.0e-13) ||
      result.dense_status != static_cast<int>(AxialSymmetryStatus::valid))
    CCTK_VERROR("ParticleAHFinderX axial-symmetry accelerator self-test "
                "failed");
  CCTK_INFO("Variational axial-symmetry accelerator self-test passed");
  state.axial_symmetry_self_test_complete = true;
}

} // namespace ParticleAHFinderX
