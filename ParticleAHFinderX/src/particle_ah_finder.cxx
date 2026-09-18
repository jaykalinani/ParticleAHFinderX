/**
 * \file particle_ah_finder.cxx
 * \brief Cactus scheduling and persistent surface lifecycle.
 *
 * Configured particles are born once, remain in their AMReX containers, and
 * are never recreated by analysis cadence. Surface relaxation is an internal
 * pseudo-time solve on a frozen physical slice and does not participate in
 * the physical evolution integrator's stage schedule.
 */
#include "runtime.hxx"
#include "candidate_manager.hxx"
#include "surface_manager.hxx"
#include "surface_router.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <driver.hxx>

#include <AMReX_Config.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ParticleAHFinderX {

static_assert(AMREX_SPACEDIM == 3,
              "ParticleAHFinderX requires three-dimensional AMReX");
static_assert(sizeof(CCTK_REAL) == sizeof(amrex::Real),
              "Cactus and AMReX real precision must match");

namespace {

constexpr std::array<const char *, 12> adm_variable_names{{
    "ADMBaseX::gxx", "ADMBaseX::gxy", "ADMBaseX::gxz",
    "ADMBaseX::gyy", "ADMBaseX::gyz", "ADMBaseX::gzz",
    "ADMBaseX::kxx", "ADMBaseX::kxy", "ADMBaseX::kxz",
    "ADMBaseX::kyy", "ADMBaseX::kyz", "ADMBaseX::kzz",
}};

class SolverActivity final {
public:
  explicit SolverActivity(Runtime &state) : state_(state) {
    if (state_.solver_in_flight)
      CCTK_ERROR("ParticleAHFinderX surface work was entered recursively");
    state_.solver_in_flight = true;
  }

  ~SolverActivity() { state_.solver_in_flight = false; }

  SolverActivity(const SolverActivity &) = delete;
  SolverActivity &operator=(const SolverActivity &) = delete;

private:
  Runtime &state_;
};

bool policy_uses_external(const char *const policy) {
  return CCTK_EQUALS(policy, "external") ||
         CCTK_EQUALS(policy, "puncture_tracker") ||
         CCTK_EQUALS(policy, "puncture_tracker_displacement") ||
         CCTK_EQUALS(policy, "external_then_centroid") ||
         CCTK_EQUALS(policy, "centroid_then_external") ||
         CCTK_EQUALS(policy, "blend_external_centroid");
}

std::string base_variable_name(const char *const configured_name) {
  std::string name(configured_name ? configured_name : "");
  if (name.empty() || name.back() != ']')
    return name;

  const auto open = name.rfind('[');
  if (open == std::string::npos || open + 1 == name.size() - 1)
    return name;
  for (std::size_t i = open + 1; i + 1 < name.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(name[i])))
      return name;

  // Cactus vector-group elements are registered as names ending in an
  // explicit index, for example PunctureTracker::pt_loc_x[0]. Resolve the
  // first element to obtain the vector storage base, then apply the requested
  // element offset when reading the scalar array. Removing the brackets
  // entirely produces a name which CCTK_VarIndex cannot resolve.
  name.replace(open + 1, name.size() - open - 2, "0");
  return name;
}

int variable_array_index(const char *const configured_name) {
  const std::string name(configured_name ? configured_name : "");
  if (name.empty() || name.back() != ']')
    return 0;
  const auto open = name.rfind('[');
  if (open == std::string::npos || open + 1 == name.size() - 1)
    return -1;
  long long index = 0;
  for (std::size_t i = open + 1; i + 1 < name.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(name[i])))
      return -1;
    const int digit = name[i] - '0';
    if (index > (std::numeric_limits<int>::max() - digit) / 10)
      return -1;
    index = 10 * index + digit;
  }
  return static_cast<int>(index);
}

int resolve_real_variable(const char *const configured_name,
                          const int required_group_type,
                          const char *const purpose,
                          int *const array_index = nullptr) {
  const std::string name = base_variable_name(configured_name);
  const int element = variable_array_index(configured_name);
  if (name.empty()) {
    CCTK_PARAMWARN(purpose);
    return -1;
  }

  const int variable = CCTK_VarIndex(name.c_str());
  if (variable < 0) {
    const std::string message =
        std::string("ParticleAHFinderX cannot resolve ") + purpose + " \"" +
        configured_name + "\"";
    CCTK_PARAMWARN(message.c_str());
    return -1;
  }

  const int group_index = CCTK_GroupIndexFromVarI(variable);
  cGroup group;
  if (group_index < 0 || CCTK_GroupData(group_index, &group) != 0 ||
      group.vartype != CCTK_VARIABLE_REAL ||
      group.grouptype != required_group_type) {
    const std::string message =
        std::string("ParticleAHFinderX requires ") + purpose + " \"" +
        configured_name + "\" to be a CCTK_REAL " +
        (required_group_type == CCTK_GF ? "grid function" : "scalar");
    CCTK_PARAMWARN(message.c_str());
    return -1;
  }
  if (required_group_type == CCTK_GF && group.dim != 3) {
    const std::string message =
        std::string("ParticleAHFinderX requires ") + purpose + " \"" +
        configured_name + "\" to be three-dimensional";
    CCTK_PARAMWARN(message.c_str());
    return -1;
  }
  const int vector_length = group.vectorgroup ? group.vectorlength : 1;
  if (element < 0 || element >= vector_length) {
    const std::string message =
        std::string("ParticleAHFinderX requires ") + purpose + " \"" +
        configured_name + "\" to select an array element in [0," +
        std::to_string(vector_length - 1) + "]";
    CCTK_PARAMWARN(message.c_str());
    return -1;
  }
  if (array_index)
    *array_index = element;
  return variable;
}

CenterPolicy configured_center_policy(const char *const policy) {
  if (CCTK_EQUALS(policy, "external") ||
      CCTK_EQUALS(policy, "puncture_tracker"))
    return CenterPolicy::external;
  if (CCTK_EQUALS(policy, "puncture_tracker_displacement"))
    return CenterPolicy::puncture_tracker_displacement;
  if (CCTK_EQUALS(policy, "centroid"))
    return CenterPolicy::centroid;
  if (CCTK_EQUALS(policy, "external_then_centroid"))
    return CenterPolicy::external_then_centroid;
  if (CCTK_EQUALS(policy, "centroid_then_external"))
    return CenterPolicy::centroid_then_external;
  if (CCTK_EQUALS(policy, "blend_external_centroid"))
    return CenterPolicy::blend_external_centroid;
  return CenterPolicy::manual;
}

SurfaceRole configured_surface_role(const char *const role) {
  if (CCTK_EQUALS(role, "common"))
    return SurfaceRole::common;
  if (CCTK_EQUALS(role, "other"))
    return SurfaceRole::other;
  return SurfaceRole::individual;
}

ResidualTolerances configured_tolerances(const int surface) {
  DECLARE_CCTK_PARAMETERS;
  if (CCTK_EQUALS(residual_profile[surface], "compatibility"))
    return {1.0e-2, 1.0e-5};
  if (CCTK_EQUALS(residual_profile[surface], "dynamic_tracking"))
    return {1.0e-5, 1.0e-4};
  return {custom_theta_l2_tolerance[surface],
          custom_theta_linf_tolerance[surface]};
}

int surface_cadence(const int surface) {
  DECLARE_CCTK_PARAMETERS;
  return find_every_surface[surface] >= 0 ? find_every_surface[surface]
                                          : find_every;
}

bool any_surface_due(const Runtime &state, const int iteration) {
  for (const auto &surface : state.surfaces)
    if (surface.lifecycle != ParticleLifecycle::retired &&
        surface.search_every > 0 && iteration >= surface.next_search_iteration &&
        iteration % surface.search_every == 0)
      return true;
  return false;
}

bool any_surface_has_search_cadence() {
  DECLARE_CCTK_PARAMETERS;
  for (int surface = 0; surface < num_surfaces; ++surface)
    if (surface_enabled[surface] && surface_cadence(surface) > 0)
      return true;
  return false;
}

const char *amrex_backend_name() {
#if defined(AMREX_USE_CUDA)
  return "CUDA";
#elif defined(AMREX_USE_HIP)
  return "HIP";
#elif defined(AMREX_USE_SYCL)
  return "SYCL";
#else
  return "CPU";
#endif
}

constexpr bool amrex_uses_accelerator() {
#ifdef AMREX_USE_GPU
  return true;
#else
  return false;
#endif
}

const char *relaxation_state_name(const SurfaceRelaxationState state) {
  switch (state) {
  case SurfaceRelaxationState::idle:
    return "idle";
  case SurfaceRelaxationState::active:
    return "active";
  case SurfaceRelaxationState::converged:
    return "converged";
  case SurfaceRelaxationState::refine:
    return "refine";
  case SurfaceRelaxationState::failed:
    return "failed";
  }
  return "unknown";
}

const char *surface_role_name(const SurfaceRole role) {
  switch (role) {
  case SurfaceRole::individual:
    return "individual";
  case SurfaceRole::common:
    return "common";
  case SurfaceRole::other:
    return "other";
  }
  return "unknown";
}

void report_relaxation_trials(Runtime &state, const int pseudo_step,
                              const amrex::Real pseudo_time,
                              const char *const phase) {
  for (const auto &diagnostics :
       state.logical_surfaces->local_surface_diagnostics()) {
    if (!diagnostics.due)
      continue;
    const auto found = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), diagnostics.stable_id,
        [](const SurfaceRecord &surface, const amrex::Long stable_id) {
          return surface.stable_id < stable_id;
        });
    if (found == state.surfaces.end() ||
        found->stable_id != diagnostics.stable_id)
      CCTK_ERROR("ParticleAHFinderX relaxation report found an unknown "
                 "logical surface ID");
    CCTK_VINFO(
        "Relaxation %s: step=%d tau=%.6e surface_id=%lld role=%s "
        "angular_level=%d/%d resolution=%dx%d state=%s "
        "M*L2=%.6e M*Linf=%.6e mean(Theta)=%.6e "
        "h[min,mean,max]=[%.6e,%.6e,%.6e] invalid=%lld levels=[%d,%d]",
        phase, pseudo_step, double(pseudo_time),
        static_cast<long long>(diagnostics.stable_id),
        surface_role_name(found->role), diagnostics.angular_level + 1,
        diagnostics.angular_levels,
        found->angular_layer[diagnostics.angular_level].ntheta,
        found->angular_layer[diagnostics.angular_level].nphi,
        relaxation_state_name(diagnostics.relaxation_state),
        double(found->mass_scale * diagnostics.expansion_l2),
        double(found->mass_scale * diagnostics.expansion_linf),
        double(diagnostics.mean_expansion),
        double(diagnostics.minimum_radius), double(diagnostics.mean_radius),
        double(diagnostics.maximum_radius),
        static_cast<long long>(diagnostics.invalid_points),
        diagnostics.minimum_sampled_level,
        diagnostics.maximum_sampled_level);
  }
}

struct LayerOutcome {
  SurfaceLayerSelection selection;
  SurfaceRelaxationState relaxation_state = SurfaceRelaxationState::idle;
  amrex::Real expansion_l2 = 0;
  amrex::Real expansion_linf = 0;
  amrex::Long invalid_points = 0;
};

struct LayerResidual {
  SurfaceLayerSelection selection;
  SurfaceRelaxationState relaxation_state = SurfaceRelaxationState::idle;
  amrex::Real expansion_l2 = 0;
  amrex::Real expansion_linf = 0;
  amrex::Long invalid_points = 0;
};

std::vector<LayerResidual> collect_layer_residuals(
    Runtime &state, const std::vector<SurfaceLayerSelection> &selections) {
  std::vector<amrex::Long> counts(selections.size(), 0);
  std::vector<amrex::Long> states(selections.size(), 0);
  std::vector<amrex::Long> invalid(selections.size(), 0);
  std::vector<amrex::Real> l2(selections.size(), 0);
  std::vector<amrex::Real> linf(selections.size(), 0);
  for (const auto &diagnostics :
       state.logical_surfaces->local_surface_diagnostics()) {
    if (!diagnostics.due)
      continue;
    const SurfaceLayerSelection key{diagnostics.stable_id,
                                    diagnostics.angular_level};
    const auto found = std::lower_bound(
        selections.begin(), selections.end(), key,
        [](const SurfaceLayerSelection &first,
           const SurfaceLayerSelection &second) {
          return first.stable_id < second.stable_id ||
                 (first.stable_id == second.stable_id &&
                  first.angular_level < second.angular_level);
        });
    if (found == selections.end() || found->stable_id != key.stable_id ||
        found->angular_level != key.angular_level)
      continue;
    const std::size_t index =
        static_cast<std::size_t>(found - selections.begin());
    counts[index] = 1;
    states[index] = static_cast<int>(diagnostics.relaxation_state);
    invalid[index] = diagnostics.invalid_points;
    l2[index] = diagnostics.expansion_l2;
    linf[index] = diagnostics.expansion_linf;
  }
  if (!selections.empty()) {
    if (selections.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
      CCTK_ERROR("ParticleAHFinderX layer-residual table exceeds the MPI "
                 "reduction count range");
    const int count = static_cast<int>(selections.size());
    amrex::ParallelDescriptor::ReduceLongSum(counts.data(), count);
    amrex::ParallelDescriptor::ReduceLongSum(states.data(), count);
    amrex::ParallelDescriptor::ReduceLongSum(invalid.data(), count);
    amrex::ParallelDescriptor::ReduceRealSum(l2.data(), count);
    amrex::ParallelDescriptor::ReduceRealSum(linf.data(), count);
  }

  std::vector<LayerResidual> residuals(selections.size());
  for (std::size_t index = 0; index < selections.size(); ++index) {
    if (counts[index] != 1)
      CCTK_VERROR("ParticleAHFinderX expected exactly one residual owner for "
                  "surface %lld angular level %d, found %lld",
                  static_cast<long long>(selections[index].stable_id),
                  selections[index].angular_level,
                  static_cast<long long>(counts[index]));
    residuals[index] = {
        selections[index],
        static_cast<SurfaceRelaxationState>(states[index]), l2[index],
        linf[index], invalid[index]};
  }
  return residuals;
}

std::vector<LayerOutcome> collect_layer_outcomes(
    Runtime &state, const std::vector<SurfaceLayerSelection> &selections) {
  std::vector<amrex::Long> counts(selections.size(), 0);
  std::vector<amrex::Long> states(selections.size(), 0);
  std::vector<amrex::Long> invalid(selections.size(), 0);
  std::vector<amrex::Real> l2(selections.size(), 0);
  std::vector<amrex::Real> linf(selections.size(), 0);
  for (const auto &diagnostics :
       state.logical_surfaces->local_surface_diagnostics()) {
    const SurfaceLayerSelection key{diagnostics.stable_id,
                                    diagnostics.angular_level};
    const auto found = std::lower_bound(
        selections.begin(), selections.end(), key,
        [](const SurfaceLayerSelection &first,
           const SurfaceLayerSelection &second) {
          return first.stable_id < second.stable_id ||
                 (first.stable_id == second.stable_id &&
                  first.angular_level < second.angular_level);
        });
    if (found == selections.end() || found->stable_id != key.stable_id ||
        found->angular_level != key.angular_level)
      continue;
    const std::size_t index =
        static_cast<std::size_t>(found - selections.begin());
    counts[index] = 1;
    states[index] = static_cast<int>(diagnostics.relaxation_state);
    invalid[index] = diagnostics.invalid_points;
    l2[index] = diagnostics.expansion_l2;
    linf[index] = diagnostics.expansion_linf;
  }
  if (!selections.empty()) {
    if (selections.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
      CCTK_ERROR("ParticleAHFinderX angular-layer outcome table exceeds the "
                 "MPI reduction count range");
    const int count = static_cast<int>(selections.size());
    amrex::ParallelDescriptor::ReduceLongSum(counts.data(), count);
    amrex::ParallelDescriptor::ReduceLongSum(states.data(), count);
    amrex::ParallelDescriptor::ReduceLongSum(invalid.data(), count);
    amrex::ParallelDescriptor::ReduceRealSum(l2.data(), count);
    amrex::ParallelDescriptor::ReduceRealSum(linf.data(), count);
  }

  std::vector<LayerOutcome> outcomes(selections.size());
  for (std::size_t index = 0; index < selections.size(); ++index) {
    if (counts[index] != 1)
      CCTK_VERROR("ParticleAHFinderX expected exactly one logical owner for "
                  "surface %lld angular level %d, found %lld",
                  static_cast<long long>(selections[index].stable_id),
                  selections[index].angular_level,
                  static_cast<long long>(counts[index]));
    auto relaxation_state =
        static_cast<SurfaceRelaxationState>(states[index]);
    if (relaxation_state == SurfaceRelaxationState::active)
      relaxation_state = SurfaceRelaxationState::failed;
    if (relaxation_state != SurfaceRelaxationState::converged &&
        relaxation_state != SurfaceRelaxationState::refine &&
        relaxation_state != SurfaceRelaxationState::failed)
      CCTK_VERROR("ParticleAHFinderX surface %lld angular level %d ended in "
                  "non-terminal relaxation state %lld",
                  static_cast<long long>(selections[index].stable_id),
                  selections[index].angular_level,
                  static_cast<long long>(states[index]));
    outcomes[index] = {selections[index], relaxation_state, l2[index],
                       linf[index], invalid[index]};
  }
  return outcomes;
}

amrex::Long selected_particle_count(
    const Runtime &state,
    const std::vector<SurfaceLayerSelection> &selections) {
  amrex::Long count = 0;
  for (const auto &selection : selections) {
    const auto found = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), selection.stable_id,
        [](const SurfaceRecord &surface, const amrex::Long stable_id) {
          return surface.stable_id < stable_id;
        });
    if (found == state.surfaces.end() ||
        found->stable_id != selection.stable_id ||
        selection.angular_level < 0 ||
        selection.angular_level >= found->angular_levels)
      CCTK_ERROR("ParticleAHFinderX selected an invalid angular layer");
    const amrex::Long layer_count =
        found->angular_layer[selection.angular_level].particle_count;
    if (layer_count <= 0 ||
        count > std::numeric_limits<amrex::Long>::max() - layer_count)
      CCTK_ERROR("ParticleAHFinderX selected particle count overflowed");
    count += layer_count;
  }
  return count;
}

} // namespace

Runtime &runtime() {
  static Runtime state;
  return state;
}

extern "C" void ParticleAHFinderX_ParamCheck(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  if (!active)
    return;

#ifndef HAVE_CAPABILITY_openPMD_api
  if (particle_out_every > 0)
    CCTK_PARAMWARN("ParticleAHFinderX parallel particle output was requested, "
                   "but this configuration lacks openPMD_api");
#endif

  if (interpolation_order == 3 && any_surface_has_search_cadence() &&
      (cctk_nghostzones[0] < 2 || cctk_nghostzones[1] < 2 ||
       cctk_nghostzones[2] < 2))
    CCTK_PARAMWARN("ParticleAHFinderX cubic interpolation requires at least "
                   "two ghost zones in every Cartesian direction");
  if (minimum_relaxation_steps > maximum_relaxation_steps)
    CCTK_PARAMWARN("ParticleAHFinderX minimum_relaxation_steps cannot exceed "
                   "maximum_relaxation_steps");
  if (run_domain_exit_self_test &&
      (!retire_on_domain_exit || domain_exit_grace_searches != 1 ||
       num_surfaces != 1 || !surface_enabled[0]))
    CCTK_PARAMWARN("ParticleAHFinderX domain-exit self-test requires exactly "
                   "one enabled configured surface, retire_on_domain_exit=yes, "
                   "and domain_exit_grace_searches=1");
  if (!(minimum_radius_factor > 0.0 && minimum_radius_factor < 1.0) ||
      !(maximum_radius_factor > 1.0))
    CCTK_PARAMWARN("ParticleAHFinderX relaxation radius factors must satisfy "
                   "0 < minimum < 1 < maximum");
  if (amrex::Long(candidate_nphi) != 2 * amrex::Long(candidate_ntheta) ||
      candidate_ntheta % 2 != 0 ||
      candidate_nphi % 2 != 0)
    CCTK_PARAMWARN("ParticleAHFinderX dynamic candidate resolution must be "
                   "even and satisfy nphi = 2*ntheta");
  if (!(candidate_expiry_factor > 1.0) ||
      !(candidate_maximum_radius_factor >= candidate_initial_radius_factor) ||
      candidate_retry_max_iterations < candidate_retry_base_iterations)
    CCTK_PARAMWARN("ParticleAHFinderX candidate expiry, radius bracket, and "
                   "retry bounds are inconsistent");
  if (export_spherical_surfaces &&
      !CCTK_IsThornActive("SphericalSurface"))
    CCTK_PARAMWARN("ParticleAHFinderX SphericalSurface export requires the "
                   "SphericalSurface thorn to be active");
  if (export_spherical_surfaces &&
      !CCTK_IsFunctionAliased("sf_IdFromName"))
    CCTK_PARAMWARN("ParticleAHFinderX SphericalSurface export requires the "
                   "sf_IdFromName provider");
  for (int slot = 0; export_spherical_surfaces &&
                     slot < max_compatibility_slots;
       ++slot)
    if (spherical_surface_index[slot] >= 0 &&
        spherical_surface_name[slot][0] != '\0')
      CCTK_PARAMWARN("ParticleAHFinderX compatibility slots must map by "
                     "either SphericalSurface index or name, not both");

  try {
    for (const char *const variable : adm_variable_names)
      resolve_mesh_field(variable);
  } catch (const std::exception &error) {
    CCTK_PARAMWARN(error.what());
  }

  for (int surface = 0; surface < num_surfaces; ++surface) {
    if (!surface_enabled[surface])
      continue;
    if (nphi[surface] != 2 * ntheta[surface])
      CCTK_PARAMWARN("ParticleAHFinderX version-one latitude/longitude "
                     "surfaces require nphi = 2*ntheta");
    if (ntheta[surface] % 2 != 0 || nphi[surface] % 2 != 0)
      CCTK_PARAMWARN("ParticleAHFinderX angular resolutions must be even");
    if (amrex::Long(ntheta[surface]) >
        std::numeric_limits<amrex::Long>::max() /
            amrex::Long(nphi[surface]))
      CCTK_PARAMWARN("ParticleAHFinderX angular surface size overflows the "
                     "AMReX particle index type");

    if (policy_uses_external(center_policy[surface])) {
      const std::array<const char *, 3> sources{{
          external_center_x[surface], external_center_y[surface],
          external_center_z[surface],
      }};
      for (const char *const source : sources)
        resolve_real_variable(source, CCTK_SCALAR,
                              "external center source");
    }

    const auto tolerances = configured_tolerances(surface);
    if (!(tolerances.l2_times_mass > 0.0) ||
        !(tolerances.linf_times_mass > 0.0))
      CCTK_PARAMWARN(
          "ParticleAHFinderX expansion tolerances must be positive");
  }
}

extern "C" void ParticleAHFinderX_Setup(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();

  state.num_patches = CarpetX::ghext->num_patches();
  if (state.num_patches != 1)
    CCTK_VERROR("ParticleAHFinderX version one supports one Cartesian "
                "CarpetX patch; a batched device coordinate map is required "
                "for %d patches",
                state.num_patches);

#ifdef AMREX_USE_GPU
  if (require_gpu_aware_mpi && amrex::ParallelDescriptor::NProcs() > 1 &&
      !amrex::ParallelDescriptor::UseGpuAwareMpi())
    CCTK_ERROR("ParticleAHFinderX requires GPU-aware MPI for multi-rank "
               "accelerator routing");
#endif

  if (state.containers.empty()) {
    state.containers.reserve(CarpetX::ghext->num_patches());
    for (auto &patch : CarpetX::ghext->patchdata)
      state.containers.emplace_back(
          std::make_unique<SurfaceParticleContainer>(patch.amrcore.get()));
  } else {
    // BASEGRID may be re-entered by a driver. Keep the same containers and
    // populations; only refresh their hierarchy metadata.
    for (auto &container : state.containers)
      container->resize_after_regrid();
  }
  if (!state.logical_surfaces)
    state.logical_surfaces = std::make_unique<LogicalSurfaceStorage>();

  try {
    for (std::size_t i = 0; i < adm_variable_names.size(); ++i)
      state.adm_fields[i] = resolve_mesh_field(adm_variable_names[i]);
  } catch (const std::exception &error) {
    CCTK_VERROR("%s", error.what());
  }

  for (int surface = 0; surface < num_surfaces; ++surface) {
    state.tolerances[surface] = configured_tolerances(surface);
    if (!policy_uses_external(center_policy[surface]))
      continue;
    const std::array<const char *, 3> sources{{
        external_center_x[surface], external_center_y[surface],
        external_center_z[surface],
    }};
    for (int d = 0; d < 3; ++d) {
      int array_index = 0;
      state.center_providers[surface].variable_indices[d] =
          resolve_real_variable(sources[d], CCTK_SCALAR,
                                "external center source", &array_index);
      state.center_providers[surface].array_indices[d] = array_index;
    }
  }

  state.setup = true;
  if (verbose)
    CCTK_VINFO("Configured %d surface(s), attached %zu persistent AMReX "
               "container(s), backend=%s, accelerator=%s, GPU-aware MPI=%s",
               int(num_surfaces), state.containers.size(),
               amrex_backend_name(),
               amrex_uses_accelerator() ? "yes" : "no",
               amrex::ParallelDescriptor::UseGpuAwareMpi() ? "yes" : "no");
}

extern "C" void ParticleAHFinderX_Initialise(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  if (!state.setup)
    CCTK_ERROR("ParticleAHFinderX initialization ran before setup");
  if (state.initial_surfaces_created)
    return;

  amrex::Long expected_particle_count = 0;
  std::vector<SurfaceSeed> pending_seeds;
  for (int surface = 0; surface < num_surfaces; ++surface) {
    if (!surface_enabled[surface])
      continue;

    if (state.next_surface_id ==
        std::numeric_limits<amrex::Long>::max())
      CCTK_ERROR("ParticleAHFinderX stable surface ID allocator overflowed");
    SurfaceSeed seed;
    seed.surface_id = state.next_surface_id++;
    seed.first_particle_id = state.next_particle_id;
    seed.generation = 0;
    seed.angular_level = 0;
    seed.angular_levels = 1;
    seed.ntheta = ntheta[surface];
    seed.nphi = nphi[surface];
    seed.center = {{initial_center_x[surface], initial_center_y[surface],
                    initial_center_z[surface]}};
    seed.radius = initial_radius[surface];
    seed.lifecycle = ParticleLifecycle::active;
    seed.role = configured_surface_role(initial_role[surface]);
    const amrex::Long count = seed.particle_count();

    SurfaceRecord record;
    record.stable_id = seed.surface_id;
    record.first_particle_id = seed.first_particle_id;
    record.particle_count = count;
    record.configured_slot = surface;
    record.compatibility_slot = surface + 1;
    record.search_every = surface_cadence(surface);
    record.next_search_iteration = 0;
    record.birth_iteration = cctk_iteration;
    record.last_state_change_iteration = cctk_iteration;
    record.generation = seed.generation;
    record.ntheta = seed.ntheta;
    record.nphi = seed.nphi;
    record.center_policy = configured_center_policy(center_policy[surface]);
    record.predict_center = predict_centers[surface];
    record.external_center_weight = external_center_weight[surface];
    record.manual_center = seed.center;
    record.center = seed.center;
    record.radius = seed.radius;
    record.mass_scale = mass_scale[surface];
    record.tolerances = state.tolerances[surface];
    record.lifecycle = seed.lifecycle;
    record.role = seed.role;
    record.birth_time = cctk_time;
    record.published = true;
    initialize_angular_layers(
        record, enable_angular_continuation,
        record.role == SurfaceRole::common
            ? candidate_angular_continuation_min_ntheta
            : angular_continuation_min_ntheta,
        angular_continuation_max_levels);
    const amrex::Long total_count = total_angular_particle_count(record);
    if (state.next_particle_id >
        std::numeric_limits<amrex::Long>::max() - total_count)
      CCTK_ERROR("ParticleAHFinderX stable particle ID allocator overflowed");
    if (expected_particle_count >
        std::numeric_limits<amrex::Long>::max() - total_count)
      CCTK_ERROR("ParticleAHFinderX configured particle population "
                 "overflowed the global count type");
    state.surfaces.push_back(record);
    state.next_particle_id += total_count;
    expected_particle_count += total_count;
  }

  assign_logical_owners(state.surfaces);
  pending_seeds = particle_seeds(state.surfaces);
  // This is the only initial-population allocation site. All configured
  // births share one batched device pass; future candidates use the same
  // lifecycle operation only at their quiescent birth boundary.
  for (auto &container : state.containers)
    container->create_surfaces(pending_seeds);
  state.population_creation_events +=
      static_cast<amrex::Long>(pending_seeds.size());
  state.logical_surfaces->initialize(local_logical_seeds(state.surfaces));

  for (auto &container : state.containers) {
    container->rebuild_amr_masks();
    container->redistribute_global();
  }

  amrex::Long actual_particle_count = 0;
  for (const auto &container : state.containers)
    actual_particle_count += container->local_particle_count();
  amrex::ParallelDescriptor::ReduceLongSum(actual_particle_count);
  if (actual_particle_count != expected_particle_count)
    CCTK_VERROR("ParticleAHFinderX created %lld particles but expected %lld",
                static_cast<long long>(actual_particle_count),
                static_cast<long long>(expected_particle_count));

  state.global_particle_count = actual_particle_count;
  state.initial_surfaces_created = true;
  amrex::Long global_logical_count =
      state.logical_surfaces->local_point_count();
  amrex::ParallelDescriptor::ReduceLongSum(global_logical_count);
  if (global_logical_count != expected_particle_count)
    CCTK_VERROR("ParticleAHFinderX created %lld logical points but expected "
                "%lld",
                static_cast<long long>(global_logical_count),
                static_cast<long long>(expected_particle_count));
  if (verbose) {
    CCTK_VINFO("Created %zu persistent surface population(s) containing %lld "
               "AMReX particles; ordinary searches will reuse them in place",
               state.surfaces.size(),
               static_cast<long long>(state.global_particle_count));
    CCTK_VINFO("Logical owner rank %d holds %d complete surface(s) and %lld "
               "persistent device point(s)",
               amrex::ParallelDescriptor::MyProc(),
               state.logical_surfaces->local_surface_count(),
               static_cast<long long>(
                   state.logical_surfaces->local_point_count()));
  }
}

extern "C" void ParticleAHFinderX_Regrid(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  auto &state = runtime();
  if (!state.setup)
    return;
  for (auto &container : state.containers) {
    container->resize_after_regrid();
    container->rebuild_amr_masks();
    container->redistribute_global();
  }
  if (state.regrid_events == std::numeric_limits<amrex::Long>::max())
    CCTK_ERROR("ParticleAHFinderX regrid event counter overflowed");
  ++state.regrid_events;
  state.last_regrid_iteration = cctk_iteration;
}

extern "C" void ParticleAHFinderX_Analysis(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  if (!state.setup)
    CCTK_ERROR("ParticleAHFinderX analysis ran before setup");
  if (!hierarchy_is_time_aligned())
    return;
  amrex::Long expected_creation_events = 0;
  for (const auto &surface : state.surfaces)
    expected_creation_events += surface.angular_levels;
  if (!state.initial_surfaces_created ||
      state.population_creation_events != expected_creation_events ||
      !state.logical_surfaces || !state.logical_surfaces->initialized() ||
      state.logical_surfaces->creation_events() < 1)
    CCTK_ERROR("ParticleAHFinderX persistent population lifecycle invariant "
               "was violated");
  SolverActivity solver_activity(state);
  manage_candidates_before_search(
      state, cctk_iteration, static_cast<amrex::Real>(cctk_time));
  if (!any_surface_due(state, cctk_iteration))
    return;

  std::vector<amrex::Long> due_surface_ids;
  for (const auto &surface : state.surfaces) {
    if (surface.lifecycle != ParticleLifecycle::retired &&
        surface.search_every > 0 &&
        cctk_iteration >= surface.next_search_iteration &&
        cctk_iteration % surface.search_every == 0)
      due_surface_ids.push_back(surface.stable_id);
  }
  if (state.last_center_prediction_iteration != cctk_iteration) {
    predict_surface_centers(
        cctkGH, state, due_surface_ids, cctk_iteration,
        static_cast<amrex::Real>(cctk_time),
        maximum_center_extrapolation_intervals);
    state.last_center_prediction_iteration = cctk_iteration;
  }
  if (due_surface_ids.empty())
    return;
  std::vector<SurfaceLayerSelection> due_surface_layers;
  std::vector<SurfaceLayerSelection> bootstrap_surface_layers;
  due_surface_layers.reserve(due_surface_ids.size());
  for (auto &surface : state.surfaces) {
    if (!std::binary_search(due_surface_ids.begin(), due_surface_ids.end(),
                            surface.stable_id))
      continue;
    surface.active_angular_level =
        surface.last_success_iteration >= 0 ? surface.angular_levels - 1 : 0;
    due_surface_layers.push_back(
        {surface.stable_id, surface.active_angular_level});
    if (surface.last_success_iteration < 0)
      for (int level = 0; level < surface.angular_levels; ++level)
        bootstrap_surface_layers.push_back({surface.stable_id, level});
    else
      bootstrap_surface_layers.push_back(
          {surface.stable_id, surface.active_angular_level});
  }
  state.logical_surfaces->set_due_surface_layers(bootstrap_surface_layers);
  for (auto &container : state.containers)
    container->set_active_surface_layers(bootstrap_surface_layers);
  amrex::Long active_particle_count =
      selected_particle_count(state, bootstrap_surface_layers);
  for (auto &container : state.containers)
    container->begin_sampled_level_tracking();

  try {
    ensure_mesh_fields_ready(
        cctkGH,
        std::vector<MeshFieldRef>(state.adm_fields.begin(),
                                  state.adm_fields.end()));
    if (!CarpetX::active_levels)
      throw std::runtime_error(
          "ParticleAHFinderX analysis has no active CarpetX levels");

    const auto gather = [&](const AdmGatherPass pass) {
      for (auto &container : state.containers)
        container->begin_adm_gather();

      bool gathered_level = false;
      CarpetX::active_levels->loop_serially([&](auto &level) {
        const AdmFieldLayout fields =
            adm_field_layout(level, state.adm_fields);
        if (!state.reported_adm_centering) {
          const bool vertex_centered = fields.centering.nodal[0] &&
                                       fields.centering.nodal[1] &&
                                       fields.centering.nodal[2];
          CCTK_VINFO(
              "Sampling ADMBaseX metric/curvature directly from native %s "
              "data on the AMReX device; nodal flags=[%d,%d,%d], no "
              "vertex-to-cell conversion",
              vertex_centered ? "vertex-centered" : "non-vertex-centered",
              fields.centering.nodal[0], fields.centering.nodal[1],
              fields.centering.nodal[2]);
          state.reported_adm_centering = true;
        }
        state.containers.at(level.patch)
            ->gather_adm_level(level.level, *fields.metric,
                               fields.metric_component, *fields.curv,
                               fields.curv_component, fields.centering,
                               interpolation_order, pass);
        gathered_level = true;
      });
      if (!gathered_level)
        throw std::runtime_error(
            "ParticleAHFinderX found no active level for the ADM gather");

      std::array<amrex::Long, 7> global_stats{{0, 0, 0, 0, 0, 0, 0}};
      for (const auto &container : state.containers) {
        const AdmGatherStats stats = container->local_adm_gather_stats();
        global_stats[0] += stats.sampled;
        global_stats[1] += stats.not_sampled;
        global_stats[2] += stats.needs_local_redistribute;
        global_stats[3] += stats.needs_hierarchy_redistribute;
        global_stats[4] += stats.outside_supported_domain;
        global_stats[5] += stats.invalid_ghost_data;
        global_stats[6] += stats.nonfinite;
      }
      amrex::ParallelDescriptor::ReduceLongSum(global_stats.data(),
                                                int(global_stats.size()));
      return global_stats;
    };

    const auto sample_adm = [&]() {
      auto global_stats = gather(AdmGatherPass::initial);
      if (global_stats[2] > 0) {
        if (verbose)
          CCTK_VINFO("Retrying %lld ADM stencil miss(es) after bounded "
                     "AMReX redistribution",
                     static_cast<long long>(global_stats[2]));
        for (auto &container : state.containers)
          container->redistribute_local_hierarchy(max_num_cells_moved);
        global_stats = gather(AdmGatherPass::local_retry);
      }
      if (global_stats[2] > 0 || global_stats[3] > 0) {
        if (verbose)
          CCTK_VINFO("Retrying %lld ADM ownership/stencil miss(es) after "
                     "hierarchy redistribution",
                     static_cast<long long>(global_stats[2] +
                                            global_stats[3]));
        for (auto &container : state.containers)
          container->redistribute_global();
        global_stats = gather(AdmGatherPass::final);
      }

      amrex::Long classified = 0;
      for (const auto count : global_stats)
        classified += count;
      if (classified != active_particle_count ||
          global_stats[1] != 0 || global_stats[2] != 0 ||
          global_stats[3] != 0 || global_stats[4] != 0 ||
          global_stats[5] != 0 || global_stats[6] != 0)
        CCTK_VERROR(
            "ParticleAHFinderX ADM gather failed: sampled=%lld "
            "not_sampled=%lld needs_local_redistribute=%lld "
            "needs_hierarchy_redistribute=%lld outside_domain=%lld "
            "invalid_ghost_data=%lld nonfinite=%lld expected=%lld",
            static_cast<long long>(global_stats[0]),
            static_cast<long long>(global_stats[1]),
            static_cast<long long>(global_stats[2]),
            static_cast<long long>(global_stats[3]),
            static_cast<long long>(global_stats[4]),
            static_cast<long long>(global_stats[5]),
            static_cast<long long>(global_stats[6]),
            static_cast<long long>(active_particle_count));
      state.last_adm_gather_iteration = cctk_iteration;
    };

    const auto route_and_evaluate = [&]() {
      const auto routing = route_particles_to_logical_owners(
          state.containers, *state.logical_surfaces,
          LogicalRoutingMode::sampled_fields);
      amrex::Long global_routing[5]{routing.sent, routing.received,
                                    routing.missing, routing.duplicate,
                                    routing.invalid};
      amrex::ParallelDescriptor::ReduceLongSum(global_routing, 5);
      if (global_routing[0] != active_particle_count ||
          global_routing[1] != active_particle_count ||
          global_routing[2] != 0 || global_routing[3] != 0 ||
          global_routing[4] != 0)
        CCTK_VERROR(
            "ParticleAHFinderX logical routing failed: sent=%lld "
            "received=%lld missing=%lld duplicate=%lld invalid=%lld "
            "expected=%lld",
            static_cast<long long>(global_routing[0]),
            static_cast<long long>(global_routing[1]),
            static_cast<long long>(global_routing[2]),
            static_cast<long long>(global_routing[3]),
            static_cast<long long>(global_routing[4]),
            static_cast<long long>(active_particle_count));
      state.logical_surfaces->evaluate_angular_derivatives();
      state.logical_surfaces->evaluate_expansion();
      const ExpansionStats local_expansion =
          state.logical_surfaces->local_expansion_stats();
      amrex::Real expansion_sums[2]{
          local_expansion.area, local_expansion.theta_squared_integral};
      amrex::Real expansion_linf = local_expansion.theta_linf;
      amrex::Long invalid_expansion = local_expansion.invalid;
      amrex::ParallelDescriptor::ReduceRealSum(expansion_sums, 2);
      amrex::ParallelDescriptor::ReduceRealMax(expansion_linf);
      amrex::ParallelDescriptor::ReduceLongSum(invalid_expansion);
      if (invalid_expansion != 0 || !(expansion_sums[0] > 0))
        CCTK_VERROR("ParticleAHFinderX expansion evaluation failed at %lld "
                    "point(s); total area=%.17g",
                    static_cast<long long>(invalid_expansion),
                    double(expansion_sums[0]));
      const amrex::Real mean_theta_squared =
          expansion_sums[1] / expansion_sums[0];
      state.last_expansion_l2 =
          mean_theta_squared > 0
              ? mean_theta_squared * amrex::Math::rsqrt(mean_theta_squared)
              : 0;
      state.last_expansion_linf = expansion_linf;
      state.last_logical_route_iteration = cctk_iteration;
    };

    const auto refresh_source_metadata = [&]() {
      const auto routing = route_particles_to_logical_owners(
          state.containers, *state.logical_surfaces,
          LogicalRoutingMode::source_metadata);
      amrex::Long global_routing[5]{routing.sent, routing.received,
                                    routing.missing, routing.duplicate,
                                    routing.invalid};
      amrex::ParallelDescriptor::ReduceLongSum(global_routing, 5);
      if (global_routing[0] != active_particle_count ||
          global_routing[1] != active_particle_count ||
          global_routing[2] != 0 || global_routing[3] != 0 ||
          global_routing[4] != 0)
        CCTK_VERROR(
            "ParticleAHFinderX source-metadata refresh failed: sent=%lld "
            "received=%lld missing=%lld duplicate=%lld invalid=%lld "
            "expected=%lld",
            static_cast<long long>(global_routing[0]),
            static_cast<long long>(global_routing[1]),
            static_cast<long long>(global_routing[2]),
            static_cast<long long>(global_routing[3]),
            static_cast<long long>(global_routing[4]),
            static_cast<long long>(active_particle_count));
    };

    amrex::Long due_particle_count = 0;
    for (const auto &surface : state.surfaces)
      if (std::binary_search(due_surface_ids.begin(), due_surface_ids.end(),
                             surface.stable_id))
        due_particle_count +=
            surface.angular_layer[surface.active_angular_level]
                .particle_count;
    amrex::Real solver_wall_start = 0;
    if (solve_surfaces && print_timing_stats) {
      amrex::Gpu::streamSynchronize();
      amrex::ParallelDescriptor::Barrier();
      solver_wall_start = amrex::second();
    }

    sample_adm();
    route_and_evaluate();

    if (run_expansion_self_test &&
        !state.expansion_self_test_complete) {
      amrex::Real maximum_error =
          state.logical_surfaces->local_minkowski_expansion_error();
      amrex::Real maximum_circumference_error = 0;
      amrex::Real maximum_circumference_bound = 0;
      amrex::Real maximum_circumference_excess = 0;
      const amrex::Real roundoff_tolerance =
          amrex::Real(2.0e-10) +
          amrex::Real(10000) *
              std::numeric_limits<amrex::Real>::epsilon();
      for (const auto &diagnostics :
           state.logical_surfaces->local_surface_diagnostics()) {
        const auto found = std::lower_bound(
            state.surfaces.begin(), state.surfaces.end(),
            diagnostics.stable_id,
            [](const SurfaceRecord &surface, const amrex::Long stable_id) {
              return surface.stable_id < stable_id;
            });
        if (found == state.surfaces.end() ||
            found->stable_id != diagnostics.stable_id)
          CCTK_ERROR("ParticleAHFinderX expansion self-test found an unknown "
                     "logical surface ID");
        const amrex::Real expected =
            2 * amrex::Math::pi<amrex::Real>() * diagnostics.mean_radius;
        const int diagnostic_ntheta =
            found->angular_layer[diagnostics.angular_level].ntheta;
        const amrex::Real dtheta =
            amrex::Math::pi<amrex::Real>() / diagnostic_ntheta;
        const amrex::Real equator_interpolation =
            (9 * std::cos(dtheta / 2) - std::cos(3 * dtheta / 2)) / 8;
        for (int plane = 0; plane < 3; ++plane) {
          const amrex::Real error = std::abs(
              diagnostics.proper_circumference[plane] - expected);
          const amrex::Real bound =
              roundoff_tolerance +
              (plane == 0
                   ? amrex::Real(1.05) * expected *
                         std::abs(equator_interpolation - 1)
                   : 0);
          maximum_circumference_error =
              amrex::max(maximum_circumference_error, error);
          maximum_circumference_bound =
              amrex::max(maximum_circumference_bound, bound);
          maximum_circumference_excess = amrex::max(
              maximum_circumference_excess, error - bound);
        }
      }
      amrex::ParallelDescriptor::ReduceRealMax(maximum_error);
      amrex::ParallelDescriptor::ReduceRealMax(maximum_circumference_error);
      amrex::ParallelDescriptor::ReduceRealMax(maximum_circumference_bound);
      amrex::ParallelDescriptor::ReduceRealMax(maximum_circumference_excess);
      if (maximum_error > roundoff_tolerance ||
          maximum_circumference_excess > 0)
        CCTK_VERROR("ParticleAHFinderX expansion self-test failed: "
                    "expansion_error=%.17g circumference_error=%.17g "
                    "circumference_bound=%.17g roundoff_tolerance=%.17g",
                    double(maximum_error),
                    double(maximum_circumference_error),
                    double(maximum_circumference_bound),
                    double(roundoff_tolerance));
      CCTK_VINFO("Minkowski coordinate-sphere expansion self-test passed: "
                 "maximum |Theta-2/h|=%.3g circumference error=%.3g "
                 "(analytic cubic bound %.3g)",
                 double(maximum_error),
                 double(maximum_circumference_error),
                 double(maximum_circumference_bound));
      state.expansion_self_test_complete = true;
    }

    if (run_adm_gather_self_test &&
        !state.adm_gather_self_test_complete) {
      amrex::Real maximum_error = 0;
      for (const auto &container : state.containers)
        maximum_error = amrex::max(
            maximum_error, container->local_minkowski_adm_error());
      amrex::ParallelDescriptor::ReduceRealMax(maximum_error);
      const amrex::Real tolerance =
          5000 * std::numeric_limits<amrex::Real>::epsilon();
      if (maximum_error > tolerance)
        CCTK_VERROR("ParticleAHFinderX live ADM gather self-test failed: "
                    "error=%.17g tolerance=%.17g",
                    double(maximum_error), double(tolerance));
      CCTK_VINFO("Live native-centered ADM gather self-test passed: "
                 "maximum Minkowski value/gradient error=%.3g",
                 double(maximum_error));
      state.adm_gather_self_test_complete = true;
    }

    const auto route_updates = [&]() {
      const auto routing = route_updates_to_mesh_owners(
          *state.logical_surfaces, state.containers);
      amrex::Long global_routing[5]{routing.sent, routing.received,
                                    routing.missing, routing.duplicate,
                                    routing.invalid};
      amrex::ParallelDescriptor::ReduceLongSum(global_routing, 5);
      if (global_routing[0] != active_particle_count ||
          global_routing[1] != active_particle_count ||
          global_routing[2] != 0 || global_routing[3] != 0 ||
          global_routing[4] != 0)
        CCTK_VERROR(
            "ParticleAHFinderX reverse routing failed: sent=%lld "
            "received=%lld missing=%lld duplicate=%lld invalid=%lld "
            "expected=%lld",
            static_cast<long long>(global_routing[0]),
            static_cast<long long>(global_routing[1]),
            static_cast<long long>(global_routing[2]),
            static_cast<long long>(global_routing[3]),
            static_cast<long long>(global_routing[4]),
            static_cast<long long>(active_particle_count));
    };

    const auto convergence = [&](const bool accept_convergence) {
      auto result = state.logical_surfaces->update_solver_convergence(
          accept_convergence, angular_continuation_cutoff_threshold);
      amrex::Real maxima[4]{result.maximum_ratio,
                            result.maximum_l2_times_mass,
                            result.maximum_linf_times_mass,
                            result.maximum_cutoff_indicator};
      amrex::Long counts[5]{result.invalid, result.active, result.converged,
                            result.refine, result.failed};
      amrex::ParallelDescriptor::ReduceRealMax(maxima, 4);
      amrex::ParallelDescriptor::ReduceLongSum(counts, 5);
      result.maximum_ratio = maxima[0];
      result.maximum_l2_times_mass = maxima[1];
      result.maximum_linf_times_mass = maxima[2];
      result.maximum_cutoff_indicator = maxima[3];
      result.invalid = counts[0];
      result.active = counts[1];
      result.converged = counts[2];
      result.refine = counts[3];
      result.failed = counts[4];
      return result;
    };

    if (solve_surfaces) {
      const RelaxationIntegrator integrator =
          CCTK_EQUALS(relaxation_integrator, "ssprk3")
              ? RelaxationIntegrator::ssprk3
              : RelaxationIntegrator::rk4;
      const int num_relaxation_stages =
          relaxation_stage_count(integrator);
      std::vector<SurfaceLayerSelection> pending_layers = due_surface_layers;
      std::vector<amrex::Long> converged_surface_ids;
      std::vector<amrex::Long> failed_surface_ids;
      std::vector<SurfaceLayerSelection> searched_target_layers;
      std::vector<int> surface_steps(due_surface_ids.size(), 0);
      int total_completed_steps = 0;
      amrex::Long processed_point_steps = 0;
      int total_extrapolation_trials = 0;
      amrex::Long extrapolation_point_evaluations = 0;
      amrex::Real maximum_terminal_l2_times_mass = 0;
      amrex::Real maximum_terminal_linf_times_mass = 0;
      amrex::Real maximum_terminal_ratio = 0;

      while (!pending_layers.empty()) {
        for (const auto &selection : pending_layers) {
          const auto found = std::lower_bound(
              state.surfaces.begin(), state.surfaces.end(),
              selection.stable_id,
              [](const SurfaceRecord &surface, const amrex::Long stable_id) {
                return surface.stable_id < stable_id;
              });
          if (found == state.surfaces.end() ||
              found->stable_id != selection.stable_id)
            CCTK_ERROR("ParticleAHFinderX lost a selected angular layer");
          if (selection.angular_level + 1 == found->angular_levels)
            searched_target_layers.push_back(selection);
        }
        state.logical_surfaces->set_due_surface_layers(pending_layers);
        for (auto &container : state.containers)
          container->set_active_surface_layers(pending_layers);
        active_particle_count =
            selected_particle_count(state, pending_layers);
        state.logical_surfaces->begin_relaxation_search(
            eta_damping_times_mass);
        auto residual = convergence(minimum_relaxation_steps == 0);
        int completed_steps = 0;
        amrex::Real completed_pseudo_time = 0;
        struct ExtrapolationSchedule {
          SurfaceLayerSelection selection;
          amrex::Real mass_scale = 1;
          int next_event = 1;
          bool reference_valid = false;
        };
        std::vector<ExtrapolationSchedule> extrapolation_schedule;
        extrapolation_schedule.reserve(pending_layers.size());
        for (const auto &selection : pending_layers) {
          const auto found = std::lower_bound(
              state.surfaces.begin(), state.surfaces.end(),
              selection.stable_id,
              [](const SurfaceRecord &surface, const amrex::Long stable_id) {
                return surface.stable_id < stable_id;
              });
          if (found == state.surfaces.end() ||
              found->stable_id != selection.stable_id)
            CCTK_ERROR("ParticleAHFinderX extrapolation schedule lost a "
                       "selected surface");
          extrapolation_schedule.push_back(
              {selection, found->mass_scale, 1, false});
        }
        const auto test_relaxation_extrapolation =
            [&](const std::vector<SurfaceLayerSelection> &requested_layers) {
              const auto requested_residuals =
                  collect_layer_residuals(state, requested_layers);
              std::vector<SurfaceLayerSelection> active_layers;
              std::vector<const SurfaceRecord *> active_surfaces;
              std::vector<RelaxationExtrapolationMode> active_modes;
              std::vector<amrex::Real> baseline_ratio;
              std::vector<amrex::Real> baseline_l2;
              std::vector<amrex::Real> baseline_linf;
              for (const auto &entry : requested_residuals) {
                const auto found = std::lower_bound(
                    state.surfaces.begin(), state.surfaces.end(),
                    entry.selection.stable_id,
                    [](const SurfaceRecord &surface,
                       const amrex::Long stable_id) {
                      return surface.stable_id < stable_id;
                    });
                if (found == state.surfaces.end() ||
                    found->stable_id != entry.selection.stable_id)
                  CCTK_ERROR("ParticleAHFinderX extrapolation residual lost "
                             "its surface metadata");
                const amrex::Real ratio = relaxation_residual_ratio(
                    entry.expansion_l2, entry.expansion_linf,
                    found->mass_scale, found->tolerances.l2_times_mass,
                    found->tolerances.linf_times_mass);
                if (entry.relaxation_state ==
                        SurfaceRelaxationState::active &&
                    entry.invalid_points == 0 &&
                    ratio > 0 && std::isfinite(ratio)) {
                  active_layers.push_back(entry.selection);
                  active_surfaces.push_back(&*found);
                  active_modes.push_back(
                      entry.selection.angular_level + 1 ==
                              found->angular_levels
                          ? RelaxationExtrapolationMode::mean_height
                          : RelaxationExtrapolationMode::full_shape);
                  baseline_ratio.push_back(ratio);
                  baseline_l2.push_back(entry.expansion_l2);
                  baseline_linf.push_back(entry.expansion_linf);
                }
              }
              if (active_layers.empty())
                return;

              state.logical_surfaces->begin_extrapolation_trials(
                  active_layers);
              std::vector<amrex::Real> best_ratio = baseline_ratio;
              std::vector<amrex::Real> best_l2 = baseline_l2;
              std::vector<amrex::Real> best_linf = baseline_linf;
              std::vector<amrex::Real> best_overstep(active_layers.size(),
                                                     amrex::Real(1));
              std::vector<bool> searching(active_layers.size(), true);
              std::vector<SurfaceExtrapolationControl> controls(
                  active_layers.size());

              const auto evaluate_controls =
                  [&](const bool reset_accepted_velocity) {
                    state.logical_surfaces->set_extrapolation_trial(
                        controls, eta_damping_times_mass,
                        minimum_radius_factor, maximum_radius_factor,
                        reset_accepted_velocity);
                    state.logical_surfaces->rebuild_positions();
                    route_updates();
                    // Extrapolation can intentionally move farther than one
                    // ordinary CFL-limited step.  Trial populations therefore
                    // take the safe hierarchy-synchronized redistribution.
                    for (auto &container : state.containers)
                      container->redistribute_global();
                    sample_adm();
                    route_and_evaluate();
                    route_updates();
                    if (extrapolation_point_evaluations >
                        std::numeric_limits<amrex::Long>::max() -
                            active_particle_count)
                      CCTK_ERROR("ParticleAHFinderX extrapolation point "
                                 "counter overflowed");
                    extrapolation_point_evaluations +=
                        active_particle_count;
                  };

              for (amrex::Real overstep = 2;
                   overstep <= maximum_relaxation_extrapolation_factor;
                   overstep *= amrex::Real(1.2)) {
                bool any_searching = false;
                for (std::size_t index = 0; index < active_layers.size();
                     ++index) {
                  controls[index] = {
                      active_layers[index].stable_id,
                      active_layers[index].angular_level,
                      searching[index] ? overstep : amrex::Real(1),
                      active_modes[index]};
                  any_searching |= searching[index];
                }
                if (!any_searching)
                  break;
                evaluate_controls(false);
                ++total_extrapolation_trials;
                const auto trial_residuals =
                    collect_layer_residuals(state, active_layers);
                for (std::size_t index = 0; index < active_layers.size();
                     ++index) {
                  if (!searching[index])
                    continue;
                  const auto &trial = trial_residuals[index];
                  const auto &surface = *active_surfaces[index];
                  const amrex::Real trial_ratio = relaxation_residual_ratio(
                      trial.expansion_l2, trial.expansion_linf,
                      surface.mass_scale,
                      surface.tolerances.l2_times_mass,
                      surface.tolerances.linf_times_mass);
                  if (trial.relaxation_state ==
                          SurfaceRelaxationState::active &&
                      trial.invalid_points == 0 &&
                      std::isfinite(trial_ratio) &&
                      trial_ratio < best_ratio[index]) {
                    best_ratio[index] = trial_ratio;
                    best_l2[index] = trial.expansion_l2;
                    best_linf[index] = trial.expansion_linf;
                    best_overstep[index] = overstep;
                  } else {
                    searching[index] = false;
                  }
                }
              }

              for (std::size_t index = 0; index < active_layers.size();
                   ++index) {
                const amrex::Real relative_improvement =
                    (baseline_ratio[index] - best_ratio[index]) /
                    baseline_ratio[index];
                if (!(relative_improvement >=
                      minimum_relaxation_extrapolation_improvement)) {
                  best_overstep[index] = 1;
                  best_ratio[index] = baseline_ratio[index];
                  best_l2[index] = baseline_l2[index];
                  best_linf[index] = baseline_linf[index];
                }
                controls[index] = {active_layers[index].stable_id,
                                   active_layers[index].angular_level,
                                   best_overstep[index],
                                   active_modes[index]};
              }
              evaluate_controls(true);
              for (std::size_t index = 0; index < active_layers.size();
                   ++index)
                if (best_overstep[index] > 1 && verbose) {
                  const auto &surface = *active_surfaces[index];
                  CCTK_VINFO(
                      "Accepted relaxation extrapolation: surface_id=%lld "
                      "angular_level=%d mode=%s factor=%.6g "
                      "ratio %.6e -> %.6e "
                      "M*L2 %.6e -> %.6e M*Linf %.6e -> %.6e",
                      static_cast<long long>(
                          active_layers[index].stable_id),
                      active_layers[index].angular_level + 1,
                      active_modes[index] ==
                              RelaxationExtrapolationMode::mean_height
                          ? "mean-height"
                          : "full-shape",
                      double(best_overstep[index]),
                      double(baseline_ratio[index]),
                      double(best_ratio[index]),
                      double(surface.mass_scale * baseline_l2[index]),
                      double(surface.mass_scale * best_l2[index]),
                      double(surface.mass_scale * baseline_linf[index]),
                      double(surface.mass_scale * best_linf[index]));
                }
            };
        if (relaxation_report_every > 0)
          report_relaxation_trials(state, completed_steps,
                                   completed_pseudo_time, "initial");

        for (int step = 0; step < maximum_relaxation_steps &&
                           residual.active > 0;
             ++step) {
          amrex::Real pseudo_dt =
              state.logical_surfaces->prepare_relaxation_time_steps(
                  relaxation_cfl_factor);
          amrex::ParallelDescriptor::ReduceRealMin(pseudo_dt);
          if (!(pseudo_dt > 0) || !amrex::Math::isfinite(pseudo_dt)) {
            residual.failed += residual.active;
            residual.active = 0;
            break;
          }

          state.logical_surfaces->begin_relaxation_step();
          for (int stage = 0; stage < num_relaxation_stages; ++stage) {
            state.logical_surfaces->advance_relaxation_stage(
                integrator, stage, eta_damping_times_mass,
                angular_dissipation_strength, minimum_radius_factor,
                maximum_radius_factor);
            if (stage + 1 < num_relaxation_stages) {
              state.logical_surfaces->evaluate_angular_derivatives();
              state.logical_surfaces->evaluate_expansion();
            }
          }
          amrex::Long invalid_relaxation =
              state.logical_surfaces->reject_invalid_relaxation_steps();
          amrex::ParallelDescriptor::ReduceLongSum(invalid_relaxation);
          if (invalid_relaxation != 0 && verbose)
            CCTK_VINFO("Rejected %lld invalid relaxation point update(s); "
                       "only affected surfaces will be rolled back",
                       static_cast<long long>(invalid_relaxation));

          state.logical_surfaces->rebuild_positions();
          state.logical_surfaces->evaluate_angular_derivatives();
          state.logical_surfaces->evaluate_expansion();
          route_updates();
          for (auto &container : state.containers) {
            // Rejecting one point atomically restores its entire surface to
            // the committed shape. That rollback is not bounded by the
            // ordinary per-step cell-displacement contract.
            if (invalid_relaxation != 0)
              container->redistribute_global();
            else
              container->redistribute_local_hierarchy(max_num_cells_moved);
          }
          sample_adm();
          route_and_evaluate();
          completed_steps = step + 1;
          completed_pseudo_time += pseudo_dt;
          residual = convergence(completed_steps >=
                                 minimum_relaxation_steps);
          if (use_relaxation_extrapolation && residual.active > 0) {
            std::vector<SurfaceLayerSelection> reference_layers;
            std::vector<SurfaceLayerSelection> extrapolation_layers;
            for (auto &schedule : extrapolation_schedule) {
              const amrex::Real event_time =
                  schedule.next_event *
                  relaxation_extrapolation_interval_times_mass *
                  schedule.mass_scale;
              if (completed_pseudo_time < event_time)
                continue;
              if (schedule.next_event <= 2 ||
                  schedule.next_event % 2 == 0) {
                reference_layers.push_back(schedule.selection);
                schedule.reference_valid = true;
              } else if (schedule.reference_valid) {
                extrapolation_layers.push_back(schedule.selection);
                schedule.reference_valid = false;
              }
              ++schedule.next_event;
            }
            state.logical_surfaces->store_extrapolation_reference(
                reference_layers);
            if (!extrapolation_layers.empty()) {
              test_relaxation_extrapolation(extrapolation_layers);
              residual = convergence(completed_steps >=
                                     minimum_relaxation_steps);
            }
          }
          if (relaxation_report_every > 0 &&
              completed_steps % relaxation_report_every == 0)
            report_relaxation_trials(state, completed_steps,
                                     completed_pseudo_time, "progress");
        }

        const amrex::Long failed_layers = residual.failed + residual.active;
        if (failed_layers != 0)
          report_relaxation_trials(state, completed_steps,
                                   completed_pseudo_time,
                                   "final trial before rollback");
        if (residual.refine != 0) {
          if (verbose)
            CCTK_VINFO("Angular resolution gate promoted %lld "
                       "under-resolved layer(s): maximum cutoff indicator "
                       "%.6e exceeds threshold %.6e",
                       static_cast<long long>(residual.refine),
                       double(residual.maximum_cutoff_indicator),
                       double(angular_continuation_cutoff_threshold));
          report_relaxation_trials(state, completed_steps,
                                   completed_pseudo_time,
                                   "under-resolved; refining");
        }
        const auto outcomes =
            collect_layer_outcomes(state, pending_layers);
        state.logical_surfaces->finish_relaxation_search();
        std::vector<SurfaceLayerSelection> promoted_layers;
        std::vector<SurfaceLayerSelection> next_layers;
        std::vector<SurfaceLayerSelection> terminal_layers;
        promoted_layers.reserve(outcomes.size());
        next_layers.reserve(outcomes.size());
        terminal_layers.reserve(outcomes.size());

        amrex::Long active_points = 0;
        for (const auto &outcome : outcomes) {
          const auto found = std::lower_bound(
              state.surfaces.begin(), state.surfaces.end(),
              outcome.selection.stable_id,
              [](const SurfaceRecord &surface, const amrex::Long stable_id) {
                return surface.stable_id < stable_id;
              });
          if (found == state.surfaces.end() ||
              found->stable_id != outcome.selection.stable_id ||
              outcome.selection.angular_level < 0 ||
              outcome.selection.angular_level >= found->angular_levels)
            CCTK_ERROR("ParticleAHFinderX angular continuation found invalid "
                       "surface metadata");
          auto &surface = *found;
          auto &layer =
              surface.angular_layer[outcome.selection.angular_level];
          layer.relaxation_state = outcome.relaxation_state;
          layer.last_relaxation_steps = completed_steps;
          layer.expansion_l2 = outcome.expansion_l2;
          layer.expansion_linf = outcome.expansion_linf;
          if (active_points > std::numeric_limits<amrex::Long>::max() -
                                  layer.particle_count)
            CCTK_ERROR("ParticleAHFinderX active angular-layer point count "
                       "overflowed");
          active_points += layer.particle_count;
          const auto due = std::lower_bound(due_surface_ids.begin(),
                                            due_surface_ids.end(),
                                            surface.stable_id);
          if (due == due_surface_ids.end() || *due != surface.stable_id)
            CCTK_ERROR("ParticleAHFinderX angular continuation found a "
                       "surface outside the due set");
          const std::size_t due_index =
              static_cast<std::size_t>(due - due_surface_ids.begin());
          if (surface_steps[due_index] >
              std::numeric_limits<int>::max() - completed_steps)
            CCTK_ERROR("ParticleAHFinderX relaxation step counter overflowed");
          surface_steps[due_index] += completed_steps;

          const amrex::Real l2_times_mass =
              surface.mass_scale * outcome.expansion_l2;
          const amrex::Real linf_times_mass =
              surface.mass_scale * outcome.expansion_linf;
          const amrex::Real ratio = amrex::max(
              l2_times_mass / surface.tolerances.l2_times_mass,
              linf_times_mass / surface.tolerances.linf_times_mass);
          maximum_terminal_l2_times_mass =
              amrex::max(maximum_terminal_l2_times_mass, l2_times_mass);
          maximum_terminal_linf_times_mass =
              amrex::max(maximum_terminal_linf_times_mass, linf_times_mass);
          maximum_terminal_ratio =
              amrex::max(maximum_terminal_ratio, ratio);

          if (outcome.relaxation_state ==
              SurfaceRelaxationState::failed) {
            surface.relaxation_state = SurfaceRelaxationState::failed;
            failed_surface_ids.push_back(surface.stable_id);
            terminal_layers.push_back(outcome.selection);
          } else if (outcome.selection.angular_level + 1 <
                     surface.angular_levels) {
            promoted_layers.push_back(outcome.selection);
            next_layers.push_back(
                {surface.stable_id, outcome.selection.angular_level + 1});
            surface.active_angular_level =
                outcome.selection.angular_level + 1;
          } else {
            if (outcome.relaxation_state ==
                SurfaceRelaxationState::refine)
              CCTK_ERROR("ParticleAHFinderX target angular layer requested "
                         "an impossible further refinement");
            surface.relaxation_state = SurfaceRelaxationState::converged;
            converged_surface_ids.push_back(surface.stable_id);
            terminal_layers.push_back(outcome.selection);
          }
        }
        if (completed_steps >
            std::numeric_limits<int>::max() - total_completed_steps)
          CCTK_ERROR("ParticleAHFinderX total relaxation step counter "
                     "overflowed");
        total_completed_steps += completed_steps;
        if (completed_steps != 0 &&
            active_points >
                std::numeric_limits<amrex::Long>::max() / completed_steps)
          CCTK_ERROR("ParticleAHFinderX relaxation point-step counter "
                     "overflowed");
        const amrex::Long point_steps =
            active_points * static_cast<amrex::Long>(completed_steps);
        if (processed_point_steps >
            std::numeric_limits<amrex::Long>::max() - point_steps)
          CCTK_ERROR("ParticleAHFinderX relaxation point-step counter "
                     "overflowed");
        processed_point_steps += point_steps;

        // A successful source layer initializes its already-persistent next
        // layer on the owner GPU. Failed layers atomically return to their
        // committed state in finish_relaxation_search().
        state.logical_surfaces->prolong_surface_layers(promoted_layers);
        if (!next_layers.empty()) {
          std::vector<SurfaceLayerSelection> refresh_layers = terminal_layers;
          refresh_layers.insert(refresh_layers.end(), next_layers.begin(),
                                next_layers.end());
          std::sort(refresh_layers.begin(), refresh_layers.end(),
                    [](const SurfaceLayerSelection &first,
                       const SurfaceLayerSelection &second) {
                      return first.stable_id < second.stable_id ||
                             (first.stable_id == second.stable_id &&
                              first.angular_level < second.angular_level);
                    });
          state.logical_surfaces->set_due_surface_layers(refresh_layers);
          for (auto &container : state.containers)
            container->set_active_surface_layers(refresh_layers);
          active_particle_count =
              selected_particle_count(state, refresh_layers);
        }
        state.logical_surfaces->rebuild_positions();
        // Redistribution during the coarse solve reorders every persistent
        // AMReX particle, including inactive fine layers. Refresh only their
        // reverse-routing addresses before sending the prolonged state; do
        // not replace that state with the inactive particles' old values.
        refresh_source_metadata();
        route_updates();
        // Rollback and inter-layer prolongation can move points farther than
        // the bounded per-step allowance. This is a hierarchy-synchronized
        // analysis point, so reassign every persistent particle globally.
        for (auto &container : state.containers)
          container->redistribute_global();
        sample_adm();
        route_and_evaluate();
        route_updates();
        pending_layers = std::move(next_layers);
      }

      std::sort(converged_surface_ids.begin(), converged_surface_ids.end());
      std::sort(failed_surface_ids.begin(), failed_surface_ids.end());
      std::sort(searched_target_layers.begin(), searched_target_layers.end(),
                [](const SurfaceLayerSelection &first,
                   const SurfaceLayerSelection &second) {
                  return first.stable_id < second.stable_id ||
                         (first.stable_id == second.stable_id &&
                          first.angular_level < second.angular_level);
                });
      searched_target_layers.erase(
          std::unique(searched_target_layers.begin(),
                      searched_target_layers.end(),
                      [](const SurfaceLayerSelection &first,
                         const SurfaceLayerSelection &second) {
                        return first.stable_id == second.stable_id &&
                               first.angular_level == second.angular_level;
                      }),
          searched_target_layers.end());

      // A heterogeneous batch can finish one target layer while another
      // surface is still progressing through coarser layers. Refresh all
      // target layers that were actually searched once, then form their
      // bounded physical diagnostics without exposing an internal layer.
      if (!searched_target_layers.empty()) {
        state.logical_surfaces->set_due_surface_layers(
            searched_target_layers);
        for (auto &container : state.containers)
          container->set_active_surface_layers(searched_target_layers);
        active_particle_count =
            selected_particle_count(state, searched_target_layers);
        sample_adm();
        route_and_evaluate();
        route_updates();
        update_surface_diagnostics(
            *state.logical_surfaces, state.surfaces, cctk_iteration,
            static_cast<amrex::Real>(cctk_time), total_completed_steps);
      }

      for (std::size_t index = 0; index < due_surface_ids.size(); ++index) {
        auto found = std::lower_bound(
            state.surfaces.begin(), state.surfaces.end(),
            due_surface_ids[index],
            [](const SurfaceRecord &surface, const amrex::Long stable_id) {
              return surface.stable_id < stable_id;
            });
        if (found == state.surfaces.end() ||
            found->stable_id != due_surface_ids[index])
          CCTK_ERROR("ParticleAHFinderX lost a due surface after angular "
                     "continuation");
        found->last_search_iteration = cctk_iteration;
        found->last_search_time = static_cast<amrex::Real>(cctk_time);
        found->last_search_steps = surface_steps[index];
        const bool converged = std::binary_search(
            converged_surface_ids.begin(), converged_surface_ids.end(),
            found->stable_id);
        found->relaxation_state =
            converged ? SurfaceRelaxationState::converged
                      : SurfaceRelaxationState::failed;
        if (converged)
          found->last_success_iteration = cctk_iteration;
      }

      const amrex::Long converged_surfaces =
          static_cast<amrex::Long>(converged_surface_ids.size());
      const amrex::Long failed_surfaces =
          static_cast<amrex::Long>(failed_surface_ids.size());
      if (converged_surfaces + failed_surfaces !=
          static_cast<amrex::Long>(due_surface_ids.size()))
        CCTK_ERROR("ParticleAHFinderX angular continuation did not produce "
                   "one terminal outcome per due physical surface");
      state.successful_searches += converged_surfaces;
      state.failed_searches += failed_surfaces;
      if (verbose)
        CCTK_VINFO(
            "GPU hyperbolic relaxation (%s) completed %d sequential "
            "level-step(s), point-steps=%lld, extrapolation trials=%d "
            "point-evaluations=%lld: converged=%lld failed=%lld "
            "max terminal (M*L2)=%.3g max terminal (M*Linf)=%.3g",
            relaxation_integrator, total_completed_steps,
            static_cast<long long>(processed_point_steps),
            total_extrapolation_trials,
            static_cast<long long>(extrapolation_point_evaluations),
            static_cast<long long>(converged_surfaces),
            static_cast<long long>(failed_surfaces),
            double(maximum_terminal_l2_times_mass),
            double(maximum_terminal_linf_times_mass));
      if (failed_surfaces != 0)
        CCTK_VWARN(
            CCTK_WARN_ALERT,
            "ParticleAHFinderX restored %lld unconverged physical "
            "surface(s) after %d sequential angular-level step(s) "
            "(max terminal ratio %.3g)",
            static_cast<long long>(failed_surfaces), total_completed_steps,
            double(maximum_terminal_ratio));
      state.last_relaxation_steps = total_completed_steps;
      state.last_search_converged = failed_surfaces == 0;
    } else {
      route_updates();
    }
    if (solve_surfaces && print_timing_stats) {
      amrex::Gpu::streamSynchronize();
      const amrex::Real local_wall_time =
          amrex::second() - solver_wall_start;
      amrex::Real wall_times[3]{local_wall_time, local_wall_time,
                                local_wall_time};
      amrex::ParallelDescriptor::ReduceRealMin(wall_times[0]);
      amrex::ParallelDescriptor::ReduceRealSum(wall_times[1]);
      amrex::ParallelDescriptor::ReduceRealMax(wall_times[2]);
      wall_times[1] /= amrex::ParallelDescriptor::NProcs();
      if (amrex::ParallelDescriptor::IOProcessor())
        CCTK_VINFO(
            "Solver timing: surfaces=%zu points=%lld steps=%d rank wall "
            "min/avg/max=%.6g/%.6g/%.6g s",
            due_surface_ids.size(),
            static_cast<long long>(due_particle_count),
            state.last_relaxation_steps, double(wall_times[0]),
            double(wall_times[1]), double(wall_times[2]));
    }
    if (!solve_surfaces)
      update_surface_diagnostics(
          *state.logical_surfaces, state.surfaces, cctk_iteration,
          static_cast<amrex::Real>(cctk_time), 0);
    finalize_candidates_after_search(
        state, cctk_iteration, static_cast<amrex::Real>(cctk_time));
    update_tracking_history(cctkGH, state, cctk_iteration,
                            static_cast<amrex::Real>(cctk_time));
  } catch (const std::exception &error) {
    CCTK_VERROR("ParticleAHFinderX ADM gather failed: %s", error.what());
  }

  if (!state.reported_solver_pending && !solve_surfaces) {
    CCTK_VINFO("ParticleAHFinderX is reusing %lld persistent particles in %zu "
               "surface population(s); live ADM values and metric "
               "derivatives and expansion remain in device particle "
               "storage; solve_surfaces=no leaves relaxation disabled",
               static_cast<long long>(state.global_particle_count),
               state.surfaces.size());
    state.reported_solver_pending = true;
  }
}

extern "C" int ParticleAHFinderX_Shutdown() {
  runtime() = Runtime{};
  return 0;
}

} // namespace ParticleAHFinderX
