/**
 * \file particle_ah_finder.cxx
 * \brief Cactus scheduling and persistent surface lifecycle.
 *
 * Configured particles are born once, remain in their AMReX containers, and
 * are never recreated by analysis cadence. Surface relaxation is an internal
 * pseudo-time solve on a frozen physical slice and does not participate in
 * the physical evolution integrator's stage schedule.
 */
#include "analysis_api.hxx"
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

#ifdef AMREX_USE_MPI
#include <mpi.h>
#ifdef OPEN_MPI
#include <mpi-ext.h>
#endif
#endif

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

struct GpuAwareMpiCapability {
  bool query_available = false;
  bool supported = false;
};

GpuAwareMpiCapability query_gpu_aware_mpi_capability() {
  GpuAwareMpiCapability capability;
#if defined(AMREX_USE_GPU) && defined(AMREX_USE_MPI)
  int query_available = 0;
  int supported = 0;

#if defined(AMREX_USE_CUDA)
#if defined(MPICH_NUMVERSION) && MPICH_NUMVERSION >= 40000000
  query_available = 1;
  supported = MPIX_Query_cuda_support() != 0;
#elif defined(MPIX_CUDA_AWARE_SUPPORT)
  query_available = 1;
  supported = MPIX_Query_cuda_support() != 0;
#endif
#elif defined(AMREX_USE_HIP)
#if defined(MPICH_NUMVERSION) && MPICH_NUMVERSION >= 40000000
  query_available = 1;
  supported = MPIX_Query_hip_support() != 0;
#elif defined(MPIX_ROCM_AWARE_SUPPORT)
  query_available = 1;
  supported = MPIX_Query_rocm_support() != 0;
#endif
#elif defined(AMREX_USE_SYCL)
#if defined(MPICH_NUMVERSION) && MPICH_NUMVERSION >= 40000000
  query_available = 1;
  supported = MPIX_Query_ze_support() != 0;
#endif
#endif

  amrex::ParallelDescriptor::ReduceIntMin(query_available);
  if (query_available) {
    amrex::ParallelDescriptor::ReduceIntMin(supported);
    capability.query_available = true;
    capability.supported = supported != 0;
  }
#endif
  return capability;
}

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

class ActiveBandDeferred final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void accumulate_routing_stats(SurfaceRoutingStats &total,
                              const SurfaceRoutingStats &increment) {
  total.sent += increment.sent;
  total.received += increment.received;
  total.missing += increment.missing;
  total.duplicate += increment.duplicate;
  total.invalid += increment.invalid;
  total.outside_active_band += increment.outside_active_band;
  total.device_payload_bytes_sent += increment.device_payload_bytes_sent;
  total.device_payload_bytes_received +=
      increment.device_payload_bytes_received;
  total.device_local_copy_bytes += increment.device_local_copy_bytes;
  total.host_point_data_bytes += increment.host_point_data_bytes;
  total.control_device_to_host_bytes +=
      increment.control_device_to_host_bytes;
  total.control_host_to_device_bytes +=
      increment.control_host_to_device_bytes;
  total.mpi_control_bytes_sent += increment.mpi_control_bytes_sent;
  total.mpi_control_bytes_received += increment.mpi_control_bytes_received;
  total.device_to_host_control_copies +=
      increment.device_to_host_control_copies;
  total.host_to_device_control_copies +=
      increment.host_to_device_control_copies;
  total.mpi_count_collectives += increment.mpi_count_collectives;
  total.mpi_payload_exchange_rounds +=
      increment.mpi_payload_exchange_rounds;
  total.mpi_payload_messages_sent +=
      increment.mpi_payload_messages_sent;
  total.mpi_payload_messages_received +=
      increment.mpi_payload_messages_received;
  total.explicit_device_synchronizations +=
      increment.explicit_device_synchronizations;
  total.forward_topology_reuses += increment.forward_topology_reuses;
  total.forward_topology_rebuilds += increment.forward_topology_rebuilds;
}

struct SolverTimingStats {
  amrex::Real adm_gather = 0;
  amrex::Real mesh_to_logical = 0;
  amrex::Real source_metadata = 0;
  amrex::Real logical_to_mesh = 0;
  amrex::Real redistribution = 0;
  amrex::Real convergence = 0;
  amrex::Real pseudo_steps = 0;
  amrex::Real extrapolation = 0;
  amrex::Long adm_gather_calls = 0;
  amrex::Long redistribution_calls = 0;
  amrex::Long convergence_calls = 0;
  amrex::Long pseudo_steps_timed = 0;
  amrex::Long extrapolation_calls = 0;
  amrex::Long timestep_min_collectives = 0;
  amrex::Long pseudo_time_min_collectives = 0;
  amrex::Long device_convergence_checks = 0;
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

std::vector<SurfaceLevelRange>
global_surface_level_ranges(Runtime &state,
                            const std::vector<SurfaceLevelRequest> &requests) {
  std::vector<amrex::Long> counts(requests.size(), 0);
  std::vector<int> minimum(requests.size(),
                           std::numeric_limits<int>::max());
  std::vector<int> maximum(requests.size(), -1);
  for (auto &container : state.containers) {
    const auto local = container->local_surface_level_ranges(requests);
    if (local.size() != requests.size())
      CCTK_ERROR("ParticleAHFinderX received an inconsistent local surface "
                 "level-range table");
    for (std::size_t index = 0; index < requests.size(); ++index) {
      if (local[index].stable_id != requests[index].stable_id)
        CCTK_ERROR("ParticleAHFinderX surface level-range IDs changed order");
      counts[index] += local[index].particle_count;
      if (local[index].particle_count > 0) {
        minimum[index] = std::min(minimum[index], local[index].minimum_level);
        maximum[index] = std::max(maximum[index], local[index].maximum_level);
      }
    }
  }
  if (!requests.empty()) {
    const int count = static_cast<int>(requests.size());
    amrex::ParallelDescriptor::ReduceLongSum(counts.data(), count);
    amrex::ParallelDescriptor::ReduceIntMin(minimum.data(), count);
    amrex::ParallelDescriptor::ReduceIntMax(maximum.data(), count);
  }

  std::vector<SurfaceLevelRange> result(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index)
    result[index] = {requests[index].stable_id, counts[index],
                     counts[index] > 0 ? minimum[index] : -1,
                     maximum[index]};
  return result;
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
       state.logical_surfaces->local_solver_surface_states()) {
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
       state.logical_surfaces->local_solver_surface_states()) {
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

std::vector<SurfaceLayerSelection> final_analysis_surface_layers(
    const Runtime &state,
    const std::vector<SurfaceLayerSelection> &searched_target_layers,
    const int iteration) {
  std::vector<SurfaceLayerSelection> result;
  result.reserve(searched_target_layers.size());
  for (const auto &selection : searched_target_layers) {
    const auto found = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), selection.stable_id,
        [](const SurfaceRecord &surface, const amrex::Long stable_id) {
          return surface.stable_id < stable_id;
        });
    if (found == state.surfaces.end() ||
        found->stable_id != selection.stable_id ||
        selection.angular_level + 1 != found->angular_levels ||
        found->last_search_iteration != iteration)
      CCTK_ERROR("ParticleAHFinderX final analysis route lost a searched "
                 "target layer");
    // Candidate finalization can retire or reset a failed provisional
    // population after its physical diagnostics were recorded. Keep only
    // surviving converged targets in the route consumed by AH_analysis.
    if (found->lifecycle != ParticleLifecycle::retired &&
        found->relaxation_state == SurfaceRelaxationState::converged)
      result.push_back(selection);
  }
  return result;
}

bool same_surface_layers(
    const std::vector<SurfaceLayerSelection> &first,
    const std::vector<SurfaceLayerSelection> &second) {
  return first.size() == second.size() &&
         std::equal(first.begin(), first.end(), second.begin(),
                    [](const SurfaceLayerSelection &left,
                       const SurfaceLayerSelection &right) {
                      return left.stable_id == right.stable_id &&
                             left.angular_level == right.angular_level;
                    });
}

std::vector<RadialCacheLayer> local_radial_cache_layers(
    const Runtime &state,
    const std::vector<SurfaceLayerSelection> &selections) {
  std::vector<RadialCacheLayer> result;
  amrex::Long cache_offset = 0;
  const int rank = amrex::ParallelDescriptor::MyProc();
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
      CCTK_ERROR("ParticleAHFinderX radial-cache selection lost its "
                 "surface metadata");
    if (found->logical_owner_rank != rank)
      continue;
    const auto &layer = found->angular_layer[selection.angular_level];
    result.push_back({found->stable_id, layer.logical_owner_offset,
                      layer.particle_count, cache_offset,
                      selection.angular_level});
    if (cache_offset > std::numeric_limits<amrex::Long>::max() -
                           layer.particle_count)
      CCTK_ERROR("ParticleAHFinderX radial-cache point count overflowed");
    cache_offset += layer.particle_count;
  }
  return result;
}

std::vector<RadialCacheLayer>
local_active_radial_cache_layers(Runtime &state) {
  std::vector<SurfaceLayerSelection> active_layers;
  for (const auto &surface :
       state.logical_surfaces->local_solver_surface_states())
    if (surface.due &&
        surface.relaxation_state == SurfaceRelaxationState::active)
      active_layers.push_back(
          {surface.stable_id, surface.angular_level});
  std::sort(active_layers.begin(), active_layers.end(),
            [](const SurfaceLayerSelection &first,
               const SurfaceLayerSelection &second) {
              return first.stable_id < second.stable_id ||
                     (first.stable_id == second.stable_id &&
                      first.angular_level < second.angular_level);
            });
  return local_radial_cache_layers(state, active_layers);
}

struct SurfaceLayerBatch {
  std::vector<SurfaceLayerSelection> selected;
  std::vector<SurfaceLayerSelection> deferred;
  bool use_radial_cache = false;
};

SurfaceLayerBatch radial_cache_batch(
    const Runtime &state,
    const std::vector<SurfaceLayerSelection> &pending_layers,
    const int radial_nodes, const int memory_limit_mib) {
  SurfaceLayerBatch result;
  result.use_radial_cache = true;
  const amrex::Long bytes_per_point =
      amrex::Long(30) * radial_nodes * amrex::Long(sizeof(amrex::Real)) +
      amrex::Long(2 * sizeof(amrex::Real) + sizeof(amrex::Long) +
                  sizeof(int));
  const amrex::Long memory_limit =
      amrex::Long(memory_limit_mib) * 1024 * 1024;
  std::vector<amrex::Long> owner_bytes(
      amrex::ParallelDescriptor::NProcs(), 0);
  SurfaceRole selected_role = SurfaceRole::other;
  bool have_selected_role = false;

  for (const auto &selection : pending_layers) {
    if (!result.use_radial_cache && !result.selected.empty()) {
      result.deferred.push_back(selection);
      continue;
    }
    const auto found = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), selection.stable_id,
        [](const SurfaceRecord &surface, const amrex::Long stable_id) {
          return surface.stable_id < stable_id;
        });
    if (found == state.surfaces.end() ||
        found->stable_id != selection.stable_id ||
        selection.angular_level < 0 ||
        selection.angular_level >= found->angular_levels ||
        found->logical_owner_rank < 0 ||
        found->logical_owner_rank >=
            amrex::ParallelDescriptor::NProcs())
      CCTK_ERROR("ParticleAHFinderX radial-cache batching found invalid "
                 "surface metadata");
    // Keep speculative common surfaces separate from established individual
    // surfaces. A common candidate can legitimately leave its radial envelope;
    // that must not force an otherwise well-behaved individual-surface batch
    // back onto the direct path.
    if (have_selected_role && found->role != selected_role) {
      result.deferred.push_back(selection);
      continue;
    }
    const amrex::Long points =
        found->angular_layer[selection.angular_level].particle_count;
    const amrex::Long bytes =
        points <= std::numeric_limits<amrex::Long>::max() / bytes_per_point
            ? points * bytes_per_point
            : std::numeric_limits<amrex::Long>::max();
    auto &load = owner_bytes[found->logical_owner_rank];
    if (bytes <= memory_limit && load <= memory_limit - bytes) {
      result.selected.push_back(selection);
      if (!have_selected_role) {
        selected_role = found->role;
        have_selected_role = true;
      }
      load += bytes;
    } else if (result.selected.empty() && bytes > memory_limit) {
      // A single oversized layer cannot be split without angular sharding.
      // Process it once through the direct path, then continue cache batching
      // the remaining independent surfaces.
      result.selected.push_back(selection);
      selected_role = found->role;
      have_selected_role = true;
      result.use_radial_cache = false;
    } else {
      result.deferred.push_back(selection);
    }
  }
  if (result.selected.empty())
    CCTK_ERROR("ParticleAHFinderX radial-cache batching made no progress");
  return result;
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
  if (use_radial_geometry_cache && radial_geometry_cache_nodes < 4)
    CCTK_PARAMWARN("ParticleAHFinderX radial geometry caching requires at "
                   "least four radial nodes");
  if (use_radial_geometry_cache &&
      (!(radial_geometry_cache_half_width_factor > 0) ||
       !(radial_geometry_cache_level_width_factor > 0) ||
       !(radial_geometry_cache_level_width_factor <= 1) ||
       (radial_geometry_cache_recenter_passes > 0 &&
        (!(radial_geometry_cache_recenter_width_factor > 0) ||
         !(radial_geometry_cache_recenter_width_factor <= 1))) ||
       !(radial_geometry_cache_validation_tolerance > 0) ||
       radial_geometry_cache_max_memory_mib <= 0))
    CCTK_PARAMWARN("ParticleAHFinderX radial geometry caching requires a "
                   "positive interval half-width, a continuation-level "
                   "width factor in (0,1], a recenter width factor in "
                   "(0,1] when recentering is enabled, a positive validation "
                   "tolerance, and a positive memory ceiling");
  if (use_radial_geometry_cache &&
      radial_geometry_cache_direct_step_reserve >=
          maximum_relaxation_steps)
    CCTK_PARAMWARN(
        "ParticleAHFinderX radial_geometry_cache_direct_step_reserve must "
        "be smaller than maximum_relaxation_steps");
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
  state.interpolation_order = interpolation_order;

  state.num_patches = CarpetX::ghext->num_patches();
  if (state.num_patches != 1)
    CCTK_VERROR("ParticleAHFinderX version one supports one Cartesian "
                "CarpetX patch; a batched device coordinate map is required "
                "for %d patches",
                state.num_patches);

#ifdef AMREX_USE_GPU
  const auto mpi_capability = query_gpu_aware_mpi_capability();
  if (require_gpu_aware_mpi && amrex::ParallelDescriptor::NProcs() > 1) {
    if (!amrex::ParallelDescriptor::UseGpuAwareMpi() &&
        mpi_capability.supported)
      amrex::ParallelDescriptor::use_gpu_aware_mpi = true;
    if (!amrex::ParallelDescriptor::UseGpuAwareMpi() &&
        mpi_capability.query_available)
      CCTK_ERROR("The MPI runtime reports that accelerator-aware "
                 "communication is unavailable");
    if (!amrex::ParallelDescriptor::UseGpuAwareMpi())
      CCTK_ERROR("ParticleAHFinderX requires AMReX direct-device MPI; "
                 "enable amrex.use_gpu_aware_mpi before AMReX "
                 "initialization");
  }
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
  if (!state.radial_geometry_cache)
    state.radial_geometry_cache = std::make_unique<RadialGeometryCache>();

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

  for (auto &container : state.containers) {
    container->rebuild_amr_masks();
    container->redistribute_global();
  }
  const auto owner_stats = assign_spatial_logical_owners(state);
  state.logical_surfaces->initialize(local_logical_seeds(state.surfaces));

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
    const double colocated_fraction =
        owner_stats.particle_count > 0
            ? double(owner_stats.colocated_particles) /
                  double(owner_stats.particle_count)
            : 1.0;
    CCTK_VINFO("Provisional spatial logical ownership colocates %lld/%lld "
               "persistent particle(s) (%.2f%%); maximum owner load=%lld; "
               "the first complete hierarchy will finalize placement",
               static_cast<long long>(owner_stats.colocated_particles),
               static_cast<long long>(owner_stats.particle_count),
               100.0 * colocated_fraction,
               static_cast<long long>(owner_stats.maximum_owner_load));
    CCTK_VINFO("Created %zu persistent surface population(s) containing %lld "
               "AMReX particles; ordinary searches will reuse them in place",
               state.surfaces.size(),
               static_cast<long long>(state.global_particle_count));
    CCTK_VINFO("Logical owner rank %d holds %d angular layer(s) and %lld "
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
  if (state.radial_geometry_cache)
    state.radial_geometry_cache->invalidate();
  if (state.regrid_events == std::numeric_limits<amrex::Long>::max())
    CCTK_ERROR("ParticleAHFinderX regrid event counter overflowed");
  ++state.regrid_events;
  state.last_regrid_iteration = cctk_iteration;
}

void run_analysis(const cGH *const cctkGH, const int forced_surface_count) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  const bool forced = forced_surface_count >= 0;
  if (!state.setup)
    CCTK_ERROR("ParticleAHFinderX analysis ran before setup");
  ActiveLevelBand band;
  try {
    band = active_level_band();
  } catch (const std::exception &error) {
    CCTK_VERROR("ParticleAHFinderX cannot establish the active level band: %s",
                error.what());
  }
  if (!state.initial_spatial_ownership_finalized) {
    if (!band.complete_hierarchy)
      return;
    const bool already_searched = std::any_of(
        state.surfaces.begin(), state.surfaces.end(),
        [](const SurfaceRecord &surface) {
          return surface.last_search_iteration >= 0;
        });
    if (already_searched)
      CCTK_ERROR("ParticleAHFinderX cannot finalize initial spatial logical "
                 "ownership after a surface search");
    const auto owner_stats = assign_spatial_logical_owners(state);
    state.logical_surfaces = std::make_unique<LogicalSurfaceStorage>();
    state.logical_surfaces->initialize(local_logical_seeds(state.surfaces));
    state.radial_geometry_cache->invalidate();
    state.initial_spatial_ownership_finalized = true;
    if (verbose && amrex::ParallelDescriptor::IOProcessor()) {
      const double colocated_fraction =
          owner_stats.particle_count > 0
              ? double(owner_stats.colocated_particles) /
                    double(owner_stats.particle_count)
              : 1.0;
      CCTK_VINFO("Final spatial logical ownership colocates %lld/%lld "
                 "persistent particle(s) (%.2f%%); maximum owner load=%lld",
                 static_cast<long long>(owner_stats.colocated_particles),
                 static_cast<long long>(owner_stats.particle_count),
                 100.0 * colocated_fraction,
                 static_cast<long long>(owner_stats.maximum_owner_load));
    }
  }
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
  // Candidate birth, retirement, radius resets, and owner compaction perform
  // hierarchy-wide particle operations. Keep those lifecycle transitions at a
  // complete synchronized hierarchy even though established surfaces may be
  // solved on a smaller active band.
  if (band.complete_hierarchy && !forced)
    manage_candidates_before_search(
        state, cctk_iteration, static_cast<amrex::Real>(cctk_time));

  std::vector<amrex::Long> requested_surface_ids;
  for (const auto &surface : state.surfaces) {
    const bool selected =
        forced ? surface.configured_slot >= 0 &&
                     surface.configured_slot < forced_surface_count
               : surface.search_every > 0 &&
                     cctk_iteration >= surface.next_search_iteration &&
                     cctk_iteration % surface.search_every == 0 &&
                     surface.last_search_iteration != cctk_iteration &&
                     (band.complete_hierarchy ||
                      surface.lifecycle != ParticleLifecycle::candidate);
    if (surface.lifecycle != ParticleLifecycle::retired && selected)
      requested_surface_ids.push_back(surface.stable_id);
  }
  if (requested_surface_ids.empty())
    return;

  std::vector<SurfaceLevelRequest> level_requests;
  std::vector<amrex::Long> requested_multilevel_tracking_ids;
  level_requests.reserve(requested_surface_ids.size());
  requested_multilevel_tracking_ids.reserve(requested_surface_ids.size());
  for (const auto stable_id : requested_surface_ids) {
    const auto found = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), stable_id,
        [](const SurfaceRecord &surface, const amrex::Long id) {
          return surface.stable_id < id;
        });
    if (found == state.surfaces.end() || found->stable_id != stable_id)
      CCTK_ERROR("ParticleAHFinderX level eligibility lost a surface record");
    const bool multilevel_tracking =
        use_multilevel_tracking && found->last_success_iteration >= 0 &&
        found->angular_levels > 1;
    const int selected_level =
        found->last_success_iteration >= 0 && !multilevel_tracking
            ? found->angular_levels - 1
            : -1;
    level_requests.push_back({stable_id, selected_level});
    if (multilevel_tracking)
      requested_multilevel_tracking_ids.push_back(stable_id);
  }
  const auto level_ranges =
      global_surface_level_ranges(state, level_requests);
  std::vector<amrex::Long> due_surface_ids;
  std::vector<amrex::Long> multilevel_tracking_ids;
  std::vector<SurfaceLevelRequest> target_fallback_requests;
  due_surface_ids.reserve(requested_surface_ids.size());
  multilevel_tracking_ids.reserve(requested_surface_ids.size());
  target_fallback_requests.reserve(requested_surface_ids.size());
  for (std::size_t index = 0; index < requested_surface_ids.size(); ++index) {
    const auto found = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(),
        requested_surface_ids[index],
        [](const SurfaceRecord &surface, const amrex::Long stable_id) {
          return surface.stable_id < stable_id;
        });
    if (found == state.surfaces.end() ||
        found->stable_id != requested_surface_ids[index])
      CCTK_ERROR("ParticleAHFinderX level eligibility lost a surface record");
    const int selected_level = level_requests[index].angular_level;
    const amrex::Long expected =
        selected_level >= 0
            ? found->angular_layer[selected_level].particle_count
            : total_angular_particle_count(*found);
    const auto &range = level_ranges[index];
    if (range.particle_count != expected || range.minimum_level < 0 ||
        range.maximum_level < range.minimum_level)
      CCTK_VERROR(
          "ParticleAHFinderX surface %lld has invalid persistent AMR "
          "ownership metadata: count=%lld expected=%lld levels=[%d,%d]",
          static_cast<long long>(found->stable_id),
          static_cast<long long>(range.particle_count),
          static_cast<long long>(expected), range.minimum_level,
          range.maximum_level);
    if (band.contains(range.minimum_level) &&
        band.contains(range.maximum_level)) {
      due_surface_ids.push_back(found->stable_id);
      if (std::binary_search(requested_multilevel_tracking_ids.begin(),
                             requested_multilevel_tracking_ids.end(),
                             found->stable_id))
        multilevel_tracking_ids.push_back(found->stable_id);
    } else if (std::binary_search(requested_multilevel_tracking_ids.begin(),
                                  requested_multilevel_tracking_ids.end(),
                                  found->stable_id)) {
      // Coarse helper layers can straddle an asynchronous level even when the
      // accepted target layer remains wholly inside the active band. Preserve
      // the requested tracking cadence by falling back to a target-only solve;
      // an initial acquisition still requires its complete continuation tree.
      target_fallback_requests.push_back(
          {found->stable_id, found->angular_levels - 1});
    }
  }
  amrex::Long target_fallback_count = 0;
  if (!target_fallback_requests.empty()) {
    const auto fallback_ranges =
        global_surface_level_ranges(state, target_fallback_requests);
    for (std::size_t index = 0; index < target_fallback_requests.size();
         ++index) {
      const auto found = std::lower_bound(
          state.surfaces.begin(), state.surfaces.end(),
          target_fallback_requests[index].stable_id,
          [](const SurfaceRecord &surface, const amrex::Long stable_id) {
            return surface.stable_id < stable_id;
          });
      if (found == state.surfaces.end() ||
          found->stable_id != target_fallback_requests[index].stable_id)
        CCTK_ERROR("ParticleAHFinderX target-only fallback lost a surface "
                   "record");
      const auto &range = fallback_ranges[index];
      const auto expected =
          found->angular_layer[found->angular_levels - 1].particle_count;
      if (range.particle_count != expected || range.minimum_level < 0 ||
          range.maximum_level < range.minimum_level)
        CCTK_VERROR(
            "ParticleAHFinderX surface %lld has invalid target-layer AMR "
            "ownership metadata: count=%lld expected=%lld levels=[%d,%d]",
            static_cast<long long>(found->stable_id),
            static_cast<long long>(range.particle_count),
            static_cast<long long>(expected), range.minimum_level,
            range.maximum_level);
      if (band.contains(range.minimum_level) &&
          band.contains(range.maximum_level)) {
        due_surface_ids.push_back(found->stable_id);
        ++target_fallback_count;
      }
    }
    std::sort(due_surface_ids.begin(), due_surface_ids.end());
  }
  if (verbose && (!band.complete_hierarchy ||
                  due_surface_ids.size() != requested_surface_ids.size()))
    CCTK_VINFO(
        "Active-level-band eligibility at iteration %d time %.17g: "
        "band=[%d,%d) level_iteration=%lld/%lld requested=%zu eligible=%zu "
        "deferred=%zu "
        "target-only-fallback=%lld complete_hierarchy=%s",
        cctk_iteration, double(cctk_time), band.minimum_level,
        band.maximum_level,
        static_cast<long long>(band.logical_iteration_numerator),
        static_cast<long long>(band.logical_iteration_denominator),
        requested_surface_ids.size(),
        due_surface_ids.size(),
        requested_surface_ids.size() - due_surface_ids.size(),
        static_cast<long long>(target_fallback_count),
        band.complete_hierarchy ? "yes" : "no");
  if (due_surface_ids.empty())
    return;
  predict_surface_centers(
      cctkGH, state, due_surface_ids, cctk_iteration,
      static_cast<amrex::Real>(cctk_time),
      maximum_center_extrapolation_intervals, band.minimum_level,
      band.maximum_level, band.complete_hierarchy);
  if (due_surface_ids.empty())
    return;
  std::vector<SurfaceLayerSelection> due_surface_layers;
  std::vector<SurfaceLayerSelection> bootstrap_surface_layers;
  due_surface_layers.reserve(due_surface_ids.size());
  for (auto &surface : state.surfaces) {
    if (!std::binary_search(due_surface_ids.begin(), due_surface_ids.end(),
                            surface.stable_id))
      continue;
    const bool multilevel_tracking = std::binary_search(
        multilevel_tracking_ids.begin(), multilevel_tracking_ids.end(),
        surface.stable_id);
    surface.active_angular_level =
        surface.last_success_iteration >= 0 && !multilevel_tracking
            ? surface.angular_levels - 1
            : 0;
    due_surface_layers.push_back(
        {surface.stable_id, surface.active_angular_level});
    if (surface.last_success_iteration < 0 || multilevel_tracking)
      for (int level = 0; level < surface.angular_levels; ++level)
        bootstrap_surface_layers.push_back({surface.stable_id, level});
    else
      bootstrap_surface_layers.push_back(
          {surface.stable_id, surface.active_angular_level});
  }
  state.logical_surfaces->set_due_surface_layers(bootstrap_surface_layers);
  state.logical_surfaces->set_relaxed_helper_surface_ids(
      multilevel_tracking_ids);
  for (auto &container : state.containers)
    container->set_active_surface_layers(bootstrap_surface_layers);
  amrex::Long active_particle_count =
      selected_particle_count(state, bootstrap_surface_layers);
  auto &return_route = state.analysis_route_plan;
  return_route.valid = false;
  state.analysis_active_point_count = 0;
  SurfaceRoutingStats routing_performance;
  SolverTimingStats solver_timing;
  const bool measure_solver = solve_surfaces && print_timing_stats;
  amrex::Long mesh_to_logical_calls = 0;
  amrex::Long source_metadata_calls = 0;
  amrex::Long logical_to_mesh_calls = 0;
  for (auto &container : state.containers)
    container->begin_sampled_level_tracking();

  const auto redistribute_bounded = [&]() {
    const amrex::Real start = measure_solver ? amrex::second() : 0;
    return_route.valid = false;
    for (auto &container : state.containers) {
      if (band.complete_hierarchy)
        container->redistribute_local_hierarchy(max_num_cells_moved);
      else
        container->redistribute_active_band(
            band.minimum_level, band.maximum_level, max_num_cells_moved);
    }
    if (measure_solver) {
      solver_timing.redistribution += amrex::second() - start;
      ++solver_timing.redistribution_calls;
    }
  };
  const auto redistribute_unbounded = [&]() {
    const amrex::Real start = measure_solver ? amrex::second() : 0;
    return_route.valid = false;
    for (auto &container : state.containers) {
      if (band.complete_hierarchy)
        container->redistribute_global();
      else
        container->redistribute_active_band(band.minimum_level,
                                            band.maximum_level, 0);
    }
    if (measure_solver) {
      solver_timing.redistribution += amrex::second() - start;
      ++solver_timing.redistribution_calls;
    }
  };
  try {
    ensure_mesh_fields_ready(
        cctkGH,
        std::vector<MeshFieldRef>(state.adm_fields.begin(),
                                  state.adm_fields.end()));
    if (!CarpetX::active_levels)
      throw std::runtime_error(
          "ParticleAHFinderX analysis has no active CarpetX levels");

    const auto gather = [&](const AdmGatherPass pass) {
      const amrex::Real start = measure_solver ? amrex::second() : 0;
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
      if (measure_solver) {
        solver_timing.adm_gather += amrex::second() - start;
        ++solver_timing.adm_gather_calls;
      }
      return global_stats;
    };

    const auto sample_adm = [&]() {
      auto global_stats = gather(AdmGatherPass::initial);
      if (global_stats[2] > 0) {
        if (verbose)
          CCTK_VINFO("Retrying %lld ADM stencil miss(es) after bounded "
                     "AMReX redistribution",
                     static_cast<long long>(global_stats[2]));
        redistribute_bounded();
        global_stats = gather(AdmGatherPass::local_retry);
      }
      if (global_stats[2] > 0 || global_stats[3] > 0) {
        if (verbose)
          CCTK_VINFO("Retrying %lld ADM ownership/stencil miss(es) after "
                     "%s redistribution",
                     static_cast<long long>(global_stats[2] +
                                            global_stats[3]),
                     band.complete_hierarchy ? "hierarchy"
                                             : "active-band hierarchy");
        // Every proposed update was preflighted against this synchronized
        // band before it entered the particle container. A coarse/fine owner
        // change inside the band is therefore safe; a point requiring an
        // inactive level is rejected earlier by route_updates().
        redistribute_unbounded();
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

    const auto route_samples = [&]() {
      const amrex::Real routing_start = measure_solver ? amrex::second() : 0;
      const auto routing = route_particles_to_logical_owners(
          state.containers, *state.logical_surfaces,
          LogicalRoutingMode::sampled_fields, return_route,
          active_particle_count);
      accumulate_routing_stats(routing_performance, routing);
      ++mesh_to_logical_calls;
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
      if (measure_solver)
        solver_timing.mesh_to_logical += amrex::second() - routing_start;
      state.last_logical_route_iteration = cctk_iteration;
    };

    const auto evaluate_surface = [&]() {
      state.logical_surfaces->evaluate_surface_geometry();
    };

    const auto refresh_source_metadata = [&]() {
      const amrex::Real routing_start = measure_solver ? amrex::second() : 0;
      const auto routing = route_particles_to_logical_owners(
          state.containers, *state.logical_surfaces,
          LogicalRoutingMode::source_metadata, return_route,
          active_particle_count);
      accumulate_routing_stats(routing_performance, routing);
      ++source_metadata_calls;
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
      if (measure_solver)
        solver_timing.source_metadata += amrex::second() - routing_start;
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
    route_samples();
    evaluate_surface();

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

    const auto route_updates = [&](const bool defer_outside_band = true) {
      const amrex::Real routing_start = measure_solver ? amrex::second() : 0;
      const auto routing = route_updates_to_mesh_owners(
          *state.logical_surfaces, state.containers, return_route,
          band.complete_hierarchy ? -1 : band.minimum_level,
          band.complete_hierarchy ? -1 : band.maximum_level);
      accumulate_routing_stats(routing_performance, routing);
      ++logical_to_mesh_calls;
      amrex::Long global_routing[6]{
          routing.sent,    routing.received, routing.missing,
          routing.duplicate, routing.invalid, routing.outside_active_band};
      amrex::ParallelDescriptor::ReduceLongSum(global_routing, 6);
      if (measure_solver)
        solver_timing.logical_to_mesh += amrex::second() - routing_start;
      if (global_routing[5] != 0) {
        if (defer_outside_band)
          throw ActiveBandDeferred(
              "a proposed particle update left the synchronized active "
              "level band");
        return false;
      }
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
      return true;
    };

    if (solve_surfaces && !multilevel_tracking_ids.empty()) {
      // The first route refreshed every persistent layer from its AMReX
      // particle. Restrict the accepted target shape into coarse layers, then
      // return those low-pass states before beginning the correction solve.
      state.logical_surfaces->restrict_target_to_coarse_layers(
          multilevel_tracking_ids);
      state.logical_surfaces->rebuild_positions();
      route_updates();
      redistribute_unbounded();
      sample_adm();
      route_samples();
      evaluate_surface();
    }

    const auto convergence = [&](const bool accept_convergence,
                                 const bool fail_invalid_geometry = true) {
      const amrex::Real start = measure_solver ? amrex::second() : 0;
      auto result = state.logical_surfaces->update_solver_convergence(
          accept_convergence, angular_continuation_cutoff_threshold,
          angular_continuation_helper_tolerance_factor,
          fail_invalid_geometry);
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
      if (measure_solver) {
        solver_timing.convergence += amrex::second() - start;
        ++solver_timing.convergence_calls;
      }
      return result;
    };

    std::vector<SurfaceLayerSelection> searched_target_layers;
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
      std::vector<int> surface_steps(due_surface_ids.size(), 0);
      int total_completed_steps = 0;
      amrex::Long processed_point_steps = 0;
      amrex::Long cached_completed_steps = 0;
      amrex::Long direct_completed_steps = 0;
      amrex::Long cached_point_steps = 0;
      amrex::Long direct_point_steps = 0;
      int total_extrapolation_trials = 0;
      amrex::Long extrapolation_point_evaluations = 0;
      amrex::Real maximum_accepted_target_l2_times_mass = 0;
      amrex::Real maximum_accepted_target_linf_times_mass = 0;
      amrex::Real maximum_helper_l2_times_mass = 0;
      amrex::Real maximum_helper_linf_times_mass = 0;
      amrex::Real maximum_failed_terminal_ratio = 0;
      amrex::Long accepted_target_layers = 0;
      amrex::Long successful_helper_layers = 0;

      while (!pending_layers.empty()) {
        std::vector<SurfaceLayerSelection> deferred_batch_layers;
        bool cache_this_batch = false;
        if (use_radial_geometry_cache) {
          auto batch = radial_cache_batch(
              state, pending_layers, radial_geometry_cache_nodes,
              radial_geometry_cache_max_memory_mib);
          pending_layers = std::move(batch.selected);
          deferred_batch_layers = std::move(batch.deferred);
          cache_this_batch = batch.use_radial_cache;
        }
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
        return_route.valid = false;
        state.logical_surfaces->set_due_surface_layers(pending_layers);
        for (auto &container : state.containers)
          container->set_active_surface_layers(pending_layers);
        active_particle_count =
            selected_particle_count(state, pending_layers);
        state.logical_surfaces->begin_relaxation_search(
            eta_damping_times_mass);
        // Selecting the next continuation batch changes the active particle
        // population.  The sampled geometry for these persistent layers is
        // already current, but the cached logical-to-mesh transpose belongs
        // to the previous (possibly larger) selection.  Rebuild only source
        // metadata before the first accepted update of this batch.
        refresh_source_metadata();
        const auto initial_cache_layers =
            cache_this_batch
                ? local_radial_cache_layers(state, pending_layers)
                : std::vector<RadialCacheLayer>{};
        const auto bytes_per_point =
            amrex::Long(30) * radial_geometry_cache_nodes *
                amrex::Long(sizeof(amrex::Real)) +
            amrex::Long(2 * sizeof(amrex::Real) + sizeof(amrex::Long) +
                        sizeof(int));
        const amrex::Long cache_memory_limit =
            amrex::Long(radial_geometry_cache_max_memory_mib) * 1024 * 1024;
        std::vector<RadialCacheMeshLevel> cache_mesh_levels;
        if (cache_this_batch) {
          cache_mesh_levels.reserve(
              static_cast<std::size_t>(band.maximum_level -
                                       band.minimum_level));
          CarpetX::active_levels->loop_serially([&](auto &level) {
            const AdmFieldLayout fields =
                adm_field_layout(level, state.adm_fields);
            cache_mesh_levels.push_back(
                {level.level, fields.metric, fields.curv,
                 fields.metric_component, fields.curv_component,
                 fields.centering});
          });
        }
        const auto build_radial_cache =
            [&](const std::vector<RadialCacheLayer> &cache_layers,
                const amrex::Real half_width_factor,
                const char *const phase, const int recenter_pass) {
              if (!cache_this_batch)
                return false;
              amrex::Long local_cache_points = 0;
              for (const auto &layer : cache_layers)
                local_cache_points += layer.point_count;
              amrex::Long maximum_cache_points = local_cache_points;
              amrex::ParallelDescriptor::ReduceLongMax(maximum_cache_points);
              const amrex::Long maximum_cache_bytes =
                  maximum_cache_points <=
                          std::numeric_limits<amrex::Long>::max() /
                              bytes_per_point
                      ? maximum_cache_points * bytes_per_point
                      : std::numeric_limits<amrex::Long>::max();
              if (maximum_cache_bytes > cache_memory_limit) {
                if (verbose && amrex::ParallelDescriptor::IOProcessor())
                  CCTK_VINFO(
                      "Using direct frozen-slice sampling for a radial-cache "
                      "batch requiring %.3f MiB above the %d MiB owner limit",
                      double(maximum_cache_bytes) / (1024.0 * 1024.0),
                      radial_geometry_cache_max_memory_mib);
                return false;
              }
              const amrex::Real cache_start =
                  measure_solver ? amrex::second() : 0;
              state.radial_geometry_cache->prepare(
                  state.logical_surfaces->device_view(), cache_layers,
                  radial_geometry_cache_nodes, half_width_factor,
                  radial_geometry_cache_level_width_factor,
                  minimum_radius_factor, maximum_radius_factor);
              const auto cache_routing =
                  state.radial_geometry_cache->sample_batched(
                      state.logical_surfaces->device_view(),
                      *state.containers.front(), cache_mesh_levels,
                      band.minimum_level, band.maximum_level,
                      interpolation_order);
              accumulate_routing_stats(routing_performance, cache_routing);
              ++mesh_to_logical_calls;
              ++logical_to_mesh_calls;
              amrex::Long cache_route_counts[8]{
                  cache_routing.sent,
                  cache_routing.received,
                  cache_routing.missing,
                  cache_routing.duplicate,
                  cache_routing.invalid,
                  cache_routing.outside_active_band,
                  local_cache_points * radial_geometry_cache_nodes,
                  static_cast<amrex::Long>(cache_layers.size())};
              amrex::ParallelDescriptor::ReduceLongSum(cache_route_counts, 8);
              if (cache_route_counts[0] != cache_route_counts[6] ||
                  cache_route_counts[1] != cache_route_counts[6] ||
                  cache_route_counts[2] != 0 ||
                  cache_route_counts[3] != 0)
                CCTK_VERROR(
                    "ParticleAHFinderX batched radial-probe routing failed: "
                    "sent=%lld received=%lld missing=%lld duplicate=%lld "
                    "expected=%lld",
                    static_cast<long long>(cache_route_counts[0]),
                    static_cast<long long>(cache_route_counts[1]),
                    static_cast<long long>(cache_route_counts[2]),
                    static_cast<long long>(cache_route_counts[3]),
                    static_cast<long long>(cache_route_counts[6]));
              const bool cache_band_eligible =
                  cache_route_counts[4] == 0 && cache_route_counts[5] == 0;
              RadialCacheComparison comparison;
              if (cache_band_eligible)
                comparison = state.radial_geometry_cache->compare_direct(
                    state.logical_surfaces->device_view());
              else
                comparison.invalid = 1;
              amrex::Real cache_errors[2]{
                  comparison.maximum_absolute_error,
                  comparison.maximum_scaled_error};
              amrex::ParallelDescriptor::ReduceRealMax(cache_errors, 2);
              amrex::ParallelDescriptor::ReduceLongSum(comparison.invalid);
              const bool accepted =
                  comparison.invalid == 0 &&
                  cache_errors[1] <=
                      radial_geometry_cache_validation_tolerance;
              if (accepted) {
                state.radial_geometry_cache->interpolate(
                    state.logical_surfaces->device_view());
                evaluate_surface();
              } else {
                state.radial_geometry_cache->invalidate();
                evaluate_surface();
              }
              if (verbose && amrex::ParallelDescriptor::IOProcessor()) {
                CCTK_VINFO(
                    "Built owner-GPU radial geometry cache: layers=%lld "
                    "maximum-points=%lld nodes=%d payload=%.3f MiB "
                    "direct-check absolute/scaled=%.3e/%.3e invalid=%lld "
                    "active-band-compatible=%s accepted=%s wall=%.6g s "
                    "phase=%s recenter-pass=%d half-width-factor=%.6g "
                    "level-width-factor=%.6g batched-probes=%lld "
                    "route-calls=1/1",
                    static_cast<long long>(cache_route_counts[7]),
                    static_cast<long long>(maximum_cache_points),
                    radial_geometry_cache_nodes,
                    double(maximum_cache_bytes) / (1024.0 * 1024.0),
                    double(cache_errors[0]), double(cache_errors[1]),
                    static_cast<long long>(comparison.invalid),
                    cache_band_eligible ? "yes" : "no",
                    accepted ? "yes" : "no",
                    measure_solver ? double(amrex::second() - cache_start)
                                   : 0.0,
                    phase, recenter_pass, double(half_width_factor),
                    double(radial_geometry_cache_level_width_factor),
                    static_cast<long long>(cache_route_counts[6]));
              }
              return accepted;
            };
        amrex::Real radial_cache_half_width =
            radial_geometry_cache_half_width_factor;
        int radial_cache_recenter_passes_completed = 0;
        bool radial_cache_active = false;
        if (cache_this_batch)
          radial_cache_active = build_radial_cache(
              initial_cache_layers, radial_cache_half_width, "initial", 0);
        auto residual = convergence(minimum_relaxation_steps == 0,
                                    !radial_cache_active);
        const auto restore_direct_geometry =
            [&](const bool reopen_cached_terminals) {
              if (reopen_cached_terminals)
                state.logical_surfaces
                    ->reactivate_due_layers_for_direct_validation();
              state.logical_surfaces->rebuild_positions();
              route_updates();
              redistribute_unbounded();
              sample_adm();
              route_samples();
              evaluate_surface();
              radial_cache_active = false;
            };
        // A cached terminal state is provisional. Validate it against the
        // mesh first; only a layer reopened by that direct check may use a
        // narrower cache, and every narrower terminal is checked again here.
        const auto validate_cached_terminal =
            [&](const bool accept_convergence, const bool allow_recenter) {
              while (radial_cache_active && residual.active == 0) {
                restore_direct_geometry(true);
                residual = convergence(accept_convergence);
                if (residual.active == 0 ||
                    !allow_recenter ||
                    radial_cache_recenter_passes_completed >=
                        radial_geometry_cache_recenter_passes)
                  break;
                radial_cache_half_width *=
                    radial_geometry_cache_recenter_width_factor;
                const int recenter_pass =
                    radial_cache_recenter_passes_completed + 1;
                const auto recenter_cache_layers =
                    local_active_radial_cache_layers(state);
                radial_cache_active = build_radial_cache(
                    recenter_cache_layers, radial_cache_half_width,
                    "recenter", recenter_pass);
                if (!radial_cache_active)
                  break;
                radial_cache_recenter_passes_completed = recenter_pass;
                residual = convergence(accept_convergence, false);
                if (residual.invalid != 0) {
                  restore_direct_geometry(false);
                  residual = convergence(accept_convergence);
                }
              }
            };
        if (radial_cache_active && residual.invalid != 0) {
          restore_direct_geometry(false);
          residual = convergence(minimum_relaxation_steps == 0);
        }
        validate_cached_terminal(minimum_relaxation_steps == 0, true);
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
              const amrex::Real extrapolation_start =
                  measure_solver ? amrex::second() : 0;
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
                          ? RelaxationExtrapolationMode::low_modes
                          : RelaxationExtrapolationMode::full_shape);
                  baseline_ratio.push_back(ratio);
                  baseline_l2.push_back(entry.expansion_l2);
                  baseline_linf.push_back(entry.expansion_linf);
                }
              }
              if (active_layers.empty())
                return false;

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
              bool cached_trial_left_interval = false;

              const auto evaluate_controls =
                  [&](const bool reset_accepted_velocity) {
                    state.logical_surfaces->set_extrapolation_trial(
                        controls, eta_damping_times_mass,
                        minimum_radius_factor, maximum_radius_factor,
                        reset_accepted_velocity);
                    state.logical_surfaces->rebuild_positions();
                    if (radial_cache_active) {
                      state.radial_geometry_cache->interpolate(
                          state.logical_surfaces->device_view());
                      evaluate_surface();
                    } else {
                      route_updates();
                      // Extrapolation can intentionally move farther than one
                      // ordinary CFL-limited step. Reassign within the current
                      // synchronized band; a required level outside the band
                      // is detected by sample_adm() and deferred atomically.
                      redistribute_unbounded();
                      sample_adm();
                      route_samples();
                      evaluate_surface();
                      route_updates();
                    }
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
                state.logical_surfaces->evaluate_solver_residuals();
                const auto trial_residuals =
                    collect_layer_residuals(state, active_layers);
                for (std::size_t index = 0; index < active_layers.size();
                     ++index) {
                  if (!searching[index])
                    continue;
                  const auto &trial = trial_residuals[index];
                  cached_trial_left_interval |=
                      radial_cache_active && trial.invalid_points != 0;
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
                              RelaxationExtrapolationMode::low_modes
                          ? "monopole-dipole"
                          : "full-shape",
                      double(best_overstep[index]),
                      double(baseline_ratio[index]),
                      double(best_ratio[index]),
                      double(surface.mass_scale * baseline_l2[index]),
                      double(surface.mass_scale * best_l2[index]),
                      double(surface.mass_scale * baseline_linf[index]),
                      double(surface.mass_scale * best_linf[index]));
                }
              if (measure_solver) {
                solver_timing.extrapolation +=
                    amrex::second() - extrapolation_start;
                ++solver_timing.extrapolation_calls;
              }
              return cached_trial_left_interval;
            };
        if (relaxation_report_every > 0)
          report_relaxation_trials(state, completed_steps,
                                   completed_pseudo_time, "initial");

        while (completed_steps < maximum_relaxation_steps &&
               residual.active > 0) {
          const bool cached_step = radial_cache_active;
          const amrex::Real pseudo_step_start =
              measure_solver ? amrex::second() : 0;
          state.logical_surfaces->prepare_relaxation_time_steps(
              relaxation_cfl_factor);
          amrex::Real pseudo_dt = 0;
          if (!radial_cache_active) {
            pseudo_dt = state.logical_surfaces
                            ->local_minimum_relaxation_time_step();
            amrex::ParallelDescriptor::ReduceRealMin(pseudo_dt);
            ++solver_timing.timestep_min_collectives;
            if (pseudo_dt ==
                std::numeric_limits<amrex::Real>::max()) {
              residual = convergence(completed_steps >=
                                     minimum_relaxation_steps);
              if (residual.active == 0)
                break;
              CCTK_ERROR("ParticleAHFinderX direct relaxation retained an "
                         "active layer without a finite local timestep");
            }
            if (!(pseudo_dt > 0) ||
                !amrex::Math::isfinite(pseudo_dt)) {
              residual.failed += residual.active;
              residual.active = 0;
              break;
            }
          }

          state.logical_surfaces->begin_relaxation_step();
          for (int stage = 0; stage < num_relaxation_stages; ++stage) {
            state.logical_surfaces->advance_relaxation_stage(
                integrator, stage, eta_damping_times_mass,
                angular_dissipation_strength, minimum_radius_factor,
                maximum_radius_factor);
            if (stage + 1 < num_relaxation_stages) {
              state.logical_surfaces->evaluate_surface_geometry();
            }
          }
          amrex::Long invalid_relaxation = 0;
          if (radial_cache_active) {
            // Cached relaxation needs no mesh-owner notification. Mark and
            // restore any invalid surface on the device, then let the
            // following synchronized convergence record provide the
            // authoritative global failure count. This removes a device
            // reduction, host
            // synchronization, and MPI collective from every ordinary
            // cached pseudo-step.
            state.logical_surfaces
                ->reject_invalid_relaxation_steps_on_device();
          } else {
            invalid_relaxation =
                state.logical_surfaces->reject_invalid_relaxation_steps();
            amrex::ParallelDescriptor::ReduceLongSum(invalid_relaxation);
          }
          if (invalid_relaxation != 0 && verbose)
            CCTK_VINFO("Rejected %lld invalid relaxation point update(s); "
                       "only affected surfaces will be rolled back",
                       static_cast<long long>(invalid_relaxation));
          state.logical_surfaces->accumulate_relaxation_pseudo_time();

          state.logical_surfaces->rebuild_positions();
          // The final internal stage used geometry sampled at the start of
          // this pseudo-step. The cache path refreshes all 30 ADM components
          // locally on the owner GPU. The direct path updates the persistent
          // particle positions and re-gathers the frozen physical slice.
          if (radial_cache_active) {
            state.radial_geometry_cache->interpolate(
                state.logical_surfaces->device_view());
            evaluate_surface();
          } else {
            route_updates();
            // Rejecting one point atomically restores its entire surface to
            // the committed shape. That rollback is not bounded by the
            // ordinary per-step cell-displacement contract.
            if (invalid_relaxation != 0)
              redistribute_unbounded();
            // Ordinary accepted motion stays on its current mesh owner while
            // that owner still has a complete stencil. sample_adm() performs
            // redistribution only after an actual miss.
            sample_adm();
            route_samples();
            evaluate_surface();
          }
          ++completed_steps;
          if (!cached_step)
            completed_pseudo_time += pseudo_dt;
          const bool synchronize_convergence =
              !radial_cache_active ||
              completed_steps %
                      radial_geometry_cache_convergence_interval ==
                  0 ||
              completed_steps == maximum_relaxation_steps ||
              (relaxation_report_every > 0 &&
               completed_steps % relaxation_report_every == 0) ||
              (radial_geometry_cache_direct_step_reserve > 0 &&
               completed_steps >=
                   maximum_relaxation_steps -
                       radial_geometry_cache_direct_step_reserve);
          amrex::Real synchronized_pseudo_time =
              std::numeric_limits<amrex::Real>::max();
          if (cached_step && synchronize_convergence) {
            synchronized_pseudo_time =
                state.logical_surfaces
                    ->local_minimum_relaxation_pseudo_time();
            amrex::ParallelDescriptor::ReduceRealMin(
                synchronized_pseudo_time);
            ++solver_timing.pseudo_time_min_collectives;
          }
          if (synchronize_convergence) {
            residual = convergence(
                completed_steps >= minimum_relaxation_steps,
                !radial_cache_active);
          } else {
            state.logical_surfaces->update_solver_convergence_on_device(
                completed_steps >= minimum_relaxation_steps,
                angular_continuation_cutoff_threshold,
                angular_continuation_helper_tolerance_factor,
                !radial_cache_active);
            ++solver_timing.device_convergence_checks;
          }
          if (radial_cache_active && residual.invalid != 0) {
            restore_direct_geometry(false);
            residual = convergence(completed_steps >=
                                   minimum_relaxation_steps);
          }
          if (cached_step && synchronize_convergence) {
            if (synchronized_pseudo_time !=
                std::numeric_limits<amrex::Real>::max()) {
              if (!(synchronized_pseudo_time >= 0) ||
                  !amrex::Math::isfinite(synchronized_pseudo_time))
                CCTK_ERROR("ParticleAHFinderX cached relaxation produced "
                           "an invalid synchronized pseudo-time");
              completed_pseudo_time = synchronized_pseudo_time;
            } else if (residual.active != 0) {
              CCTK_ERROR("ParticleAHFinderX cached relaxation retained an "
                         "active layer without a finite pseudo-time");
            }
          }
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
              if (relaxation_extrapolation_reference_event(
                      schedule.next_event)) {
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
              const bool cached_trial_left_interval =
                  test_relaxation_extrapolation(extrapolation_layers);
              residual = convergence(
                  completed_steps >= minimum_relaxation_steps,
                  !radial_cache_active);
              if (radial_cache_active &&
                  (cached_trial_left_interval || residual.invalid != 0)) {
                if (cached_trial_left_interval && verbose &&
                    amrex::ParallelDescriptor::IOProcessor())
                  CCTK_VINFO(
                      "Radial geometry cache handed %lld active layer(s) "
                      "to direct sampling after an extrapolation trial "
                      "left its interval at step %d",
                      static_cast<long long>(residual.active),
                      completed_steps);
                restore_direct_geometry(cached_trial_left_interval);
                residual = convergence(completed_steps >=
                                       minimum_relaxation_steps);
              }
            }
          }
          const bool cache_recenter_budget_available =
              completed_steps < maximum_relaxation_steps &&
              (radial_geometry_cache_direct_step_reserve == 0 ||
               completed_steps <
                   maximum_relaxation_steps -
                       radial_geometry_cache_direct_step_reserve);
          validate_cached_terminal(
              completed_steps >= minimum_relaxation_steps,
              cache_recenter_budget_available);
          const bool cache_step_budget_exhausted =
              radial_cache_active &&
              residual.active > 0 &&
              radial_geometry_cache_direct_step_reserve > 0 &&
              completed_steps >=
                  maximum_relaxation_steps -
                      radial_geometry_cache_direct_step_reserve;
          if (cache_step_budget_exhausted) {
            if (verbose && amrex::ParallelDescriptor::IOProcessor())
              CCTK_VINFO(
                  "Radial geometry cache handed %lld active layer(s) to "
                  "direct sampling after %d step(s), reserving %d direct "
                  "step(s)",
                  static_cast<long long>(residual.active), completed_steps,
                  radial_geometry_cache_direct_step_reserve);
            restore_direct_geometry(true);
            residual = convergence(completed_steps >=
                                   minimum_relaxation_steps);
          }
          if (relaxation_report_every > 0 &&
              completed_steps % relaxation_report_every == 0)
            report_relaxation_trials(state, completed_steps,
                                     completed_pseudo_time, "progress");
          if (measure_solver) {
            solver_timing.pseudo_steps += amrex::second() - pseudo_step_start;
            ++solver_timing.pseudo_steps_timed;
          }
          auto &phase_steps = cached_step ? cached_completed_steps
                                          : direct_completed_steps;
          auto &phase_point_steps =
              cached_step ? cached_point_steps : direct_point_steps;
          if (phase_steps == std::numeric_limits<amrex::Long>::max() ||
              phase_point_steps >
                  std::numeric_limits<amrex::Long>::max() -
                      active_particle_count)
            CCTK_ERROR("ParticleAHFinderX cache-phase work counter "
                       "overflowed");
          ++phase_steps;
          phase_point_steps += active_particle_count;
        }

        if (radial_cache_active) {
          restore_direct_geometry(true);
          residual = convergence(completed_steps >=
                                 minimum_relaxation_steps);
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
        std::vector<SurfaceLayerProlongation> promoted_layers;
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
          const bool target_layer = outcome.selection.angular_level + 1 ==
                                    surface.angular_levels;
          if (outcome.relaxation_state ==
              SurfaceRelaxationState::failed) {
            maximum_failed_terminal_ratio =
                amrex::max(maximum_failed_terminal_ratio, ratio);
            surface.relaxation_state = SurfaceRelaxationState::failed;
            failed_surface_ids.push_back(surface.stable_id);
            terminal_layers.push_back(outcome.selection);
          } else if (!target_layer) {
            maximum_helper_l2_times_mass =
                amrex::max(maximum_helper_l2_times_mass, l2_times_mass);
            maximum_helper_linf_times_mass =
                amrex::max(maximum_helper_linf_times_mass, linf_times_mass);
            ++successful_helper_layers;
            const bool additive_correction = std::binary_search(
                multilevel_tracking_ids.begin(),
                multilevel_tracking_ids.end(), surface.stable_id);
            promoted_layers.push_back(
                {outcome.selection.stable_id,
                 outcome.selection.angular_level,
                 additive_correction
                     ? SurfaceProlongationMode::additive_correction
                     : SurfaceProlongationMode::replace});
            next_layers.push_back(
                {surface.stable_id, outcome.selection.angular_level + 1});
            surface.active_angular_level =
                outcome.selection.angular_level + 1;
          } else {
            if (outcome.relaxation_state ==
                SurfaceRelaxationState::refine)
              CCTK_ERROR("ParticleAHFinderX target angular layer requested "
                         "an impossible further refinement");
            maximum_accepted_target_l2_times_mass = amrex::max(
                maximum_accepted_target_l2_times_mass, l2_times_mass);
            maximum_accepted_target_linf_times_mass = amrex::max(
                maximum_accepted_target_linf_times_mass, linf_times_mass);
            ++accepted_target_layers;
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

        // The committed source buffer still contains the pre-solve coarse
        // shape here. Acquisition replaces the next layer with the prolonged
        // solution; tracking prolongs only the solved coarse correction and
        // therefore preserves the destination's pre-existing fine modes.
        state.logical_surfaces->prolong_surface_layers(promoted_layers);
        // Commit successful source layers, or atomically restore failed ones,
        // only after additive prolongation has consumed that baseline.
        state.logical_surfaces->finish_relaxation_search();
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
          return_route.valid = false;
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
        // the bounded per-step allowance. Keep reassignment inside the current
        // synchronized band; sample_adm() defers if another level is needed.
        redistribute_unbounded();
        sample_adm();
        route_samples();
        evaluate_surface();
        route_updates();
        pending_layers = std::move(deferred_batch_layers);
        pending_layers.insert(pending_layers.end(), next_layers.begin(),
                              next_layers.end());
        std::sort(pending_layers.begin(), pending_layers.end(),
                  [](const SurfaceLayerSelection &first,
                     const SurfaceLayerSelection &second) {
                    return first.stable_id < second.stable_id ||
                           (first.stable_id == second.stable_id &&
                            first.angular_level < second.angular_level);
                  });
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
        return_route.valid = false;
        state.logical_surfaces->set_due_surface_layers(
            searched_target_layers);
        for (auto &container : state.containers)
          container->set_active_surface_layers(searched_target_layers);
        active_particle_count =
            selected_particle_count(state, searched_target_layers);
        sample_adm();
        route_samples();
        evaluate_surface();
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
      if (verbose && amrex::ParallelDescriptor::IOProcessor())
        CCTK_VINFO(
            "GPU hyperbolic relaxation (%s) completed %d sequential "
            "level-step(s), point-steps=%lld, cache/direct "
            "level-steps=%lld/%lld, cache/direct point-steps=%lld/%lld, "
            "extrapolation trials=%d point-evaluations=%lld: "
            "converged=%lld failed=%lld "
            "accepted target layers=%lld max accepted target (M*L2)=%.3g "
            "max accepted target (M*Linf)=%.3g; successful helper "
            "layers=%lld max helper (M*L2)=%.3g max helper "
            "(M*Linf)=%.3g; failed layers=%lld max failed ratio=%.3g",
            relaxation_integrator, total_completed_steps,
            static_cast<long long>(processed_point_steps),
            static_cast<long long>(cached_completed_steps),
            static_cast<long long>(direct_completed_steps),
            static_cast<long long>(cached_point_steps),
            static_cast<long long>(direct_point_steps),
            total_extrapolation_trials,
            static_cast<long long>(extrapolation_point_evaluations),
            static_cast<long long>(converged_surfaces),
            static_cast<long long>(failed_surfaces),
            static_cast<long long>(accepted_target_layers),
            double(maximum_accepted_target_l2_times_mass),
            double(maximum_accepted_target_linf_times_mass),
            static_cast<long long>(successful_helper_layers),
            double(maximum_helper_l2_times_mass),
            double(maximum_helper_linf_times_mass),
            static_cast<long long>(failed_surfaces),
            double(maximum_failed_terminal_ratio));
      amrex::Long failed_published_surfaces = 0;
      amrex::Long failed_unestablished_common_seeds = 0;
      amrex::Long failed_provisional_surfaces = 0;
      amrex::Real maximum_failed_published_ratio = 0;
      amrex::Real maximum_failed_unestablished_common_ratio = 0;
      amrex::Real maximum_failed_provisional_ratio = 0;
      for (const auto stable_id : failed_surface_ids) {
        const auto found = std::lower_bound(
            state.surfaces.begin(), state.surfaces.end(), stable_id,
            [](const SurfaceRecord &surface, const amrex::Long id) {
              return surface.stable_id < id;
            });
        if (found == state.surfaces.end() || found->stable_id != stable_id)
          CCTK_ERROR("ParticleAHFinderX lost a failed surface while "
                     "reporting its terminal state");
        const auto &failed_layer =
            found->angular_layer.at(found->active_angular_level);
        const amrex::Real failed_ratio = amrex::max(
            found->mass_scale * failed_layer.expansion_l2 /
                found->tolerances.l2_times_mass,
            found->mass_scale * failed_layer.expansion_linf /
                found->tolerances.linf_times_mass);
        if (found->published && found->role == SurfaceRole::common &&
            found->last_success_iteration < 0) {
          ++failed_unestablished_common_seeds;
          maximum_failed_unestablished_common_ratio = amrex::max(
              maximum_failed_unestablished_common_ratio, failed_ratio);
        } else if (found->published) {
          ++failed_published_surfaces;
          maximum_failed_published_ratio =
              amrex::max(maximum_failed_published_ratio, failed_ratio);
        } else {
          ++failed_provisional_surfaces;
          maximum_failed_provisional_ratio =
              amrex::max(maximum_failed_provisional_ratio, failed_ratio);
        }
      }
      if (amrex::ParallelDescriptor::IOProcessor()) {
        if (failed_published_surfaces != 0)
          CCTK_VWARN(
              CCTK_WARN_ALERT,
              "ParticleAHFinderX restored %lld unconverged published "
              "surface(s) after %d sequential angular-level step(s) "
              "(max terminal ratio %.3g)",
              static_cast<long long>(failed_published_surfaces),
              total_completed_steps,
              double(maximum_failed_published_ratio));
        if (verbose && failed_unestablished_common_seeds != 0)
          CCTK_VINFO(
              "ParticleAHFinderX left %lld configured common seed(s) "
              "without an accepted horizon after %d sequential "
              "angular-level step(s); no previously converged horizon was "
              "replaced (max terminal ratio %.3g)",
              static_cast<long long>(failed_unestablished_common_seeds),
              total_completed_steps,
              double(maximum_failed_unestablished_common_ratio));
        if (verbose && failed_provisional_surfaces != 0)
          CCTK_VINFO(
              "ParticleAHFinderX left %lld unconverged provisional "
              "surface(s) unpublished after %d sequential angular-level "
              "step(s); published horizons are unchanged (max terminal "
              "ratio %.3g)",
              static_cast<long long>(failed_provisional_surfaces),
              total_completed_steps,
              double(maximum_failed_provisional_ratio));
      }
      state.last_relaxation_steps = total_completed_steps;
      state.last_search_converged = failed_surfaces == 0;
    } else {
      route_updates();
    }
    if (!solve_surfaces)
      update_surface_diagnostics(
          *state.logical_surfaces, state.surfaces, cctk_iteration,
          static_cast<amrex::Real>(cctk_time), 0);
    if (band.complete_hierarchy && !forced)
      finalize_candidates_after_search(
          state, cctk_iteration, static_cast<amrex::Real>(cctk_time));

    if (solve_surfaces && !searched_target_layers.empty()) {
      const auto analysis_target_layers = final_analysis_surface_layers(
          state, searched_target_layers, cctk_iteration);
      if (!same_surface_layers(analysis_target_layers,
                               searched_target_layers)) {
        // Retirement compacts logical storage, while a retry reset can
        // reorder persistent mesh particles. Rebuild only the surviving
        // target topology from their already sampled device data; no second
        // ADM gather or host point staging is needed.
        return_route.valid = false;
        state.logical_surfaces->set_due_surface_layers(
            analysis_target_layers);
        for (auto &container : state.containers)
          container->set_active_surface_layers(analysis_target_layers);
        active_particle_count =
            analysis_target_layers.empty()
                ? 0
                : selected_particle_count(state, analysis_target_layers);
        if (!analysis_target_layers.empty())
          route_samples();
      }
      if (!analysis_target_layers.empty() &&
          (!return_route.valid ||
           return_route.active_point_count != active_particle_count))
        CCTK_ERROR("ParticleAHFinderX final analysis route does not match "
                   "the surviving target-layer population");
      state.analysis_active_point_count = active_particle_count;
    } else if (solve_surfaces) {
      // A helper layer can fail before any target layer is searched. Do not
      // leave that internal route available to the subsequent analysis
      // schedule.
      return_route.valid = false;
      state.logical_surfaces->set_due_surface_layers({});
      for (auto &container : state.containers)
        container->set_active_surface_layers({});
      active_particle_count = 0;
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
            "Solver timing: iteration=%d time=%.17g band=[%d,%d) "
            "level_iteration=%lld/%lld complete_hierarchy=%s surfaces=%zu "
            "points=%lld steps=%d rank wall min/avg/max=%.6g/%.6g/%.6g s",
            cctk_iteration, double(cctk_time), band.minimum_level,
            band.maximum_level,
            static_cast<long long>(band.logical_iteration_numerator),
            static_cast<long long>(band.logical_iteration_denominator),
            band.complete_hierarchy ? "yes" : "no",
            due_surface_ids.size(),
            static_cast<long long>(due_particle_count),
            state.last_relaxation_steps, double(wall_times[0]),
            double(wall_times[1]), double(wall_times[2]));

      amrex::Real component_times[8]{
          solver_timing.adm_gather,
          solver_timing.mesh_to_logical,
          solver_timing.source_metadata,
          solver_timing.logical_to_mesh,
          solver_timing.redistribution,
          solver_timing.convergence,
          solver_timing.pseudo_steps,
          solver_timing.extrapolation};
      amrex::ParallelDescriptor::ReduceRealMax(component_times, 8);
      amrex::Long component_calls[9]{
          solver_timing.adm_gather_calls,
          solver_timing.redistribution_calls,
          solver_timing.convergence_calls,
          solver_timing.pseudo_steps_timed,
          solver_timing.extrapolation_calls,
          mesh_to_logical_calls,
          source_metadata_calls,
          logical_to_mesh_calls,
          solver_timing.device_convergence_checks};
      amrex::ParallelDescriptor::ReduceLongMax(component_calls, 9);
      amrex::Long scalar_collectives[4]{
          solver_timing.timestep_min_collectives,
          solver_timing.pseudo_time_min_collectives,
          component_calls[2], component_calls[2]};
      if (amrex::ParallelDescriptor::IOProcessor()) {
        CCTK_VINFO(
            "Solver timing components (rank maximum): ADM gather=%.6g s "
            "(%lld calls), mesh-to-logical=%.6g s (%lld calls), "
            "source-metadata=%.6g s (%lld calls), logical-to-mesh=%.6g s "
            "(%lld calls), redistribution=%.6g s (%lld calls), "
            "convergence=%.6g s (%lld synchronized calls, %lld "
            "device-only checks)",
            double(component_times[0]),
            static_cast<long long>(component_calls[0]),
            double(component_times[1]),
            static_cast<long long>(component_calls[5]),
            double(component_times[2]),
            static_cast<long long>(component_calls[6]),
            double(component_times[3]),
            static_cast<long long>(component_calls[7]),
            double(component_times[4]),
            static_cast<long long>(component_calls[1]),
            double(component_times[5]),
            static_cast<long long>(component_calls[2]),
            static_cast<long long>(component_calls[8]));
        CCTK_VINFO(
            "Solver timing aggregate intervals (rank maximum): pseudo-step="
            "%.6g s (%lld steps), extrapolation=%.6g s (%lld calls); these "
            "intervals include nested components and must not be summed",
            double(component_times[6]),
            static_cast<long long>(component_calls[3]),
            double(component_times[7]),
            static_cast<long long>(component_calls[4]));
        CCTK_VINFO(
            "Solver compact scalar MPI collectives: timestep-min=%lld "
            "pseudo-time-min=%lld convergence-max=%lld "
            "convergence-sum=%lld total=%lld; "
            "cached-convergence-interval=%d device-only-checks=%lld",
            static_cast<long long>(scalar_collectives[0]),
            static_cast<long long>(scalar_collectives[1]),
            static_cast<long long>(scalar_collectives[2]),
            static_cast<long long>(scalar_collectives[3]),
            static_cast<long long>(scalar_collectives[0] +
                                   scalar_collectives[1] +
                                   scalar_collectives[2] +
                                   scalar_collectives[3]),
            radial_geometry_cache_convergence_interval,
            static_cast<long long>(component_calls[8]));
      }

      amrex::Long byte_counters[8]{
          routing_performance.device_payload_bytes_sent,
          routing_performance.device_payload_bytes_received,
          routing_performance.device_local_copy_bytes,
          routing_performance.host_point_data_bytes,
          routing_performance.control_device_to_host_bytes,
          routing_performance.control_host_to_device_bytes,
          routing_performance.mpi_control_bytes_sent,
          routing_performance.mpi_control_bytes_received};
      amrex::ParallelDescriptor::ReduceLongSum(byte_counters, 8);
      amrex::Long call_counters[10]{
          routing_performance.device_to_host_control_copies,
          routing_performance.host_to_device_control_copies,
          routing_performance.mpi_count_collectives,
          routing_performance.mpi_payload_exchange_rounds,
          routing_performance.explicit_device_synchronizations,
          mesh_to_logical_calls, source_metadata_calls,
          logical_to_mesh_calls,
          routing_performance.forward_topology_reuses,
          routing_performance.forward_topology_rebuilds};
      amrex::ParallelDescriptor::ReduceLongMax(call_counters, 10);
      amrex::Long message_counters[2]{
          routing_performance.mpi_payload_messages_sent,
          routing_performance.mpi_payload_messages_received};
      amrex::ParallelDescriptor::ReduceLongSum(message_counters, 2);
      if (amrex::ParallelDescriptor::IOProcessor()) {
        CCTK_VINFO(
            "Solver routing residency: mesh-to-logical calls=%lld "
            "metadata-refresh calls=%lld logical-to-mesh calls=%lld; "
            "forward topology reuse/rebuild=%lld/%lld; point-data host "
            "staging=%lld B",
            static_cast<long long>(call_counters[5]),
            static_cast<long long>(call_counters[6]),
            static_cast<long long>(call_counters[7]),
            static_cast<long long>(call_counters[8]),
            static_cast<long long>(call_counters[9]),
            static_cast<long long>(byte_counters[3]));
        CCTK_VINFO(
            "Solver routing payload bytes (rank sum): device MPI "
            "sent/received=%lld/%lld local-device-copy=%lld; MPI count "
            "sent/received=%lld/%lld",
            static_cast<long long>(byte_counters[0]),
            static_cast<long long>(byte_counters[1]),
            static_cast<long long>(byte_counters[2]),
            static_cast<long long>(byte_counters[6]),
            static_cast<long long>(byte_counters[7]));
        CCTK_VINFO(
            "Solver routing compact control (rank-sum bytes, maximum "
            "per-rank calls): device-to-host=%lld B/%lld host-to-device=%lld "
            "B/%lld MPI count collectives/payload exchange rounds=%lld/%lld "
            "payload messages sent/received=%lld/%lld explicit device "
            "synchronizations=%lld",
            static_cast<long long>(byte_counters[4]),
            static_cast<long long>(call_counters[0]),
            static_cast<long long>(byte_counters[5]),
            static_cast<long long>(call_counters[1]),
            static_cast<long long>(call_counters[2]),
            static_cast<long long>(call_counters[3]),
            static_cast<long long>(message_counters[0]),
            static_cast<long long>(message_counters[1]),
            static_cast<long long>(call_counters[4]));
      }
    }
    if (!forced)
      update_tracking_history(cctkGH, state, cctk_iteration,
                              static_cast<amrex::Real>(cctk_time));
  } catch (const ActiveBandDeferred &error) {
    // Every selected logical layer retains a committed device state. Restore
    // it even if a coarse continuation layer just finished: in that case the
    // completed source remains committed while its unsafe fine prolongation is
    // discarded. Refresh reverse-routing addresses after any preceding band
    // redistribution, then return the persistent mesh particles to the same
    // state without touching an asynchronous level.
    state.logical_surfaces->defer_relaxation_search();
    state.logical_surfaces->rebuild_positions();
    const auto metadata = route_particles_to_logical_owners(
        state.containers, *state.logical_surfaces,
        LogicalRoutingMode::source_metadata, return_route,
        active_particle_count);
    accumulate_routing_stats(routing_performance, metadata);
    ++source_metadata_calls;
    amrex::Long metadata_stats[5]{metadata.sent, metadata.received,
                                  metadata.missing, metadata.duplicate,
                                  metadata.invalid};
    amrex::ParallelDescriptor::ReduceLongSum(metadata_stats, 5);
    if (metadata_stats[0] != active_particle_count ||
        metadata_stats[1] != active_particle_count ||
        metadata_stats[2] != 0 || metadata_stats[3] != 0 ||
        metadata_stats[4] != 0)
      CCTK_ERROR("ParticleAHFinderX could not restore source metadata after "
                 "an active-band deferral");
    const auto updates = route_updates_to_mesh_owners(
        *state.logical_surfaces, state.containers, return_route,
        band.minimum_level,
        band.maximum_level);
    accumulate_routing_stats(routing_performance, updates);
    ++logical_to_mesh_calls;
    amrex::Long update_stats[6]{
        updates.sent,    updates.received, updates.missing,
        updates.duplicate, updates.invalid, updates.outside_active_band};
    amrex::ParallelDescriptor::ReduceLongSum(update_stats, 6);
    if (update_stats[0] != active_particle_count ||
        update_stats[1] != active_particle_count || update_stats[2] != 0 ||
        update_stats[3] != 0 || update_stats[4] != 0 ||
        update_stats[5] != 0)
      CCTK_ERROR("ParticleAHFinderX could not restore mesh particles after "
                 "an active-band deferral");
    redistribute_unbounded();
    state.logical_surfaces->set_due_surface_ids({});
    for (auto &container : state.containers)
      container->set_active_surface_layers({});
    if (verbose)
      CCTK_VINFO(
          "Deferred %zu surface search(es) at iteration %d time %.17g: %s; "
          "active band=[%d,%d) level_iteration=%lld/%lld",
          due_surface_ids.size(), cctk_iteration, double(cctk_time),
          error.what(), band.minimum_level, band.maximum_level,
          static_cast<long long>(band.logical_iteration_numerator),
          static_cast<long long>(band.logical_iteration_denominator));
    return;
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

extern "C" void ParticleAHFinderX_Analysis(CCTK_ARGUMENTS) {
  run_analysis(cctkGH, -1);
}

extern "C" CCTK_INT ParticleAHFinderX_MeasureInitialMasses_Impl(
    CCTK_POINTER_TO_CONST cctkGH_pointer, const CCTK_INT npunctures,
    CCTK_REAL *const measured_mass, CCTK_INT *const valid) {
  if (!cctkGH_pointer || npunctures < 0 || !measured_mass || !valid)
    return -1;
  for (int puncture = 0; puncture < npunctures; ++puncture) {
    measured_mass[puncture] =
        std::numeric_limits<CCTK_REAL>::quiet_NaN();
    valid[puncture] = 0;
  }

  const auto *const cctkGH = static_cast<const cGH *>(cctkGH_pointer);
  auto &state = runtime();
  if (!state.setup || !state.initial_surfaces_created)
    return -1;

  run_analysis(cctkGH, npunctures);
  SurfaceAnalysisOptions analysis_options;
  analysis_options.compute_invariant_spin = true;
  const auto records = current_surface_analysis(
      cctkGH, cctkGH->cctk_iteration, false, nullptr, analysis_options);
  for (const auto &record : records) {
    const auto surface = std::lower_bound(
        state.surfaces.begin(), state.surfaces.end(), record.stable_id,
        [](const SurfaceRecord &candidate, const std::int64_t stable_id) {
          return candidate.stable_id < stable_id;
        });
    if (surface == state.surfaces.end() ||
        surface->stable_id != record.stable_id)
      CCTK_ERROR("ParticleAHFinderX calibration lost a surface record");
    const int puncture = surface->configured_slot;
    const CCTK_REAL mass = record.invariant_spin_valid
                               ? record.invariant_christodoulou_mass
                               : record.christodoulou_mass;
    if (puncture >= 0 && puncture < npunctures && std::isfinite(mass) &&
        mass > 0.0) {
      measured_mass[puncture] = mass;
      valid[puncture] = 1;
    }
    surface->last_analysis_api_iteration = -1;
  }
  return 0;
}

extern "C" int ParticleAHFinderX_Shutdown() {
  runtime() = Runtime{};
  return 0;
}

} // namespace ParticleAHFinderX
