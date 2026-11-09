/**
 * \file analysis_api.cxx
 * \brief Collective bounded adapter for GPU-native horizon analysis.
 */
#include "analysis_api.hxx"
#include "runtime.hxx"

#include <AMReX_Gpu.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <type_traits>

namespace ParticleAHFinderX {

namespace {

constexpr std::array<const char *, 8> gauge_variable_names{{
    "ADMBaseX::alp",     "ADMBaseX::betax",   "ADMBaseX::betay",
    "ADMBaseX::betaz",   "ADMBaseX::dtalp",   "ADMBaseX::dtbetax",
    "ADMBaseX::dtbetay", "ADMBaseX::dtbetaz",
}};

constexpr std::array<const char *, 10> matter_variable_names{{
    "TmunuBaseX::eTtt", "TmunuBaseX::eTtx", "TmunuBaseX::eTty",
    "TmunuBaseX::eTtz", "TmunuBaseX::eTxx", "TmunuBaseX::eTxy",
    "TmunuBaseX::eTxz", "TmunuBaseX::eTyy", "TmunuBaseX::eTyz",
    "TmunuBaseX::eTzz",
}};

} // namespace

SurfaceAnalysisRuntimeStatus surface_analysis_runtime_status() {
  const auto &state = runtime();
  SurfaceAnalysisRuntimeStatus status;
  status.ready = state.initial_surfaces_created && state.logical_surfaces &&
                 state.logical_surfaces->initialized();
  status.recovered = state.recovered;
  status.recovered_iteration = state.recovered_iteration;
#ifdef AMREX_USE_GPU
  status.accelerator_backend = true;
#endif
  status.gpu_aware_mpi =
      status.accelerator_backend &&
      amrex::ParallelDescriptor::UseGpuAwareMpi();
  return status;
}

std::vector<SurfaceAnalysisRecord>
current_surface_analysis(const cGH *const cctkGH, const int iteration,
                         const bool published_only,
                         SurfaceAnalysisPerformance *const performance,
                         const SurfaceAnalysisOptions &options) {
  static_assert(std::is_same_v<CCTK_REAL, amrex::Real>,
                "Cactus and AMReX real precision must match");
  static_assert(sizeof(std::int64_t) >= sizeof(amrex::Long),
                "Stable surface IDs require at least 64 public bits");

  if (performance)
    *performance = {};
  if (options.compute_multipoles && !options.compute_invariant_spin)
    amrex::Abort("ParticleAHFinderX multipoles require invariant-spin "
                 "analysis");
  const bool compute_ricci_scalars =
      options.ricci_source != SurfaceRicciSource::off;
  const bool use_matter_ricci =
      options.ricci_source == SurfaceRicciSource::tmunu;
  if (options.ricci_source != SurfaceRicciSource::off &&
      options.ricci_source != SurfaceRicciSource::vacuum &&
      options.ricci_source != SurfaceRicciSource::tmunu)
    amrex::Abort("ParticleAHFinderX received an invalid Ricci source");
  if ((options.compute_weyl_curvature || options.compute_weinberg_fluxes ||
       compute_ricci_scalars) &&
      !cctkGH)
    amrex::Abort("ParticleAHFinderX mesh-field analysis requires a valid "
                 "Cactus grid hierarchy");
  if (options.compute_invariant_spin &&
      (options.axial_harmonic_degree < 1 ||
       options.axial_harmonic_degree > maximum_axial_harmonic_degree ||
       options.axial_maximum_jacobi_sweeps <= 0 ||
       !(options.axial_eigen_tolerance > 0) ||
       !(options.axial_minimum_axis_separation > 0)))
    amrex::Abort("ParticleAHFinderX received invalid public axial-analysis "
                 "options");
  const amrex::Real total_start = amrex::second();
  auto &state = runtime();
  if (!state.initial_surfaces_created || !state.logical_surfaces ||
      !state.logical_surfaces->initialized()) {
    if (performance)
      performance->total_seconds = amrex::second() - total_start;
    return {};
  }
  constexpr int real_components = surface_analysis_real_components;
  constexpr int long_components = surface_analysis_long_components;
  const bool has_surface = std::any_of(
      state.surfaces.begin(), state.surfaces.end(),
      [=](const SurfaceRecord &surface) {
        return surface.last_search_iteration == iteration &&
               surface.last_analysis_api_iteration != iteration &&
               surface.relaxation_state ==
                   SurfaceRelaxationState::converged &&
               surface.lifecycle != ParticleLifecycle::retired &&
               (!published_only || surface.published);
      });
  if (!has_surface) {
    if (performance)
      performance->total_seconds = amrex::second() - total_start;
    return {};
  }

  // Reduce only surfaces searched and not yet consumed in this Cactus
  // iteration. Retired candidates remain in stable-ID history, but must not
  // make every later analysis collective grow with the lifetime birth count.
  std::vector<amrex::Long> surface_ids;
  surface_ids.reserve(state.surfaces.size());
  amrex::Long previous_id = 0;
  for (const auto &surface : state.surfaces) {
    if (surface.stable_id <= previous_id)
      amrex::Abort("ParticleAHFinderX analysis requires unique increasing "
                   "stable surface IDs");
    previous_id = surface.stable_id;
    if (surface.last_search_iteration == iteration &&
        surface.last_analysis_api_iteration != iteration)
      surface_ids.push_back(surface.stable_id);
  }
  if (surface_ids.empty())
    amrex::Abort("ParticleAHFinderX analysis found no unconsumed searched "
                 "surface IDs");
  if (surface_ids.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max() /
                               real_components))
    amrex::Abort("ParticleAHFinderX analysis record exceeds the portable "
                 "device/MPI index range");
  const int surface_count = static_cast<int>(surface_ids.size());
  if (performance) {
    performance->global_surface_count = surface_count;
    performance->direct_device_collective =
#ifdef AMREX_USE_GPU
        amrex::ParallelDescriptor::UseGpuAwareMpi();
#else
        false;
#endif
  }
  if (surface_ids != state.analysis_surface_ids) {
    state.analysis_surface_ids = surface_ids;
    state.analysis_surface_ids_device.resize(surface_ids.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, surface_ids.begin(),
                     surface_ids.end(),
                     state.analysis_surface_ids_device.begin());
    if (performance) {
      performance->host_to_device_copies = 1;
      performance->compact_host_to_device_bytes =
          static_cast<std::int64_t>(surface_ids.size() *
                                    sizeof(amrex::Long));
    }
  }
  state.analysis_reals_device.resize(
      std::size_t(real_components) * surface_count);
  state.analysis_longs_device.resize(
      std::size_t(long_components) * surface_count + 1);
  if (options.compute_weyl_curvature) {
    if (state.interpolation_order != 3)
      amrex::Abort("ParticleAHFinderX Weyl analysis requires cubic finder "
                   "interpolation");
    if (state.analysis_active_point_count <= 0 ||
        !state.analysis_route_plan.valid ||
        state.analysis_route_plan.active_point_count !=
            state.analysis_active_point_count)
      amrex::Abort("ParticleAHFinderX Weyl analysis has no final target-layer "
                   "sample route");

    amrex::Long local_population = 0;
    std::vector<amrex::Long> container_offsets(state.containers.size() + 1,
                                                0);
    for (std::size_t patch = 0; patch < state.containers.size(); ++patch) {
      local_population += state.containers[patch]->local_particle_count();
      container_offsets[patch + 1] = local_population;
    }
    const amrex::Real gather_start = amrex::second();
    state.analysis_weyl_mesh_device.resize(
        static_cast<std::size_t>(
            std::max<amrex::Long>(amrex::Long(1), local_population)));
    auto *const mesh_weyl = state.analysis_weyl_mesh_device.dataPtr();
    amrex::ParallelFor(
        local_population,
        [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
          mesh_weyl[point] = MeshWeylPoint{};
        });

    int gathered_levels = 0;
    if (!CarpetX::active_levels)
      amrex::Abort("ParticleAHFinderX Weyl analysis has no active CarpetX "
                   "levels");
    CarpetX::active_levels->loop_serially([&](auto &level) {
      const AdmFieldLayout fields = adm_field_layout(level, state.adm_fields);
      if (level.patch < 0 ||
          level.patch >= static_cast<int>(state.containers.size()))
        amrex::Abort("ParticleAHFinderX Weyl analysis received an invalid "
                     "CarpetX patch");
      state.containers[level.patch]->gather_weyl_level(
          level.level, *fields.metric, fields.metric_component,
          *fields.curv, fields.curv_component, fields.centering, mesh_weyl,
          container_offsets[level.patch]);
      ++gathered_levels;
    });
    if (gathered_levels == 0)
      amrex::Abort("ParticleAHFinderX Weyl analysis found no active level");

    std::array<amrex::Long, 4> gather_stats{{0, 0, 0, 0}};
    for (std::size_t patch = 0; patch < state.containers.size(); ++patch) {
      const auto stats = state.containers[patch]->local_weyl_gather_stats(
          mesh_weyl, container_offsets[patch]);
      gather_stats[0] += stats.valid;
      gather_stats[1] += stats.invalid_adm;
      gather_stats[2] += stats.stencil_unavailable;
      gather_stats[3] += stats.nonfinite;
    }
    amrex::ParallelDescriptor::ReduceLongSum(gather_stats.data(),
                                              int(gather_stats.size()));
    const amrex::Long classified =
        gather_stats[0] + gather_stats[1] + gather_stats[2] + gather_stats[3];
    if (classified != state.analysis_active_point_count ||
        gather_stats[0] != state.analysis_active_point_count ||
        gather_stats[1] != 0 || gather_stats[2] != 0 ||
        gather_stats[3] != 0)
      amrex::Abort("ParticleAHFinderX optional Weyl gather did not cover the "
                   "final target-layer particle route");
    if (performance) {
      performance->weyl_gather_kernel_launches =
          1 + gathered_levels + static_cast<int>(state.containers.size());
      performance->weyl_mesh_workspace_bytes =
          local_population * sizeof(MeshWeylPoint);
      performance->weyl_gather_seconds = amrex::second() - gather_start;
    }

    const amrex::Real routing_start = amrex::second();
    const auto routing = route_weyl_to_logical_owners(
        mesh_weyl, *state.logical_surfaces, state.analysis_route_plan,
        state.analysis_active_point_count, local_population);
    std::array<amrex::Long, 5> global_routing{{
        routing.sent, routing.received, routing.missing, routing.duplicate,
        routing.invalid}};
    amrex::ParallelDescriptor::ReduceLongSum(global_routing.data(),
                                              int(global_routing.size()));
    if (global_routing[0] != state.analysis_active_point_count ||
        global_routing[1] != state.analysis_active_point_count ||
        global_routing[2] != 0 || global_routing[3] != 0 ||
        global_routing[4] != 0)
      amrex::Abort("ParticleAHFinderX optional Weyl route is incomplete");
    if (performance) {
      // Pack, clear arrival/status, scatter, and validate arrivals.
      performance->weyl_routing_kernel_launches = 5;
      performance->weyl_logical_workspace_bytes =
          state.logical_surfaces->local_point_count() *
          (12 * static_cast<std::int64_t>(sizeof(amrex::Real)) +
           static_cast<std::int64_t>(sizeof(int)));
      performance->weyl_device_payload_bytes =
          routing.device_payload_bytes_sent +
          routing.device_payload_bytes_received +
          routing.device_local_copy_bytes;
      performance->weyl_routing_seconds = amrex::second() - routing_start;
    }
  }
  if (options.compute_weinberg_fluxes || compute_ricci_scalars) {
    if (state.interpolation_order != 3)
      amrex::Abort("ParticleAHFinderX four-dimensional analysis requires "
                   "cubic "
                   "finder interpolation");
    if (state.analysis_active_point_count <= 0 ||
        !state.analysis_route_plan.valid ||
        state.analysis_route_plan.active_point_count !=
            state.analysis_active_point_count)
      amrex::Abort("ParticleAHFinderX four-dimensional analysis has no final "
                   "target-layer sample route");
    if (!state.analysis_gauge_fields_resolved) {
      try {
        for (std::size_t field = 0; field < gauge_variable_names.size();
             ++field)
          state.analysis_gauge_fields[field] =
              resolve_mesh_field(gauge_variable_names[field]);
      } catch (const std::exception &error) {
        amrex::Abort(error.what());
      }
      state.analysis_gauge_fields_resolved = true;
    }
    try {
      ensure_mesh_fields_ready(
          cctkGH, std::vector<MeshFieldRef>(
                      state.analysis_gauge_fields.begin(),
                      state.analysis_gauge_fields.end()));
    } catch (const std::exception &error) {
      amrex::Abort(error.what());
    }

    amrex::Long local_population = 0;
    std::vector<amrex::Long> container_offsets(state.containers.size() + 1,
                                                0);
    for (std::size_t patch = 0; patch < state.containers.size(); ++patch) {
      local_population += state.containers[patch]->local_particle_count();
      container_offsets[patch + 1] = local_population;
    }
    const amrex::Real gather_start = amrex::second();
    state.analysis_gauge_mesh_device.resize(
        static_cast<std::size_t>(
            std::max<amrex::Long>(amrex::Long(1), local_population)));
    auto *const mesh_gauge = state.analysis_gauge_mesh_device.dataPtr();
    amrex::ParallelFor(
        local_population,
        [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
          mesh_gauge[point] = MeshGaugePoint{};
        });

    int gathered_levels = 0;
    if (!CarpetX::active_levels)
      amrex::Abort("ParticleAHFinderX four-dimensional analysis has no active "
                   "CarpetX levels");
    CarpetX::active_levels->loop_serially([&](auto &level) {
      GaugeFieldLayout fields;
      try {
        fields = gauge_field_layout(level, state.analysis_gauge_fields);
      } catch (const std::exception &error) {
        amrex::Abort(error.what());
      }
      if (level.patch < 0 ||
          level.patch >= static_cast<int>(state.containers.size()))
        amrex::Abort("ParticleAHFinderX four-dimensional analysis received an "
                     "invalid CarpetX patch");
      state.containers[level.patch]->gather_gauge_level(
          level.level, *fields.lapse, fields.lapse_component, *fields.shift,
          fields.shift_component, *fields.dt_lapse,
          fields.dt_lapse_component, *fields.dt_shift,
          fields.dt_shift_component, fields.centering, mesh_gauge,
          container_offsets[level.patch]);
      ++gathered_levels;
    });
    if (gathered_levels == 0)
      amrex::Abort("ParticleAHFinderX four-dimensional analysis found no active "
                   "level");

    std::array<amrex::Long, 4> gather_stats{{0, 0, 0, 0}};
    for (std::size_t patch = 0; patch < state.containers.size(); ++patch) {
      const auto stats = state.containers[patch]->local_gauge_gather_stats(
          mesh_gauge, container_offsets[patch]);
      gather_stats[0] += stats.valid;
      gather_stats[1] += stats.invalid_adm;
      gather_stats[2] += stats.stencil_unavailable;
      gather_stats[3] += stats.nonfinite;
    }
    amrex::ParallelDescriptor::ReduceLongSum(gather_stats.data(),
                                              int(gather_stats.size()));
    const amrex::Long classified =
        gather_stats[0] + gather_stats[1] + gather_stats[2] + gather_stats[3];
    if (classified != state.analysis_active_point_count ||
        gather_stats[0] != state.analysis_active_point_count ||
        gather_stats[1] != 0 || gather_stats[2] != 0 ||
        gather_stats[3] != 0)
      amrex::Abort("ParticleAHFinderX optional gauge gather did not cover "
                   "the final target-layer particle route");
    if (performance) {
      performance->gauge_gather_kernel_launches =
          1 + gathered_levels + static_cast<int>(state.containers.size());
      performance->gauge_mesh_workspace_bytes =
          local_population * sizeof(MeshGaugePoint);
      performance->gauge_gather_seconds = amrex::second() - gather_start;
    }

    const amrex::Real routing_start = amrex::second();
    const auto routing = route_gauge_to_logical_owners(
        mesh_gauge, *state.logical_surfaces, state.analysis_route_plan,
        state.analysis_active_point_count, local_population);
    std::array<amrex::Long, 5> global_routing{{
        routing.sent, routing.received, routing.missing, routing.duplicate,
        routing.invalid}};
    amrex::ParallelDescriptor::ReduceLongSum(global_routing.data(),
                                              int(global_routing.size()));
    if (global_routing[0] != state.analysis_active_point_count ||
        global_routing[1] != state.analysis_active_point_count ||
        global_routing[2] != 0 || global_routing[3] != 0 ||
        global_routing[4] != 0)
      amrex::Abort("ParticleAHFinderX optional gauge route is incomplete");
    if (performance) {
      performance->gauge_routing_kernel_launches = 5;
      performance->gauge_logical_workspace_bytes =
          state.logical_surfaces->local_point_count() *
          (FourGaugeLayout::component_count *
               static_cast<std::int64_t>(sizeof(amrex::Real)) +
           static_cast<std::int64_t>(sizeof(int)));
      performance->gauge_device_payload_bytes =
          routing.device_payload_bytes_sent +
          routing.device_payload_bytes_received +
          routing.device_local_copy_bytes;
      performance->gauge_routing_seconds = amrex::second() - routing_start;
    }
  }
  if (use_matter_ricci) {
    if (!state.analysis_matter_fields_resolved) {
      try {
        for (std::size_t field = 0; field < matter_variable_names.size();
             ++field)
          state.analysis_matter_fields[field] =
              resolve_mesh_field(matter_variable_names[field]);
      } catch (const std::exception &error) {
        amrex::Abort(error.what());
      }
      state.analysis_matter_fields_resolved = true;
    }
    try {
      ensure_mesh_fields_ready(
          cctkGH, std::vector<MeshFieldRef>(
                      state.analysis_matter_fields.begin(),
                      state.analysis_matter_fields.end()));
    } catch (const std::exception &error) {
      amrex::Abort(error.what());
    }

    amrex::Long local_population = 0;
    std::vector<amrex::Long> container_offsets(state.containers.size() + 1,
                                                0);
    for (std::size_t patch = 0; patch < state.containers.size(); ++patch) {
      local_population += state.containers[patch]->local_particle_count();
      container_offsets[patch + 1] = local_population;
    }
    const amrex::Real gather_start = amrex::second();
    state.analysis_matter_mesh_device.resize(
        static_cast<std::size_t>(
            std::max<amrex::Long>(amrex::Long(1), local_population)));
    auto *const mesh_matter = state.analysis_matter_mesh_device.dataPtr();
    amrex::ParallelFor(
        local_population,
        [=] AMREX_GPU_DEVICE(const amrex::Long point) noexcept {
          mesh_matter[point] = MeshMatterPoint{};
        });

    int gathered_levels = 0;
    if (!CarpetX::active_levels)
      amrex::Abort("ParticleAHFinderX matter analysis has no active CarpetX "
                   "levels");
    CarpetX::active_levels->loop_serially([&](auto &level) {
      MatterFieldLayout fields;
      try {
        fields = matter_field_layout(level, state.analysis_matter_fields);
      } catch (const std::exception &error) {
        amrex::Abort(error.what());
      }
      if (level.patch < 0 ||
          level.patch >= static_cast<int>(state.containers.size()))
        amrex::Abort("ParticleAHFinderX matter analysis received an invalid "
                     "CarpetX patch");
      state.containers[level.patch]->gather_matter_level(
          level.level, *fields.tt, fields.tt_component, *fields.ti,
          fields.ti_component, *fields.ij, fields.ij_component,
          fields.centering, mesh_matter, container_offsets[level.patch]);
      ++gathered_levels;
    });
    if (gathered_levels == 0)
      amrex::Abort("ParticleAHFinderX matter analysis found no active level");

    std::array<amrex::Long, 4> gather_stats{{0, 0, 0, 0}};
    for (std::size_t patch = 0; patch < state.containers.size(); ++patch) {
      const auto stats = state.containers[patch]->local_matter_gather_stats(
          mesh_matter, container_offsets[patch]);
      gather_stats[0] += stats.valid;
      gather_stats[1] += stats.invalid_adm;
      gather_stats[2] += stats.stencil_unavailable;
      gather_stats[3] += stats.nonfinite;
    }
    amrex::ParallelDescriptor::ReduceLongSum(gather_stats.data(),
                                              int(gather_stats.size()));
    const amrex::Long classified =
        gather_stats[0] + gather_stats[1] + gather_stats[2] + gather_stats[3];
    if (classified != state.analysis_active_point_count ||
        gather_stats[0] != state.analysis_active_point_count ||
        gather_stats[1] != 0 || gather_stats[2] != 0 ||
        gather_stats[3] != 0)
      amrex::Abort("ParticleAHFinderX optional matter gather did not cover "
                   "the final target-layer particle route");
    if (performance) {
      performance->matter_gather_kernel_launches =
          1 + gathered_levels + static_cast<int>(state.containers.size());
      performance->matter_mesh_workspace_bytes =
          local_population * sizeof(MeshMatterPoint);
      performance->matter_gather_seconds = amrex::second() - gather_start;
    }

    const amrex::Real routing_start = amrex::second();
    const auto routing = route_matter_to_logical_owners(
        mesh_matter, *state.logical_surfaces, state.analysis_route_plan,
        state.analysis_active_point_count, local_population);
    std::array<amrex::Long, 5> global_routing{{
        routing.sent, routing.received, routing.missing, routing.duplicate,
        routing.invalid}};
    amrex::ParallelDescriptor::ReduceLongSum(global_routing.data(),
                                              int(global_routing.size()));
    if (global_routing[0] != state.analysis_active_point_count ||
        global_routing[1] != state.analysis_active_point_count ||
        global_routing[2] != 0 || global_routing[3] != 0 ||
        global_routing[4] != 0)
      amrex::Abort("ParticleAHFinderX optional matter route is incomplete");
    if (performance) {
      performance->matter_routing_kernel_launches = 5;
      performance->matter_logical_workspace_bytes =
          state.logical_surfaces->local_point_count() *
          (FourMatterLayout::component_count *
               static_cast<std::int64_t>(sizeof(amrex::Real)) +
           static_cast<std::int64_t>(sizeof(int)));
      performance->matter_device_payload_bytes =
          routing.device_payload_bytes_sent +
          routing.device_payload_bytes_received +
          routing.device_local_copy_bytes;
      performance->matter_routing_seconds = amrex::second() - routing_start;
    }
  }
  const amrex::Real kernel_start = amrex::second();
  AxialSymmetryControl axial_control;
  axial_control.enabled = options.compute_invariant_spin;
  axial_control.compute_multipoles = options.compute_multipoles;
  axial_control.harmonic_degree = options.axial_harmonic_degree;
  axial_control.maximum_jacobi_sweeps =
      options.axial_maximum_jacobi_sweeps;
  axial_control.eigen_tolerance = options.axial_eigen_tolerance;
  axial_control.minimum_axis_separation =
      options.axial_minimum_axis_separation;
  state.logical_surfaces->prepare_surface_analysis(
      state.analysis_surface_ids_device.dataPtr(), surface_count,
      state.analysis_reals_device.dataPtr(),
      state.analysis_longs_device.dataPtr(), axial_control,
      options.compute_coordinate_fluxes, options.compute_weyl_curvature,
      options.compute_weinberg_fluxes, compute_ricci_scalars,
      use_matter_ricci);

  // The compact buffers are produced asynchronously on the AMReX stream. MPI
  // may not be stream-aware, so make their readiness explicit before handing
  // device pointers to the portable GPU-aware collective.
  amrex::Gpu::streamSynchronize();
  if (performance) {
    const auto local_points =
        state.logical_surfaces->local_point_count();
    const int local_surfaces =
        state.logical_surfaces->local_surface_count();
    const int axial_modes =
        options.compute_invariant_spin
            ? axial_basis_count(options.axial_harmonic_degree)
            : 0;
    performance->local_surface_count = local_surfaces;
    performance->local_angular_points = local_points;
    performance->intrinsic_geometry_kernel_launches =
        local_points > 0 ? 3 : 0;
    performance->axial_symmetry_kernel_launches =
        local_surfaces > 0
            ? (options.compute_invariant_spin ? 11 : 1)
            : 0;
    performance->axial_trial_modes = axial_modes;
    performance->multipole_maximum_degree =
        options.compute_multipoles ? surface_multipole_count - 1 : 0;
    performance->intrinsic_workspace_bytes =
        local_points *
        (12 * static_cast<std::int64_t>(sizeof(amrex::Real)) +
         2 * static_cast<std::int64_t>(sizeof(int)));
    performance->axial_point_workspace_bytes =
        local_points * (8 * axial_modes + 3) *
        static_cast<std::int64_t>(sizeof(amrex::Real));
    performance->axial_projected_workspace_bytes =
        static_cast<std::int64_t>(local_surfaces) *
        (static_cast<std::int64_t>(5) * axial_modes * axial_modes +
         axial_modes + axial_surface_real_components) *
            static_cast<std::int64_t>(sizeof(amrex::Real)) +
        static_cast<std::int64_t>(2) * local_surfaces *
            static_cast<std::int64_t>(sizeof(int));
    performance->device_kernel_launches =
        performance->weyl_gather_kernel_launches +
        performance->weyl_routing_kernel_launches +
        performance->gauge_gather_kernel_launches +
        performance->gauge_routing_kernel_launches +
        performance->matter_gather_kernel_launches +
        performance->matter_routing_kernel_launches + 2 +
        (local_surfaces > 0
             ? (options.compute_invariant_spin ? 17 : 7)
             : 0);
    performance->explicit_device_synchronizations = 1;
    performance->device_kernel_seconds = amrex::second() - kernel_start;
  }
  const auto communicator = amrex::ParallelDescriptor::Communicator();
  const amrex::Real collective_start = amrex::second();
  amrex::ParallelAllReduce::Sum(state.analysis_reals_device, communicator);
  amrex::ParallelAllReduce::Sum(state.analysis_longs_device, communicator);
  if (performance) {
    performance->device_collectives = 2;
    performance->compact_collective_bytes = static_cast<std::int64_t>(
        state.analysis_reals_device.size() * sizeof(amrex::Real) +
        state.analysis_longs_device.size() * sizeof(amrex::Long));
#ifdef AMREX_USE_GPU
    if (!amrex::ParallelDescriptor::UseGpuAwareMpi()) {
      // AMReX's DeviceVector collective stages each compact buffer once in
      // each direction through pinned memory.  Account for that portable
      // fallback explicitly; it remains O(number of surfaces), never
      // O(number of angular points).
      performance->compact_collective_host_staging_copies = 4;
      performance->compact_collective_host_staging_bytes =
          2 * performance->compact_collective_bytes;
    }
#endif
    performance->collective_seconds = amrex::second() - collective_start;
  }

  amrex::Gpu::PinnedVector<amrex::Real> real_values(
      state.analysis_reals_device.size());
  amrex::Gpu::PinnedVector<amrex::Long> long_values(
      state.analysis_longs_device.size());
  const amrex::Real transfer_start = amrex::second();
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   state.analysis_reals_device.begin(),
                   state.analysis_reals_device.end(), real_values.begin());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   state.analysis_longs_device.begin(),
                   state.analysis_longs_device.end(), long_values.begin());
  if (performance) {
    performance->device_to_host_copies = 2;
    performance->compact_device_to_host_bytes = static_cast<std::int64_t>(
        real_values.size() * sizeof(amrex::Real) +
        long_values.size() * sizeof(amrex::Long));
    performance->compact_device_to_host_seconds =
        amrex::second() - transfer_start;
  }
  if (long_values[long_components * surface_count] != 0)
    amrex::Abort("ParticleAHFinderX analysis could not map an owner-local "
                 "surface into the global stable-ID table");

  std::vector<SurfaceAnalysisRecord> records;
  records.reserve(surface_ids.size());
  const amrex::Real record_start = amrex::second();
  for (std::size_t surface = 0; surface < state.surfaces.size(); ++surface) {
    auto &source = state.surfaces[surface];
    if (source.last_search_iteration != iteration ||
        source.last_analysis_api_iteration == iteration ||
        source.relaxation_state != SurfaceRelaxationState::converged ||
        source.lifecycle == ParticleLifecycle::retired ||
        (published_only && !source.published))
      continue;
    const auto compact = std::lower_bound(surface_ids.begin(),
                                          surface_ids.end(),
                                          source.stable_id);
    if (compact == surface_ids.end() || *compact != source.stable_id)
      amrex::Abort("ParticleAHFinderX analysis lost an unconsumed searched "
                   "surface ID");
    const int compact_surface =
        static_cast<int>(compact - surface_ids.begin());
    const int real_offset = real_components * compact_surface;
    const int long_offset = long_components * compact_surface;
    // CarpetX may traverse more than one synchronized active-level band at
    // the same Cactus iteration. A surface found by an earlier traversal is
    // no longer in the logical storage's current due set and was already
    // offered to the consumer at that point.
    if (long_values[long_offset] == 0)
      continue;
    if (long_values[long_offset] != 1)
      amrex::Abort("ParticleAHFinderX analysis expected exactly one logical "
                   "owner for a converged target layer");
    const amrex::Real area = real_values[real_offset];
    const amrex::Long invalid =
        long_values[long_offset + 1];
    const amrex::Long intrinsic_invalid =
        long_values[long_offset + 2];
    const amrex::Long axial_status_value =
        long_values[long_offset + 3];
    const amrex::Long axial_iterations_value =
        long_values[long_offset + 4];
    const amrex::Long multipole_invalid =
        long_values[long_offset + 5];
    const amrex::Long coordinate_flux_invalid =
        long_values[long_offset + 6];
    const amrex::Long weyl_invalid =
        long_values[long_offset + 7];
    const amrex::Long speciality_index_invalid =
        long_values[long_offset + 8];
    const amrex::Long speciality_difference_invalid =
        long_values[long_offset + 9];
    const amrex::Long weinberg_invalid =
        long_values[long_offset + 10];
    const amrex::Long ricci_invalid =
        long_values[long_offset + 11];
    if (axial_status_value < std::numeric_limits<int>::min() ||
        axial_status_value > std::numeric_limits<int>::max() ||
        axial_iterations_value < 0 ||
        axial_iterations_value > std::numeric_limits<int>::max() ||
        multipole_invalid < 0 || coordinate_flux_invalid < 0 ||
        weyl_invalid < 0 || speciality_index_invalid < 0 ||
        speciality_difference_invalid < 0 || weinberg_invalid < 0 ||
        ricci_invalid < 0)
      amrex::Abort("ParticleAHFinderX analysis returned invalid compact "
                   "integer metadata");
    const int axial_status = static_cast<int>(axial_status_value);
    const int axial_iterations =
        static_cast<int>(axial_iterations_value);
    if (invalid != 0 || source.invalid_points != 0 || !(area > 0) ||
        !std::isfinite(area))
      amrex::Abort("ParticleAHFinderX analysis received invalid converged "
                   "surface geometry");
    for (int component = 1; component < real_components; ++component)
      if (!std::isfinite(
              real_values[real_offset + component]))
        amrex::Abort("ParticleAHFinderX analysis produced a non-finite "
                     "surface integral");
    const amrex::Real area_scale =
        amrex::max(amrex::Real(1), amrex::max(area, source.area));
    const amrex::Real area_roundoff =
        amrex::Real(32) * amrex::Real(source.ntheta) *
        amrex::Real(source.nphi) *
        std::numeric_limits<amrex::Real>::epsilon();
    const amrex::Real area_tolerance =
        amrex::max(area_roundoff, amrex::Real(1.0e-12)) *
        area_scale;
    if (!(source.area > 0) || std::abs(area - source.area) > area_tolerance)
      amrex::Abort("ParticleAHFinderX analysis area does not match the "
                   "finder's final target-layer diagnostics");

    SurfaceAnalysisRecord record;
    record.stable_id = static_cast<std::int64_t>(source.stable_id);
    for (int parent = 0; parent < 2; ++parent)
      record.parent_ids[parent] =
          static_cast<std::int64_t>(source.parent_ids[parent]);
    record.generation = source.generation;
    record.lifecycle = static_cast<int>(source.lifecycle);
    record.role = static_cast<int>(source.role);
    record.ntheta = source.ntheta;
    record.nphi = source.nphi;
    record.iteration = iteration;
    record.published = source.published;
    record.time = source.last_search_time;
    record.area = area;
    record.areal_radius =
        std::sqrt(area / (amrex::Real(4) *
                          amrex::Math::pi<amrex::Real>()));
    record.irreducible_mass = record.areal_radius / amrex::Real(2);
    for (int d = 0; d < 3; ++d) {
      if (!std::isfinite(source.center[d]) ||
          !std::isfinite(source.centroid[d]))
        amrex::Abort("ParticleAHFinderX analysis received a non-finite "
                     "surface center");
      record.center[d] = source.center[d];
      record.centroid[d] = source.centroid[d];
      record.momentum_covariant[d] =
          real_values[real_offset + 1 + d];
      record.proper_circumference[d] = source.proper_circumference[d];
    }

    const amrex::Real displacement[3]{
        source.centroid[0] - source.center[0],
        source.centroid[1] - source.center[1],
        source.centroid[2] - source.center[2]};
    const amrex::Real curvature_flux[3]{
        real_values[real_offset + 4], real_values[real_offset + 5],
        real_values[real_offset + 6]};
    const amrex::Real shift[3]{
        displacement[1] * curvature_flux[2] -
            displacement[2] * curvature_flux[1],
        displacement[2] * curvature_flux[0] -
            displacement[0] * curvature_flux[2],
        displacement[0] * curvature_flux[1] -
            displacement[1] * curvature_flux[0]};
    amrex::Real spin_squared = 0;
    for (int d = 0; d < 3; ++d) {
      record.coordinate_spin_center[d] =
          real_values[real_offset + 7 + d];
      record.coordinate_spin_centroid[d] =
          record.coordinate_spin_center[d] - shift[d];
      spin_squared +=
          record.coordinate_spin_centroid[d] *
          record.coordinate_spin_centroid[d];
    }
    record.coordinate_spin_magnitude = std::sqrt(spin_squared);
    const amrex::Real irreducible_mass_squared =
        record.irreducible_mass * record.irreducible_mass;
    record.christodoulou_mass = std::sqrt(
        irreducible_mass_squared +
        spin_squared /
            (amrex::Real(4) * irreducible_mass_squared));
    record.dimensionless_spin =
        record.coordinate_spin_magnitude /
        (record.christodoulou_mass * record.christodoulou_mass);
    record.mean_expansion = source.mean_expansion;
    record.expansion_l2 = source.expansion_l2;
    record.expansion_linf = source.expansion_linf;
    record.invalid_points = static_cast<std::int64_t>(invalid);
    record.intrinsic_invalid_points =
        static_cast<std::int64_t>(intrinsic_invalid);
    if (intrinsic_invalid == 0) {
      record.intrinsic_scalar_curvature_integral =
          real_values[real_offset + 10];
      const amrex::Real normal_connection_squared_integral =
          real_values[real_offset + 11];
      if (!std::isfinite(record.intrinsic_scalar_curvature_integral) ||
          !(normal_connection_squared_integral >= 0) ||
          !std::isfinite(normal_connection_squared_integral))
        amrex::Abort("ParticleAHFinderX analysis produced invalid intrinsic "
                     "surface integrals");
      const amrex::Real eight_pi =
          amrex::Real(8) * amrex::Math::pi<amrex::Real>();
      record.gauss_bonnet_residual =
          (record.intrinsic_scalar_curvature_integral - eight_pi) /
          eight_pi;
      record.normal_connection_l2 = std::sqrt(
          normal_connection_squared_integral / record.area);
      record.intrinsic_geometry_valid = true;
    }
    record.invariant_spin_requested = options.compute_invariant_spin;
    record.multipoles_requested = options.compute_multipoles;
    record.coordinate_fluxes_requested =
        options.compute_coordinate_fluxes;
    record.multipole_invalid_points =
        static_cast<std::int64_t>(multipole_invalid);
    record.coordinate_flux_invalid_points =
        static_cast<std::int64_t>(coordinate_flux_invalid);
    record.weyl_curvature_requested = options.compute_weyl_curvature;
    record.weyl_curvature_invalid_points =
        static_cast<std::int64_t>(weyl_invalid);
    record.speciality_index_invalid_points =
        static_cast<std::int64_t>(speciality_index_invalid);
    record.speciality_difference_invalid_points =
        static_cast<std::int64_t>(speciality_difference_invalid);
    record.weinberg_fluxes_requested = options.compute_weinberg_fluxes;
    record.weinberg_flux_invalid_points =
        static_cast<std::int64_t>(weinberg_invalid);
    record.ricci_source = options.ricci_source;
    record.ricci_invalid_points =
        static_cast<std::int64_t>(ricci_invalid);
    record.invariant_spin_status = axial_status;
    record.axial_iterations = axial_iterations;
    record.axial_harmonic_degree = options.axial_harmonic_degree;
    const int expected_axial_status = static_cast<int>(
        options.compute_invariant_spin ? AxialSymmetryStatus::valid
                                       : AxialSymmetryStatus::not_requested);
    if (!options.compute_invariant_spin &&
        axial_status != expected_axial_status)
      amrex::Abort("ParticleAHFinderX disabled axial analysis returned an "
                   "unexpected status");
    if (options.compute_invariant_spin &&
        (axial_status < static_cast<int>(AxialSymmetryStatus::valid) ||
         axial_status >
             static_cast<int>(AxialSymmetryStatus::invalid_spin_integral)))
      amrex::Abort("ParticleAHFinderX axial analysis returned an invalid "
                   "status code");
    if (options.compute_invariant_spin &&
        axial_status == static_cast<int>(AxialSymmetryStatus::valid)) {
      if (!record.intrinsic_geometry_valid)
        amrex::Abort("ParticleAHFinderX accepted an axial mode with invalid "
                     "intrinsic geometry");
      record.invariant_spin_signed =
          real_values[real_offset + 12];
      record.invariant_spin_magnitude =
          std::abs(record.invariant_spin_signed);
      amrex::Real axis_norm_squared = 0;
      for (int d = 0; d < 3; ++d) {
        record.invariant_spin_axis[d] =
            real_values[real_offset + 13 + d];
        axis_norm_squared += record.invariant_spin_axis[d] *
                             record.invariant_spin_axis[d];
      }
      record.axial_eigenvalue =
          real_values[real_offset + 16];
      record.axial_eigen_residual =
          real_values[real_offset + 17];
      record.axial_killing_residual =
          real_values[real_offset + 18];
      record.axial_normalization_residual =
          real_values[real_offset + 19];
      const amrex::Real axis_tolerance =
          amrex::Real(4096) *
          std::numeric_limits<amrex::Real>::epsilon();
      if (!std::isfinite(record.invariant_spin_signed) ||
          !std::isfinite(axis_norm_squared) ||
          std::abs(axis_norm_squared - 1) > axis_tolerance ||
          !std::isfinite(record.axial_eigenvalue) ||
          !std::isfinite(record.axial_eigen_residual) ||
          !std::isfinite(record.axial_killing_residual) ||
          !std::isfinite(record.axial_normalization_residual) ||
          record.axial_eigen_residual < 0 ||
          record.axial_killing_residual < 0 ||
          record.axial_normalization_residual < 0)
        amrex::Abort("ParticleAHFinderX accepted invalid axial-symmetry "
                     "diagnostics");
      record.invariant_christodoulou_mass = std::sqrt(
          irreducible_mass_squared +
          record.invariant_spin_signed * record.invariant_spin_signed /
              (amrex::Real(4) * irreducible_mass_squared));
      record.invariant_dimensionless_spin =
          record.invariant_spin_signed /
          (record.invariant_christodoulou_mass *
           record.invariant_christodoulou_mass);
      record.invariant_spin_valid = true;
      record.invariant_coordinate_mean =
          real_values[real_offset + 20] / record.area;
      record.invariant_coordinate_minimum =
          real_values[real_offset + 21];
      record.invariant_coordinate_maximum =
          real_values[real_offset + 22];
      if (!std::isfinite(record.invariant_coordinate_mean) ||
          !std::isfinite(record.invariant_coordinate_minimum) ||
          !std::isfinite(record.invariant_coordinate_maximum) ||
          !(record.invariant_coordinate_maximum >
            record.invariant_coordinate_minimum))
        amrex::Abort("ParticleAHFinderX produced an invalid invariant "
                     "coordinate");
      record.invariant_coordinate_valid = true;

      if (options.compute_multipoles && multipole_invalid == 0) {
        const amrex::Real two_pi =
            2 * amrex::Math::pi<amrex::Real>();
        const amrex::Real eight_pi =
            8 * amrex::Math::pi<amrex::Real>();
        amrex::Real radius_power = 1;
        amrex::Real spin_radius_power = 1 / record.areal_radius;
        for (int degree = 0; degree < surface_multipole_count; ++degree) {
          record.mass_multipoles[degree] =
              real_values[real_offset + 23 + degree] *
              record.invariant_christodoulou_mass * radius_power / two_pi;
          record.spin_multipoles[degree] =
              real_values[real_offset + 23 +
                          surface_multipole_count + degree] *
              spin_radius_power / eight_pi;
          if (!std::isfinite(record.mass_multipoles[degree]) ||
              !std::isfinite(record.spin_multipoles[degree]))
            amrex::Abort("ParticleAHFinderX produced a non-finite invariant "
                         "multipole");
          radius_power *= record.areal_radius;
          spin_radius_power *= record.areal_radius;
        }
        record.multipoles_valid = true;
      }
    }
    if (!options.compute_coordinate_fluxes && coordinate_flux_invalid != 0)
      amrex::Abort("ParticleAHFinderX disabled coordinate-flux analysis "
                   "returned invalid points");
    if (options.compute_coordinate_fluxes && coordinate_flux_invalid == 0) {
      const auto flux_value = [&](const int component) {
        return real_values[real_offset +
                           surface_analysis_coordinate_flux_offset +
                           component];
      };
      record.adm_energy = flux_value(CoordinateFluxLayout::adm_energy);
      record.conformal_adm_energy =
          flux_value(CoordinateFluxLayout::conformal_adm_energy);
      for (int d = 0; d < 3; ++d) {
        record.adm_momentum[d] =
            flux_value(CoordinateFluxLayout::adm_momentum + d);
        record.adm_angular_momentum[d] =
            flux_value(CoordinateFluxLayout::adm_angular_momentum + d);
        record.conformal_adm_momentum[d] =
            flux_value(CoordinateFluxLayout::conformal_adm_momentum + d);
        record.conformal_adm_angular_momentum[d] =
            flux_value(
                CoordinateFluxLayout::conformal_adm_angular_momentum + d);
      }
      record.coordinate_fluxes_valid = true;
    }
    if (!options.compute_weyl_curvature &&
        (weyl_invalid != 0 || speciality_index_invalid != 0 ||
         speciality_difference_invalid != 0))
      amrex::Abort("ParticleAHFinderX disabled Weyl analysis returned invalid "
                   "points");
    if (options.compute_weyl_curvature && weyl_invalid == 0) {
      const auto mean_complex = [&](const int component) {
        SurfaceComplexValue value;
        value.real =
            real_values[real_offset +
                        surface_analysis_weyl_offset + component] /
            record.area;
        value.imag =
            real_values[real_offset +
                        surface_analysis_weyl_offset + component + 1] /
            record.area;
        return value;
      };
      for (int scalar = 0; scalar < 5; ++scalar)
        record.mean_weyl_scalar[scalar] = mean_complex(2 * scalar);
      record.mean_weyl_invariant_i =
          mean_complex(WeylIntegralLayout::invariant_i);
      record.mean_weyl_invariant_j =
          mean_complex(WeylIntegralLayout::invariant_j);
      record.weyl_curvature_valid = true;
      if (speciality_index_invalid == 0) {
        record.mean_speciality_index =
            mean_complex(WeylIntegralLayout::speciality_index);
        record.speciality_index_valid = true;
      }
      if (speciality_difference_invalid == 0) {
        record.mean_speciality_difference =
            mean_complex(WeylIntegralLayout::speciality_difference);
        record.speciality_difference_valid = true;
      }
    }
    if (!options.compute_weinberg_fluxes && weinberg_invalid != 0)
      amrex::Abort("ParticleAHFinderX disabled Weinberg analysis returned "
                   "invalid points");
    if (options.compute_weinberg_fluxes && weinberg_invalid == 0) {
      const auto flux_value = [&](const int component) {
        return real_values[real_offset + surface_analysis_weinberg_offset +
                           component];
      };
      record.weinberg_energy = flux_value(0);
      for (int component = 0; component < 3; ++component) {
        record.weinberg_momentum[component] = flux_value(1 + component);
        record.weinberg_angular_momentum[component] =
            flux_value(4 + component);
      }
      record.weinberg_fluxes_valid = true;
    }
    if (!compute_ricci_scalars && ricci_invalid != 0)
      amrex::Abort("ParticleAHFinderX disabled Ricci analysis returned "
                   "invalid points");
    if (compute_ricci_scalars && ricci_invalid == 0) {
      const auto mean_value = [&](const int component) {
        return real_values[real_offset + surface_analysis_ricci_offset +
                           component] /
               record.area;
      };
      record.mean_ricci_phi00 =
          mean_value(RicciIntegralLayout::phi00);
      record.mean_ricci_phi01 = {
          mean_value(RicciIntegralLayout::phi01_real),
          mean_value(RicciIntegralLayout::phi01_imag)};
      record.mean_ricci_phi02 = {
          mean_value(RicciIntegralLayout::phi02_real),
          mean_value(RicciIntegralLayout::phi02_imag)};
      record.mean_ricci_phi11 =
          mean_value(RicciIntegralLayout::phi11);
      record.mean_ricci_phi12 = {
          mean_value(RicciIntegralLayout::phi12_real),
          mean_value(RicciIntegralLayout::phi12_imag)};
      record.mean_ricci_phi22 =
          mean_value(RicciIntegralLayout::phi22);
      record.mean_ricci_lambda =
          mean_value(RicciIntegralLayout::lambda);
      record.ricci_scalars_valid = true;
    }
    records.push_back(record);
    source.last_analysis_api_iteration = iteration;
  }
  if (performance) {
    performance->returned_record_count = static_cast<int>(records.size());
    performance->host_record_seconds = amrex::second() - record_start;
    performance->total_seconds = amrex::second() - total_start;
  }
  return records;
}

} // namespace ParticleAHFinderX
