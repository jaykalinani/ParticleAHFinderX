/**
 * \file diagnostics.cxx
 * \brief Restart-safe surface diagnostics from bounded reduction records.
 */
#include "runtime.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace ParticleAHFinderX {
namespace {

bool file_is_empty(const std::string &filename) {
  std::ifstream input(filename, std::ios::binary | std::ios::ate);
  return !input || input.tellg() == 0;
}

template <std::size_t N>
void write_tsv_columns(std::ostream &output,
                       const std::array<const char *, N> &column_names) {
  for (std::size_t column = 0; column < column_names.size(); ++column)
    output << "# column " << std::setw(2) << column + 1 << " = "
           << column_names[column] << '\n';
  output << '#';
  for (std::size_t column = 0; column < column_names.size(); ++column)
    output << ' ' << column + 1 << ':' << column_names[column];
  output << '\n';
}

void discard_stale_rows(const std::string &filename,
                        const int recovery_iteration) {
  std::ifstream input(filename);
  if (!input)
    return;
  const std::string temporary = filename + ".recover.tmp";
  std::ofstream output(temporary, std::ios::out | std::ios::trunc);
  if (!output)
    CCTK_VERROR("Could not create ParticleAHFinderX diagnostic recovery file "
                "'%s'",
                temporary.c_str());
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      output << line << '\n';
      continue;
    }
    std::istringstream row(line);
    int iteration = -1;
    if (!(row >> iteration))
      CCTK_VERROR("ParticleAHFinderX cannot parse a diagnostic row in '%s'",
                  filename.c_str());
    // CarpetX checkpoints after analysis, so the recovered iteration is
    // committed. Remove only rows from the abandoned later trajectory.
    if (iteration <= recovery_iteration)
      output << line << '\n';
  }
  output.close();
  input.close();
  if (!output || std::rename(temporary.c_str(), filename.c_str()) != 0)
    CCTK_VERROR("Could not reconcile ParticleAHFinderX diagnostic file '%s' "
                "after recovery",
                filename.c_str());
}

void reconcile_diagnostics_after_recovery(const Runtime &state,
                                          const std::string &directory,
                                          const bool compatibility) {
  discard_stale_rows(directory + "/surface_diagnostics.tsv",
                     state.recovered_iteration);
  discard_stale_rows(directory + "/candidate_events.tsv",
                     state.recovered_iteration);
  if (!compatibility)
    return;
  for (const auto &surface : state.surfaces) {
    if (surface.compatibility_slot <= 0)
      continue;
    std::ostringstream name;
    name << directory << "/BH_diagnostics.ah" << surface.compatibility_slot
         << ".gp";
    discard_stale_rows(name.str(), state.recovered_iteration);
  }
}

void write_candidate_header(std::ostream &output) {
  constexpr std::array<const char *, 26> column_names{{
      "iteration",
      "time",
      "stable_id",
      "generation",
      "event_type",
      "reason",
      "parent_1",
      "parent_2",
      "related_surface_id",
      "attempt",
      "center_x",
      "center_y",
      "center_z",
      "parent_separation",
      "trigger_distance",
      "expiry_distance",
      "radius",
      "min_radius",
      "max_radius",
      "area",
      "expansion_l2",
      "expansion_linf",
      "invalid_points",
      "relaxation_steps",
      "next_search_iteration",
      "particle_count",
  }};
  output << "# schema 1\n"
            "# One tab-delimited row per candidate decision; iteration and "
            "time are Cactus cctk_iteration and cctk_time.\n"
            "# Geometric units use G=c=1. A parent or related surface ID "
            "of 0 means none; a negative iteration means not scheduled.\n"
            "# event_type: 1=birth 2=attempt 3=promotion 4=retirement\n"
            "# reason: 1=created 2=solver_not_converged "
            "3=invalid_geometry 4=parent_containment_failed "
            "5=duplicate_published_surface 6=accepted "
            "7=attempt_limit 8=separation_expired "
            "9=parent_unavailable 10=outside_domain\n";
  write_tsv_columns(output, column_names);
}

void write_candidate_events(const Runtime &state,
                            const std::string &directory) {
  if (state.pending_candidate_events.empty())
    return;
  const std::string filename = directory + "/candidate_events.tsv";
  const bool write_header = file_is_empty(filename);
  std::ofstream output(filename, std::ios::out | std::ios::app);
  if (!output)
    CCTK_VERROR("Could not open ParticleAHFinderX candidate event file '%s'",
                filename.c_str());
  if (write_header)
    write_candidate_header(output);
  output << std::setprecision(17);
  for (const auto &event : state.pending_candidate_events) {
    output << event.iteration << '\t' << event.time << '\t'
           << event.stable_id << '\t' << event.generation << '\t'
           << static_cast<int>(event.type) << '\t'
           << static_cast<int>(event.reason) << '\t'
           << event.parent_ids[0] << '\t' << event.parent_ids[1] << '\t'
           << event.related_surface_id << '\t' << event.attempt;
    for (const auto value : event.center)
      output << '\t' << value;
    output << '\t' << event.parent_separation << '\t'
           << event.trigger_distance << '\t' << event.expiry_distance
           << '\t' << event.radius << '\t' << event.minimum_radius << '\t'
           << event.maximum_radius << '\t' << event.area << '\t'
           << event.expansion_l2 << '\t'
           << event.expansion_linf << '\t' << event.invalid_points << '\t'
           << event.relaxation_steps << '\t'
           << event.next_search_iteration << '\t'
           << event.particle_count << '\n';
  }
  if (!output)
    CCTK_VERROR("Could not write ParticleAHFinderX candidate event file '%s'",
                filename.c_str());
}

void write_native_header(std::ostream &output) {
  constexpr std::array<const char *, 46> column_names{{
      "iteration",
      "time",
      "stable_id",
      "generation",
      "lifecycle",
      "role",
      "relaxation_state",
      "parent_1",
      "parent_2",
      "center_x",
      "center_y",
      "center_z",
      "centroid_x",
      "centroid_y",
      "centroid_z",
      "min_radius",
      "max_radius",
      "mean_radius",
      "area",
      "irreducible_mass",
      "areal_radius",
      "mean_expansion",
      "expansion_l2",
      "expansion_linf",
      "relaxation_steps",
      "invalid_points",
      "published",
      "birth_iteration",
      "last_state_change_iteration",
      "candidate_attempts",
      "next_search_iteration",
      "promoted_iteration",
      "retire_after_iteration",
      "xy_circumference",
      "xz_circumference",
      "yz_circumference",
      "domain_exit_checks",
      "last_domain_exit_check_iteration",
      "retired_outside_domain",
      "minimum_sampled_level",
      "maximum_sampled_level",
      "crosses_refinement_levels",
      "sampled_level_changes",
      "cumulative_sampled_level_changes",
      "regrid_events",
      "last_regrid_iteration",
  }};
  output << "# schema 7\n"
            "# One tab-delimited row per searched or state-changing surface; "
            "iteration and time are Cactus cctk_iteration and cctk_time.\n"
            "# Geometric units use G=c=1. Boolean fields use 0=false, "
            "1=true. A parent ID of 0 means no parent; -1 means a level or "
            "iteration is not applicable.\n"
            "# lifecycle: 0=candidate 1=active 2=retired; role: "
            "0=individual 1=common 2=other; relaxation_state: "
            "0=idle 1=active 2=converged 3=refine 4=failed.\n"
            "# published=1 means the stable surface is exposed to consumers; "
            "it does not imply convergence. A valid AH measurement requires "
            "relaxation_state=2 and invalid_points=0.\n"
            "# mean_expansion is the mean outgoing expansion Theta_(l), "
            "matching AHFinderDirect column 29. irreducible_mass matches "
            "AHFinderDirect m_irreducible and QLM qlm_irreducible_mass.\n";
  write_tsv_columns(output, column_names);
}

void write_native_diagnostics(const cGH *const cctkGH,
                              const Runtime &state,
                              const std::string &directory) {
  const std::string filename = directory + "/surface_diagnostics.tsv";
  const bool write_header = file_is_empty(filename);
  std::ofstream output(filename, std::ios::out | std::ios::app);
  if (!output)
    CCTK_VERROR("Could not open ParticleAHFinderX diagnostics file '%s'",
                filename.c_str());
  if (write_header)
    write_native_header(output);
  output << std::setprecision(17);
  for (const auto &surface : state.surfaces) {
    if ((surface.last_search_iteration != cctkGH->cctk_iteration &&
         surface.last_state_change_iteration != cctkGH->cctk_iteration) ||
        surface.last_native_diagnostics_iteration ==
            cctkGH->cctk_iteration)
      continue;
    const amrex::Real irreducible_mass =
        surface.area > 0
            ? std::sqrt(surface.area /
                        (16 * amrex::Math::pi<amrex::Real>()))
            : 0;
    const amrex::Real areal_radius =
        surface.area > 0
            ? std::sqrt(surface.area /
                        (4 * amrex::Math::pi<amrex::Real>()))
            : 0;
    const amrex::Real record_time =
        surface.last_search_iteration == cctkGH->cctk_iteration
            ? surface.last_search_time
            : static_cast<amrex::Real>(cctkGH->cctk_time);
    output << cctkGH->cctk_iteration << '\t' << record_time << '\t'
           << surface.stable_id << '\t' << surface.generation << '\t'
           << static_cast<int>(surface.lifecycle) << '\t'
           << static_cast<int>(surface.role) << '\t'
           << static_cast<int>(surface.relaxation_state) << '\t'
           << surface.parent_ids[0] << '\t' << surface.parent_ids[1] << '\t'
           << surface.center[0] << '\t' << surface.center[1] << '\t'
           << surface.center[2] << '\t' << surface.centroid[0] << '\t'
           << surface.centroid[1] << '\t' << surface.centroid[2] << '\t'
           << surface.minimum_radius << '\t' << surface.maximum_radius << '\t'
           << surface.mean_radius << '\t' << surface.area << '\t'
           << irreducible_mass << '\t' << areal_radius << '\t'
           << surface.mean_expansion << '\t' << surface.expansion_l2 << '\t'
           << surface.expansion_linf << '\t' << surface.last_search_steps
           << '\t' << surface.invalid_points << '\t' << surface.published
           << '\t' << surface.birth_iteration << '\t'
           << surface.last_state_change_iteration << '\t'
           << surface.candidate_attempts << '\t'
           << surface.next_search_iteration << '\t'
           << surface.promoted_iteration << '\t'
           << surface.retire_after_iteration;
    for (const auto circumference : surface.proper_circumference)
      output << '\t' << circumference;
    output << '\t' << surface.consecutive_domain_exit_checks << '\t'
           << surface.last_domain_exit_check_iteration << '\t'
           << surface.retired_outside_domain << '\t'
           << surface.minimum_sampled_level << '\t'
           << surface.maximum_sampled_level << '\t'
           << (surface.minimum_sampled_level >= 0 &&
               surface.maximum_sampled_level >
                   surface.minimum_sampled_level)
           << '\t' << surface.sampled_level_changes << '\t'
           << surface.cumulative_sampled_level_changes << '\t'
           << state.regrid_events << '\t' << state.last_regrid_iteration;
    output << '\n';
  }
  if (!output)
    CCTK_VERROR("Could not write ParticleAHFinderX diagnostics file '%s'",
                filename.c_str());
}

void write_compatibility_header(std::ostream &output, const int slot,
                                const amrex::Long stable_id) {
  output << "# apparent horizon compatibility slot " << slot << '\n'
         << "# ParticleAHFinderX stable surface ID " << stable_id << '\n'
         << "# AHFinderDirect-compatible 40-column layout; geometric units "
            "use G=c=1.\n#\n";
  const char *const names[40]{
      "cctk_iteration",
      "cctk_time",
      "centroid_x",
      "centroid_y",
      "centroid_z",
      "min radius",
      "max radius",
      "mean radius",
      "quadrupole_xx",
      "quadrupole_xy",
      "quadrupole_xz",
      "quadrupole_yy",
      "quadrupole_yz",
      "quadrupole_zz",
      "min x",
      "max x",
      "min y",
      "max y",
      "min z",
      "max z",
      "xy-plane circumference",
      "xz-plane circumference",
      "yz-plane circumference",
      "ratio of xz/xy-plane circumferences",
      "ratio of yz/xy-plane circumferences",
      "area",
      "m_irreducible",
      "areal radius",
      "expansion Theta_(l)",
      "inner expansion Theta_(n)",
      "product of the expansions",
      "mean curvature",
      "gradient of the areal radius",
      "gradient of the expansion Theta_(l)",
      "gradient of the inner expansion Theta_(n)",
      "gradient of the product of the expansions",
      "gradient of the mean curvature",
      "minimum of the mean curvature",
      "maximum of the mean curvature",
      "integral of the mean curvature",
  };
  for (int column = 0; column < 40; ++column)
    output << "# column " << std::setw(2) << column + 1 << " = "
           << names[column] << '\n';
  output << "# Columns 30-40 are deliberately NaN, not zero, because their "
            "independent geometric operators are not yet implemented.\n";
}

void write_compatibility_files(const cGH *const cctkGH,
                               const Runtime &state,
                               const std::string &directory) {
  const amrex::Real nan =
      std::numeric_limits<amrex::Real>::quiet_NaN();
  for (const auto &surface : state.surfaces) {
    if (surface.compatibility_slot <= 0 ||
        surface.last_search_iteration != cctkGH->cctk_iteration ||
        surface.last_search_time !=
            static_cast<amrex::Real>(cctkGH->cctk_time) ||
        surface.last_compatibility_diagnostics_iteration ==
            cctkGH->cctk_iteration ||
        surface.relaxation_state != SurfaceRelaxationState::converged ||
        surface.invalid_points != 0 || !(surface.area > 0))
      continue;
    std::ostringstream name;
    name << directory << "/BH_diagnostics.ah"
         << surface.compatibility_slot << ".gp";
    const bool write_header = file_is_empty(name.str());
    std::ofstream output(name.str(), std::ios::out | std::ios::app);
    if (!output)
      CCTK_VERROR("Could not open compatibility diagnostics file '%s'",
                  name.str().c_str());
    if (write_header)
      write_compatibility_header(output, surface.compatibility_slot,
                                 surface.stable_id);
    const amrex::Real irreducible_mass =
        std::sqrt(surface.area /
                  (16 * amrex::Math::pi<amrex::Real>()));
    const amrex::Real areal_radius =
        std::sqrt(surface.area /
                  (4 * amrex::Math::pi<amrex::Real>()));
    output << std::setprecision(17) << cctkGH->cctk_iteration << '\t'
           << surface.last_search_time << '\t' << surface.centroid[0] << '\t'
           << surface.centroid[1] << '\t' << surface.centroid[2] << '\t'
           << surface.minimum_radius << '\t' << surface.maximum_radius << '\t'
           << surface.mean_radius;
    for (const auto value : surface.coordinate_quadrupole)
      output << '\t' << value;
    output << '\t' << surface.position_minimum[0] << '\t'
           << surface.position_maximum[0] << '\t'
           << surface.position_minimum[1] << '\t'
           << surface.position_maximum[1] << '\t'
           << surface.position_minimum[2] << '\t'
           << surface.position_maximum[2];
    for (const auto circumference : surface.proper_circumference)
      output << '\t' << circumference;
    output << '\t'
           << surface.proper_circumference[1] /
                  surface.proper_circumference[0]
           << '\t'
           << surface.proper_circumference[2] /
                  surface.proper_circumference[0];
    output << '\t' << surface.area << '\t' << irreducible_mass << '\t'
           << areal_radius << '\t' << surface.mean_expansion;
    for (int column = 30; column <= 40; ++column)
      output << '\t' << nan;
    output << '\n';
    if (!output)
      CCTK_VERROR("Could not write compatibility diagnostics file '%s'",
                  name.str().c_str());
  }
}

} // namespace

extern "C" void ParticleAHFinderX_Diagnostics(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  const bool diagnostics_due =
      diagnostics_every > 0 && cctk_iteration % diagnostics_every == 0;
  const bool has_candidate_events = !state.pending_candidate_events.empty();
  bool has_native_record = false;
  if (diagnostics_due)
    for (const auto &surface : state.surfaces)
      has_native_record |=
          (surface.last_search_iteration == cctk_iteration ||
           surface.last_state_change_iteration == cctk_iteration) &&
          surface.last_native_diagnostics_iteration != cctk_iteration;
  const bool synchronized = CarpetX::all_levels_synchronized();
  bool has_compatibility_record = false;
  if (diagnostics_due && output_compatibility_diagnostics && synchronized)
    for (const auto &surface : state.surfaces)
      has_compatibility_record |=
          surface.compatibility_slot > 0 &&
          surface.last_search_iteration == cctk_iteration &&
          surface.last_search_time == static_cast<amrex::Real>(cctk_time) &&
          surface.last_compatibility_diagnostics_iteration != cctk_iteration &&
          surface.relaxation_state == SurfaceRelaxationState::converged &&
          surface.invalid_points == 0 && surface.area > 0;
  if (!has_native_record && !has_compatibility_record &&
      !has_candidate_events)
    return;
  if (CCTK_CreateDirectory(0755, diagnostics_out_dir) < 0)
    CCTK_VERROR("Could not create ParticleAHFinderX diagnostics directory '%s'",
                diagnostics_out_dir);
  if (state.recovered && state.recovered_iteration >= 0 &&
      state.reconciled_diagnostics_recovery_iteration !=
          state.recovered_iteration) {
    if (amrex::ParallelDescriptor::IOProcessor())
      reconcile_diagnostics_after_recovery(
          state, diagnostics_out_dir, output_compatibility_diagnostics);
    amrex::ParallelDescriptor::Barrier();
    state.reconciled_diagnostics_recovery_iteration =
        state.recovered_iteration;
  }
  if (amrex::ParallelDescriptor::IOProcessor()) {
    if (has_candidate_events)
      write_candidate_events(state, diagnostics_out_dir);
    if (has_native_record)
      write_native_diagnostics(cctkGH, state, diagnostics_out_dir);
    if (has_compatibility_record)
      write_compatibility_files(cctkGH, state, diagnostics_out_dir);
  }
  if (has_candidate_events)
    amrex::ParallelDescriptor::Barrier();
  for (auto &surface : state.surfaces) {
    if (has_native_record &&
        (surface.last_search_iteration == cctk_iteration ||
         surface.last_state_change_iteration == cctk_iteration) &&
        surface.last_native_diagnostics_iteration != cctk_iteration)
      surface.last_native_diagnostics_iteration = cctk_iteration;
    if (has_compatibility_record && surface.compatibility_slot > 0 &&
        surface.last_search_iteration == cctk_iteration &&
        surface.last_search_time == static_cast<amrex::Real>(cctk_time) &&
        surface.last_compatibility_diagnostics_iteration != cctk_iteration &&
        surface.relaxation_state == SurfaceRelaxationState::converged &&
        surface.invalid_points == 0 && surface.area > 0)
      surface.last_compatibility_diagnostics_iteration = cctk_iteration;
  }
  if (has_candidate_events)
    state.pending_candidate_events.clear();
  if (has_native_record || has_compatibility_record)
    state.last_diagnostics_iteration = cctk_iteration;
}

} // namespace ParticleAHFinderX
