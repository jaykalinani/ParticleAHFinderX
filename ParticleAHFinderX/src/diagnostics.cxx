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
    if (iteration < recovery_iteration)
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

void write_native_header(std::ostream &output) {
  output << "# schema 7\n"
            "# 1:iteration 2:time 3:stable_id 4:generation "
            "5:lifecycle 6:role 7:relaxation_state 8:parent_1 "
            "9:parent_2 10:center_x 11:center_y 12:center_z "
            "13:centroid_x 14:centroid_y 15:centroid_z "
            "16:min_radius 17:max_radius 18:mean_radius 19:area "
            "20:irreducible_mass 21:areal_radius 22:mean_expansion "
            "23:expansion_l2 24:expansion_linf 25:relaxation_steps "
            "26:invalid_points 27:published 28:birth_iteration "
            "29:last_state_change_iteration 30:candidate_attempts "
            "31:next_search_iteration 32:promoted_iteration "
            "33:retire_after_iteration 34:xy_circumference "
            "35:xz_circumference 36:yz_circumference "
            "37:domain_exit_checks 38:last_domain_exit_check_iteration "
            "39:retired_outside_domain 40:minimum_sampled_level "
            "41:maximum_sampled_level 42:crosses_refinement_levels "
            "43:sampled_level_changes "
            "44:cumulative_sampled_level_changes 45:regrid_events "
            "46:last_regrid_iteration\n";
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
    if (surface.last_search_iteration != cctkGH->cctk_iteration &&
        surface.last_state_change_iteration != cctkGH->cctk_iteration)
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
    output << cctkGH->cctk_iteration << '\t' << cctkGH->cctk_time << '\t'
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

void write_compatibility_header(std::ostream &output, const int slot) {
  output << "# apparent horizon compatibility slot " << slot << '\n';
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
  output << "# Columns 30-40 are NaN until their independent "
            "geometric operators are enabled.\n";
}

void write_compatibility_files(const cGH *const cctkGH,
                               const Runtime &state,
                               const std::string &directory) {
  const amrex::Real nan =
      std::numeric_limits<amrex::Real>::quiet_NaN();
  for (const auto &surface : state.surfaces) {
    if (surface.compatibility_slot <= 0 ||
        surface.last_search_iteration != cctkGH->cctk_iteration ||
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
      write_compatibility_header(output, surface.compatibility_slot);
    const amrex::Real irreducible_mass =
        std::sqrt(surface.area /
                  (16 * amrex::Math::pi<amrex::Real>()));
    const amrex::Real areal_radius =
        std::sqrt(surface.area /
                  (4 * amrex::Math::pi<amrex::Real>()));
    output << std::setprecision(17) << cctkGH->cctk_iteration << '\t'
           << cctkGH->cctk_time << '\t' << surface.centroid[0] << '\t'
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
  if (diagnostics_every <= 0 ||
      cctk_iteration % diagnostics_every != 0 ||
      state.last_diagnostics_iteration == cctk_iteration)
    return;
  if (CCTK_CreateDirectory(0755, diagnostics_out_dir) < 0)
    CCTK_VERROR("Could not create ParticleAHFinderX diagnostics directory '%s'",
                diagnostics_out_dir);
  if (state.recovered && state.recovered_iteration >= 0 &&
      state.last_diagnostics_iteration < state.recovered_iteration) {
    if (amrex::ParallelDescriptor::IOProcessor())
      reconcile_diagnostics_after_recovery(
          state, diagnostics_out_dir, output_compatibility_diagnostics);
    amrex::ParallelDescriptor::Barrier();
  }
  bool has_current_record = false;
  for (const auto &surface : state.surfaces)
    has_current_record |= surface.last_search_iteration == cctk_iteration ||
                          surface.last_state_change_iteration == cctk_iteration;
  if (!has_current_record)
    return;
  if (amrex::ParallelDescriptor::IOProcessor()) {
    write_native_diagnostics(cctkGH, state, diagnostics_out_dir);
    if (output_compatibility_diagnostics)
      write_compatibility_files(cctkGH, state, diagnostics_out_dir);
  }
  state.last_diagnostics_iteration = cctk_iteration;
}

} // namespace ParticleAHFinderX
