/**
 * \file analysis.cxx
 * \brief Dynamic stable-ID output for GPU-native horizon integrals.
 */
#include <particleahfinderx_analysis.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace AHAnalysis {
namespace {

struct Runtime {
  std::map<std::int64_t, int> last_output_iteration;
  std::map<std::int64_t, int> last_weinberg_output_iteration;
  std::map<std::int64_t, int> last_ricci_output_iteration;
  int reconciled_recovery_iteration = -1;
  bool loaded_output = false;
  bool loaded_weinberg_output = false;
  bool loaded_ricci_output = false;
  bool reported_not_ready = false;
  bool reported_transport = false;
};

Runtime &runtime() {
  static Runtime state;
  return state;
}

bool file_is_empty(const std::string &filename) {
  std::ifstream input(filename, std::ios::binary | std::ios::ate);
  return !input || input.tellg() == 0;
}

struct OutputRowKey {
  int iteration = -1;
  std::int64_t stable_id = 0;
};

ParticleAHFinderX::SurfaceRicciSource
selected_ricci_source(const char *const source) {
  if (CCTK_EQUALS(source, "vacuum"))
    return ParticleAHFinderX::SurfaceRicciSource::vacuum;
  if (CCTK_EQUALS(source, "tmunu"))
    return ParticleAHFinderX::SurfaceRicciSource::tmunu;
  return ParticleAHFinderX::SurfaceRicciSource::off;
}

OutputRowKey validated_output_row(const std::string &line,
                                  const std::string &filename,
                                  const std::int64_t line_number) {
  constexpr int column_count = 128;
  constexpr std::array<int, 33> integer_columns{
      0,  2,  3,  4,  5,  6,  7,  8,  9,
      10, 38, 39, 43, 44, 45, 46, 58, 59,
      60, 64, 65, 66, 67, 86, 87, 88, 103,
      104, 105, 120, 121, 124, 125};
  std::array<std::string, column_count> fields;
  std::istringstream row(line);
  for (auto &field : fields)
    if (!(row >> field))
      CCTK_VERROR("AH_analysis output '%s' line %lld has fewer than %d "
                  "columns",
                  filename.c_str(), static_cast<long long>(line_number),
                  column_count);
  std::string extra;
  if (row >> extra)
    CCTK_VERROR("AH_analysis output '%s' line %lld has more than %d "
                "columns",
                filename.c_str(), static_cast<long long>(line_number),
                column_count);

  std::array<CCTK_REAL, column_count> numbers{};
  for (int column = 0; column < column_count; ++column) {
    std::istringstream parser(fields[column]);
    if (!(parser >> numbers[column]) || !(parser >> std::ws).eof() ||
        !std::isfinite(numbers[column]))
      CCTK_VERROR("AH_analysis output '%s' line %lld has an invalid finite "
                  "number in column %d",
                  filename.c_str(), static_cast<long long>(line_number),
                  column + 1);
  }

  std::array<std::int64_t, column_count> integers{};
  for (const int column : integer_columns) {
    std::istringstream parser(fields[column]);
    if (!(parser >> integers[column]) || !(parser >> std::ws).eof())
      CCTK_VERROR("AH_analysis output '%s' line %lld has a non-integer "
                  "value in column %d",
                  filename.c_str(), static_cast<long long>(line_number),
                  column + 1);
  }

  const auto iteration = integers[0];
  const auto stable_id = integers[2];
  const auto generation = integers[3];
  const auto lifecycle = integers[4];
  const auto role = integers[5];
  const auto parent_1 = integers[6];
  const auto parent_2 = integers[7];
  const auto published = integers[8];
  const auto ntheta = integers[9];
  const auto nphi = integers[10];
  const auto invalid_points = integers[38];
  const auto intrinsic_valid = integers[39];
  const auto intrinsic_invalid_points = integers[43];
  const auto invariant_requested = integers[44];
  const auto invariant_valid = integers[45];
  const auto invariant_status = integers[46];
  const auto axial_iterations = integers[58];
  const auto axial_harmonic_degree = integers[59];
  const auto invariant_coordinate_valid = integers[60];
  const auto multipoles_requested = integers[64];
  const auto multipoles_valid = integers[65];
  const auto multipole_maximum_degree = integers[66];
  const auto multipole_invalid_points = integers[67];
  const auto coordinate_fluxes_requested = integers[86];
  const auto coordinate_fluxes_valid = integers[87];
  const auto coordinate_flux_invalid_points = integers[88];
  const auto weyl_requested = integers[103];
  const auto weyl_valid = integers[104];
  const auto weyl_invalid_points = integers[105];
  const auto speciality_index_valid = integers[120];
  const auto speciality_index_invalid_points = integers[121];
  const auto speciality_difference_valid = integers[124];
  const auto speciality_difference_invalid_points = integers[125];
  if (iteration < 0 ||
      iteration > std::numeric_limits<int>::max() || stable_id <= 0 ||
      generation < 0 || lifecycle < 0 || lifecycle > 2 || role < 0 ||
      role > 2 || published < 0 || published > 1 || ntheta <= 0 ||
      nphi <= 0 || ntheta > std::numeric_limits<int>::max() ||
      nphi > std::numeric_limits<int>::max() || invalid_points < 0 ||
      intrinsic_valid < 0 || intrinsic_valid > 1 ||
      intrinsic_invalid_points < 0 ||
      intrinsic_valid != (intrinsic_invalid_points == 0) ||
      invariant_requested < 0 || invariant_requested > 1 ||
      invariant_valid < 0 || invariant_valid > 1 ||
      invariant_status < 0 || invariant_status > 8 ||
      invariant_valid != (invariant_requested && invariant_status == 0) ||
      (!invariant_requested && invariant_status != 1) ||
      axial_iterations < 0 || axial_harmonic_degree < 1 ||
      axial_harmonic_degree > 6 ||
      invariant_coordinate_valid < 0 || invariant_coordinate_valid > 1 ||
      invariant_coordinate_valid != invariant_valid ||
      multipoles_requested < 0 || multipoles_requested > 1 ||
      multipoles_valid < 0 || multipoles_valid > 1 ||
      multipole_maximum_degree != 8 || multipole_invalid_points < 0 ||
      multipoles_valid !=
          (multipoles_requested && invariant_valid &&
           multipole_invalid_points == 0) ||
      (!multipoles_requested && multipole_invalid_points != 0) ||
      coordinate_fluxes_requested < 0 ||
      coordinate_fluxes_requested > 1 || coordinate_fluxes_valid < 0 ||
      coordinate_fluxes_valid > 1 || coordinate_flux_invalid_points < 0 ||
      coordinate_fluxes_valid !=
          (coordinate_fluxes_requested &&
           coordinate_flux_invalid_points == 0) ||
      (!coordinate_fluxes_requested &&
       coordinate_flux_invalid_points != 0) ||
      weyl_requested < 0 || weyl_requested > 1 || weyl_valid < 0 ||
      weyl_valid > 1 || weyl_invalid_points < 0 ||
      weyl_valid != (weyl_requested && weyl_invalid_points == 0) ||
      (!weyl_requested && weyl_invalid_points != 0) ||
      speciality_index_valid < 0 || speciality_index_valid > 1 ||
      speciality_index_invalid_points < 0 ||
      speciality_index_valid !=
          (weyl_valid && speciality_index_invalid_points == 0) ||
      (!weyl_requested && speciality_index_invalid_points != 0) ||
      speciality_difference_valid < 0 ||
      speciality_difference_valid > 1 ||
      speciality_difference_invalid_points < 0 ||
      speciality_difference_valid !=
          (weyl_valid && speciality_difference_invalid_points == 0) ||
      (!weyl_requested && speciality_difference_invalid_points != 0) ||
      (generation == 0 && (parent_1 != 0 || parent_2 != 0)) ||
      (generation > 0 &&
       (parent_1 <= 0 || parent_2 <= 0 || parent_1 == parent_2 ||
        std::max(parent_1, parent_2) >= stable_id)))
    CCTK_VERROR("AH_analysis output '%s' line %lld has invalid identity or "
                "topology metadata",
                filename.c_str(), static_cast<long long>(line_number));
  if (!coordinate_fluxes_valid)
    for (int column = 89; column <= 102; ++column)
      if (numbers[column] != 0)
        CCTK_VERROR("AH_analysis output '%s' line %lld has nonzero invalid "
                    "coordinate-flux data in column %d",
                    filename.c_str(), static_cast<long long>(line_number),
                    column + 1);
  if (!weyl_valid)
    for (int column = 106; column <= 119; ++column)
      if (numbers[column] != 0)
        CCTK_VERROR("AH_analysis output '%s' line %lld has nonzero invalid "
                    "Weyl data in column %d",
                    filename.c_str(), static_cast<long long>(line_number),
                    column + 1);
  if (!speciality_index_valid)
    for (int column = 122; column <= 123; ++column)
      if (numbers[column] != 0)
        CCTK_VERROR("AH_analysis output '%s' line %lld has nonzero invalid "
                    "speciality-index data in column %d",
                    filename.c_str(), static_cast<long long>(line_number),
                    column + 1);
  if (!speciality_difference_valid)
    for (int column = 126; column <= 127; ++column)
      if (numbers[column] != 0)
        CCTK_VERROR("AH_analysis output '%s' line %lld has nonzero invalid "
                    "speciality-difference data in column %d",
                    filename.c_str(), static_cast<long long>(line_number),
                    column + 1);
  if (invariant_valid) {
    const CCTK_REAL signed_spin = numbers[47];
    const CCTK_REAL spin_magnitude = numbers[48];
    const CCTK_REAL axis_norm_squared =
        numbers[49] * numbers[49] + numbers[50] * numbers[50] +
        numbers[51] * numbers[51];
    const CCTK_REAL irreducible_mass = numbers[19];
    const CCTK_REAL expected_mass = std::sqrt(
        irreducible_mass * irreducible_mass +
        signed_spin * signed_spin /
            (4 * irreducible_mass * irreducible_mass));
    const CCTK_REAL expected_chi =
        signed_spin / (expected_mass * expected_mass);
    const CCTK_REAL scale =
        std::max<CCTK_REAL>(1, std::max(std::abs(expected_mass),
                                       std::abs(numbers[52])));
    const CCTK_REAL tolerance =
        8192 * std::numeric_limits<CCTK_REAL>::epsilon();
    if (!intrinsic_valid || !(irreducible_mass > 0) ||
        std::abs(spin_magnitude - std::abs(signed_spin)) >
            tolerance * std::max<CCTK_REAL>(1, spin_magnitude) ||
        std::abs(axis_norm_squared - 1) > tolerance ||
        std::abs(numbers[52] - expected_mass) > tolerance * scale ||
        std::abs(numbers[53] - expected_chi) > tolerance ||
        numbers[55] < 0 || numbers[56] < 0 || numbers[57] < 0)
      CCTK_VERROR("AH_analysis output '%s' line %lld has inconsistent "
                  "invariant-spin diagnostics",
                  filename.c_str(), static_cast<long long>(line_number));
    const CCTK_REAL coordinate_mean = numbers[61];
    const CCTK_REAL coordinate_minimum = numbers[62];
    const CCTK_REAL coordinate_maximum = numbers[63];
    const CCTK_REAL coordinate_scale = std::max<CCTK_REAL>(
        1, std::max(std::abs(coordinate_minimum),
                    std::abs(coordinate_maximum)));
    const CCTK_REAL coordinate_tolerance =
        32768 * std::numeric_limits<CCTK_REAL>::epsilon() *
        std::max<std::int64_t>(1, ntheta * nphi);
    if (std::abs(coordinate_mean) >
            coordinate_tolerance * coordinate_scale ||
        !(coordinate_maximum > coordinate_minimum) ||
        std::abs((coordinate_maximum - coordinate_minimum) - 2) >
            coordinate_tolerance * coordinate_scale)
      CCTK_VERROR("AH_analysis output '%s' line %lld has inconsistent "
                  "invariant-coordinate diagnostics",
                  filename.c_str(), static_cast<long long>(line_number));
  }
  if (multipoles_valid) {
    const CCTK_REAL expected_mass_monopole =
        numbers[52] * (1 + numbers[41]);
    const CCTK_REAL multipole_tolerance =
        32768 * std::numeric_limits<CCTK_REAL>::epsilon() *
        std::max<std::int64_t>(1, ntheta * nphi);
    if (std::abs(numbers[68] - expected_mass_monopole) >
            multipole_tolerance *
                std::max<CCTK_REAL>(1, std::abs(expected_mass_monopole)) ||
        std::abs(numbers[77]) > multipole_tolerance ||
        std::abs(numbers[78] - numbers[47]) >
            multipole_tolerance *
                std::max<CCTK_REAL>(1, std::abs(numbers[47])))
      CCTK_VERROR("AH_analysis output '%s' line %lld has inconsistent "
                  "invariant multipoles",
                  filename.c_str(), static_cast<long long>(line_number));
  }
  return {static_cast<int>(iteration), stable_id};
}

std::map<std::int64_t, int>
validated_output_watermarks(const std::string &filename) {
  std::map<std::int64_t, int> watermarks;
  std::set<std::pair<int, std::int64_t>> keys;
  std::ifstream input(filename);
  if (!input)
    return watermarks;
  bool saw_content = false;
  bool saw_schema = false;
  std::string line;
  std::int64_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty())
      continue;
    saw_content = true;
    if (line[0] == '#') {
      if (line.compare(0, 9, "# schema ") == 0) {
        if (saw_schema)
          CCTK_VERROR("AH_analysis output '%s' declares its schema more than "
                      "once",
                      filename.c_str());
        std::istringstream schema(line.substr(9));
        int version = -1;
        if (!(schema >> version) ||
            version != ParticleAHFinderX::surface_analysis_schema)
          CCTK_VERROR(
              "AH_analysis output '%s' has schema %d; expected %d",
              filename.c_str(), version,
              ParticleAHFinderX::surface_analysis_schema);
        saw_schema = true;
      }
      continue;
    }
    const auto key = validated_output_row(line, filename, line_number);
    if (!keys.emplace(key.iteration, key.stable_id).second)
      CCTK_VERROR(
          "AH_analysis output '%s' contains duplicate key (%d,%lld)",
          filename.c_str(), key.iteration,
          static_cast<long long>(key.stable_id));
    auto &watermark = watermarks[key.stable_id];
    watermark = std::max(watermark, key.iteration);
  }
  if (saw_content && !saw_schema)
    CCTK_VERROR("AH_analysis output '%s' has no schema declaration",
                filename.c_str());
  return watermarks;
}

void load_output_watermarks(const std::string &filename, Runtime &state) {
  state.last_output_iteration = validated_output_watermarks(filename);
  state.loaded_output = true;
}

void reconcile_after_recovery(const std::string &filename,
                              const int recovery_iteration,
                              Runtime &state) {
  static_cast<void>(validated_output_watermarks(filename));
  std::ifstream input(filename);
  if (!input) {
    state.last_output_iteration.clear();
    state.loaded_output = true;
    state.reconciled_recovery_iteration = recovery_iteration;
    return;
  }
  const std::string temporary = filename + ".recover.tmp";
  std::ofstream output(temporary, std::ios::out | std::ios::trunc);
  if (!output)
    CCTK_VERROR("AH_analysis could not create recovery file '%s'",
                temporary.c_str());
  std::string line;
  std::int64_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#') {
      output << line << '\n';
      continue;
    }
    const auto key = validated_output_row(line, filename, line_number);
    // CarpetX checkpoints after CCTK_ANALYSIS. The recovered iteration is
    // therefore committed output; only rows produced after that checkpoint
    // belong to the abandoned trajectory.
    if (key.iteration <= recovery_iteration)
      output << line << '\n';
  }
  output.close();
  input.close();
  if (!output || std::rename(temporary.c_str(), filename.c_str()) != 0)
    CCTK_VERROR("AH_analysis could not reconcile output file '%s' after "
                "recovery",
                filename.c_str());
  state.reconciled_recovery_iteration = recovery_iteration;
  load_output_watermarks(filename, state);
}

constexpr int weinberg_schema = 1;
constexpr int weinberg_columns = 13;

OutputRowKey validated_weinberg_row(const std::string &line,
                                    const std::string &filename,
                                    const std::int64_t line_number) {
  std::array<std::string, weinberg_columns> fields;
  std::istringstream row(line);
  for (auto &field : fields)
    if (!(row >> field))
      CCTK_VERROR("AH_analysis Weinberg output '%s' line %lld has fewer "
                  "than %d columns",
                  filename.c_str(), static_cast<long long>(line_number),
                  weinberg_columns);
  std::string extra;
  if (row >> extra)
    CCTK_VERROR("AH_analysis Weinberg output '%s' line %lld has more than "
                "%d columns",
                filename.c_str(), static_cast<long long>(line_number),
                weinberg_columns);

  std::array<CCTK_REAL, weinberg_columns> values{};
  for (int column = 0; column < weinberg_columns; ++column) {
    std::istringstream parser(fields[column]);
    if (!(parser >> values[column]) || !(parser >> std::ws).eof() ||
        !std::isfinite(values[column]))
      CCTK_VERROR("AH_analysis Weinberg output '%s' line %lld has an "
                  "invalid finite value in column %d",
                  filename.c_str(), static_cast<long long>(line_number),
                  column + 1);
  }
  std::int64_t iteration = -1;
  std::int64_t stable_id = 0;
  std::int64_t requested = 0;
  std::int64_t valid = 0;
  std::int64_t invalid_points = -1;
  std::istringstream iteration_parser(fields[0]);
  std::istringstream id_parser(fields[2]);
  std::istringstream requested_parser(fields[3]);
  std::istringstream valid_parser(fields[4]);
  std::istringstream invalid_parser(fields[5]);
  if (!(iteration_parser >> iteration) ||
      !(iteration_parser >> std::ws).eof() ||
      !(id_parser >> stable_id) || !(id_parser >> std::ws).eof() ||
      !(requested_parser >> requested) ||
      !(requested_parser >> std::ws).eof() || !(valid_parser >> valid) ||
      !(valid_parser >> std::ws).eof() ||
      !(invalid_parser >> invalid_points) ||
      !(invalid_parser >> std::ws).eof() || iteration < 0 ||
      iteration > std::numeric_limits<int>::max() || stable_id <= 0 ||
      requested != 1 || valid < 0 || valid > 1 || invalid_points < 0 ||
      valid != (invalid_points == 0))
    CCTK_VERROR("AH_analysis Weinberg output '%s' line %lld has invalid "
                "identity or feature metadata",
                filename.c_str(), static_cast<long long>(line_number));
  if (!valid)
    for (int column = 6; column < weinberg_columns; ++column)
      if (values[column] != 0)
        CCTK_VERROR("AH_analysis Weinberg output '%s' line %lld has "
                    "nonzero invalid payload in column %d",
                    filename.c_str(), static_cast<long long>(line_number),
                    column + 1);
  return {static_cast<int>(iteration), stable_id};
}

std::map<std::int64_t, int>
validated_weinberg_watermarks(const std::string &filename) {
  std::map<std::int64_t, int> watermarks;
  std::set<std::pair<int, std::int64_t>> keys;
  std::ifstream input(filename);
  if (!input)
    return watermarks;
  bool saw_content = false;
  bool saw_schema = false;
  std::string line;
  std::int64_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty())
      continue;
    saw_content = true;
    if (line[0] == '#') {
      if (line.compare(0, 9, "# schema ") == 0) {
        if (saw_schema)
          CCTK_VERROR("AH_analysis Weinberg output '%s' declares its "
                      "schema more than once",
                      filename.c_str());
        std::istringstream schema(line.substr(9));
        int version = -1;
        if (!(schema >> version) || version != weinberg_schema)
          CCTK_VERROR("AH_analysis Weinberg output '%s' has schema %d; "
                      "expected %d",
                      filename.c_str(), version, weinberg_schema);
        saw_schema = true;
      }
      continue;
    }
    const auto key = validated_weinberg_row(line, filename, line_number);
    if (!keys.emplace(key.iteration, key.stable_id).second)
      CCTK_VERROR("AH_analysis Weinberg output '%s' contains duplicate "
                  "key (%d,%lld)",
                  filename.c_str(), key.iteration,
                  static_cast<long long>(key.stable_id));
    auto &watermark = watermarks[key.stable_id];
    watermark = std::max(watermark, key.iteration);
  }
  if (saw_content && !saw_schema)
    CCTK_VERROR("AH_analysis Weinberg output '%s' has no schema declaration",
                filename.c_str());
  return watermarks;
}

void load_weinberg_watermarks(const std::string &filename, Runtime &state) {
  state.last_weinberg_output_iteration =
      validated_weinberg_watermarks(filename);
  state.loaded_weinberg_output = true;
}

void reconcile_weinberg_after_recovery(const std::string &filename,
                                        const int recovery_iteration,
                                        Runtime &state) {
  static_cast<void>(validated_weinberg_watermarks(filename));
  std::ifstream input(filename);
  if (!input) {
    state.last_weinberg_output_iteration.clear();
    state.loaded_weinberg_output = true;
    return;
  }
  const std::string temporary = filename + ".recover.tmp";
  std::ofstream output(temporary, std::ios::out | std::ios::trunc);
  if (!output)
    CCTK_VERROR("AH_analysis could not create Weinberg recovery file '%s'",
                temporary.c_str());
  std::string line;
  std::int64_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#') {
      output << line << '\n';
      continue;
    }
    const auto key = validated_weinberg_row(line, filename, line_number);
    if (key.iteration <= recovery_iteration)
      output << line << '\n';
  }
  output.close();
  input.close();
  if (!output || std::rename(temporary.c_str(), filename.c_str()) != 0)
    CCTK_VERROR("AH_analysis could not reconcile Weinberg output '%s' "
                "after recovery",
                filename.c_str());
  load_weinberg_watermarks(filename, state);
}

void write_header(std::ostream &output) {
  constexpr std::array<const char *, 128> column_names{{
      "iteration",
      "time",
      "stable_id",
      "generation",
      "lifecycle",
      "role",
      "parent_1",
      "parent_2",
      "published",
      "ntheta",
      "nphi",
      "center_x",
      "center_y",
      "center_z",
      "centroid_x",
      "centroid_y",
      "centroid_z",
      "area",
      "areal_radius",
      "irreducible_mass",
      "momentum_covariant_x",
      "momentum_covariant_y",
      "momentum_covariant_z",
      "coordinate_spin_center_x",
      "coordinate_spin_center_y",
      "coordinate_spin_center_z",
      "coordinate_spin_centroid_x",
      "coordinate_spin_centroid_y",
      "coordinate_spin_centroid_z",
      "coordinate_spin_centroid_magnitude",
      "coordinate_christodoulou_mass",
      "coordinate_dimensionless_spin",
      "xy_circumference",
      "xz_circumference",
      "yz_circumference",
      "mean_expansion",
      "expansion_l2",
      "expansion_linf",
      "invalid_points",
      "intrinsic_geometry_valid",
      "intrinsic_scalar_curvature_integral",
      "gauss_bonnet_residual",
      "normal_connection_l2",
      "intrinsic_invalid_points",
      "invariant_spin_requested",
      "invariant_spin_valid",
      "invariant_spin_status",
      "invariant_spin_signed",
      "invariant_spin_magnitude",
      "invariant_spin_axis_x",
      "invariant_spin_axis_y",
      "invariant_spin_axis_z",
      "invariant_christodoulou_mass",
      "invariant_dimensionless_spin",
      "axial_eigenvalue",
      "axial_eigen_residual",
      "axial_killing_residual",
      "axial_normalization_residual",
      "axial_iterations",
      "axial_harmonic_degree",
      "invariant_coordinate_valid",
      "invariant_coordinate_mean",
      "invariant_coordinate_minimum",
      "invariant_coordinate_maximum",
      "multipoles_requested",
      "multipoles_valid",
      "multipole_maximum_degree",
      "multipole_invalid_points",
      "mass_multipole_l0",
      "mass_multipole_l1",
      "mass_multipole_l2",
      "mass_multipole_l3",
      "mass_multipole_l4",
      "mass_multipole_l5",
      "mass_multipole_l6",
      "mass_multipole_l7",
      "mass_multipole_l8",
      "spin_multipole_l0",
      "spin_multipole_l1",
      "spin_multipole_l2",
      "spin_multipole_l3",
      "spin_multipole_l4",
      "spin_multipole_l5",
      "spin_multipole_l6",
      "spin_multipole_l7",
      "spin_multipole_l8",
      "coordinate_fluxes_requested",
      "coordinate_fluxes_valid",
      "coordinate_flux_invalid_points",
      "adm_energy",
      "adm_momentum_x",
      "adm_momentum_y",
      "adm_momentum_z",
      "adm_angular_momentum_x",
      "adm_angular_momentum_y",
      "adm_angular_momentum_z",
      "conformal_adm_energy",
      "conformal_adm_momentum_x",
      "conformal_adm_momentum_y",
      "conformal_adm_momentum_z",
      "conformal_adm_angular_momentum_x",
      "conformal_adm_angular_momentum_y",
      "conformal_adm_angular_momentum_z",
      "weyl_curvature_requested",
      "weyl_curvature_valid",
      "weyl_curvature_invalid_points",
      "mean_psi0_real",
      "mean_psi0_imag",
      "mean_psi1_real",
      "mean_psi1_imag",
      "mean_psi2_real",
      "mean_psi2_imag",
      "mean_psi3_real",
      "mean_psi3_imag",
      "mean_psi4_real",
      "mean_psi4_imag",
      "mean_invariant_i_real",
      "mean_invariant_i_imag",
      "mean_invariant_j_real",
      "mean_invariant_j_imag",
      "speciality_index_valid",
      "speciality_index_invalid_points",
      "mean_speciality_index_real",
      "mean_speciality_index_imag",
      "speciality_difference_valid",
      "speciality_difference_invalid_points",
      "mean_speciality_difference_real",
      "mean_speciality_difference_imag",
  }};

  output << "# schema " << ParticleAHFinderX::surface_analysis_schema
         << "\n"
            "# One tab-delimited row per analyzed (iteration, stable_id); "
            "iteration and time are Cactus cctk_iteration and cctk_time.\n"
            "# Geometric units use G=c=1. Boolean fields use 0=false, "
            "1=true. A parent ID of 0 means no parent.\n"
            "# lifecycle: 0=candidate 1=active 2=retired; role: "
            "0=individual 1=common 2=other.\n"
            "# published=1 means the stable surface is exposed to consumers; "
            "AH_analysis emits physical rows only after the finder supplies "
            "valid converged geometry.\n"
            "# invariant_spin_status: 0=valid 1=not_requested "
            "2=invalid_intrinsic_geometry 3=invalid_basis "
            "4=singular_mass_matrix 5=eigensolver_failed 6=invalid_axis "
            "7=invalid_normalization 8=invalid_spin_integral.\n"
            "# A feature with requested=0 has valid=0 and zero-valued "
            "payload columns; those zeros are not physical measurements.\n"
            "# Coordinate spins use rotations about the radial-graph center "
            "and area centroid; neither is an approximate-Killing-vector "
            "spin. momentum_covariant_* corresponds to QLM's "
            "quasilocal_momentum_covariant_*.\n"
            "# Column groups: 1-17 identity/topology/position; 18-39 core "
            "geometry and coordinate charges; 40-44 intrinsic geometry; "
            "45-64 invariant spin; 65-86 multipoles; 87-103 coordinate "
            "ADM fluxes; 104-128 Weyl and speciality diagnostics.\n";
  for (std::size_t column = 0; column < column_names.size(); ++column)
    output << "# column " << std::setw(3) << column + 1 << " = "
           << column_names[column] << '\n';
  output << '#';
  for (std::size_t column = 0; column < column_names.size(); ++column)
    output << ' ' << column + 1 << ':' << column_names[column];
  output << '\n';
}

struct WriteResult {
  std::size_t records = 0;
  std::int64_t bytes = 0;
};

WriteResult write_records(
    const std::string &filename,
    const std::vector<ParticleAHFinderX::SurfaceAnalysisRecord> &records,
    Runtime &state) {
  const bool has_pending_record = std::any_of(
      records.begin(), records.end(), [&](const auto &record) {
        const auto found =
            state.last_output_iteration.find(record.stable_id);
        return found == state.last_output_iteration.end() ||
               found->second < record.iteration;
      });
  if (!has_pending_record)
    return {};
  const bool write_file_header = file_is_empty(filename);
  std::ofstream output(filename, std::ios::out | std::ios::app);
  if (!output)
    CCTK_VERROR("AH_analysis could not open output file '%s'",
                filename.c_str());
  const std::streamoff initial_position = output.tellp();
  if (write_file_header)
    write_header(output);
  output << std::setprecision(17);
  std::size_t written = 0;
  for (const auto &record : records) {
    const auto found = state.last_output_iteration.find(record.stable_id);
    if (found != state.last_output_iteration.end() &&
        found->second >= record.iteration)
      continue;
    output << record.iteration << '\t' << record.time << '\t'
           << record.stable_id << '\t' << record.generation << '\t'
           << record.lifecycle << '\t' << record.role << '\t'
           << record.parent_ids[0] << '\t' << record.parent_ids[1] << '\t'
           << record.published << '\t' << record.ntheta << '\t'
           << record.nphi;
    for (const auto value : record.center)
      output << '\t' << value;
    for (const auto value : record.centroid)
      output << '\t' << value;
    output << '\t' << record.area << '\t' << record.areal_radius << '\t'
           << record.irreducible_mass;
    for (const auto value : record.momentum_covariant)
      output << '\t' << value;
    for (const auto value : record.coordinate_spin_center)
      output << '\t' << value;
    for (const auto value : record.coordinate_spin_centroid)
      output << '\t' << value;
    output << '\t' << record.coordinate_spin_magnitude << '\t'
           << record.christodoulou_mass << '\t'
           << record.dimensionless_spin;
    for (const auto value : record.proper_circumference)
      output << '\t' << value;
    output << '\t' << record.mean_expansion << '\t'
           << record.expansion_l2 << '\t' << record.expansion_linf << '\t'
           << record.invalid_points << '\t'
           << record.intrinsic_geometry_valid << '\t'
           << record.intrinsic_scalar_curvature_integral << '\t'
           << record.gauss_bonnet_residual << '\t'
           << record.normal_connection_l2 << '\t'
           << record.intrinsic_invalid_points << '\t'
           << record.invariant_spin_requested << '\t'
           << record.invariant_spin_valid << '\t'
           << record.invariant_spin_status << '\t'
           << record.invariant_spin_signed << '\t'
           << record.invariant_spin_magnitude;
    for (const auto value : record.invariant_spin_axis)
      output << '\t' << value;
    output << '\t' << record.invariant_christodoulou_mass << '\t'
           << record.invariant_dimensionless_spin << '\t'
           << record.axial_eigenvalue << '\t'
           << record.axial_eigen_residual << '\t'
           << record.axial_killing_residual << '\t'
           << record.axial_normalization_residual << '\t'
           << record.axial_iterations << '\t'
           << record.axial_harmonic_degree << '\t'
           << record.invariant_coordinate_valid << '\t'
           << record.invariant_coordinate_mean << '\t'
           << record.invariant_coordinate_minimum << '\t'
           << record.invariant_coordinate_maximum << '\t'
           << record.multipoles_requested << '\t'
           << record.multipoles_valid << '\t'
           << record.multipole_maximum_degree << '\t'
           << record.multipole_invalid_points;
    for (const auto value : record.mass_multipoles)
      output << '\t' << value;
    for (const auto value : record.spin_multipoles)
      output << '\t' << value;
    output << '\t' << record.coordinate_fluxes_requested << '\t'
           << record.coordinate_fluxes_valid << '\t'
           << record.coordinate_flux_invalid_points << '\t'
           << record.adm_energy;
    for (const auto value : record.adm_momentum)
      output << '\t' << value;
    for (const auto value : record.adm_angular_momentum)
      output << '\t' << value;
    output << '\t' << record.conformal_adm_energy;
    for (const auto value : record.conformal_adm_momentum)
      output << '\t' << value;
    for (const auto value : record.conformal_adm_angular_momentum)
      output << '\t' << value;
    output << '\t' << record.weyl_curvature_requested << '\t'
           << record.weyl_curvature_valid << '\t'
           << record.weyl_curvature_invalid_points;
    for (const auto &value : record.mean_weyl_scalar)
      output << '\t' << value.real << '\t' << value.imag;
    output << '\t' << record.mean_weyl_invariant_i.real << '\t'
           << record.mean_weyl_invariant_i.imag << '\t'
           << record.mean_weyl_invariant_j.real << '\t'
           << record.mean_weyl_invariant_j.imag << '\t'
           << record.speciality_index_valid << '\t'
           << record.speciality_index_invalid_points << '\t'
           << record.mean_speciality_index.real << '\t'
           << record.mean_speciality_index.imag << '\t'
           << record.speciality_difference_valid << '\t'
           << record.speciality_difference_invalid_points << '\t'
           << record.mean_speciality_difference.real << '\t'
           << record.mean_speciality_difference.imag;
    output << '\n';
    state.last_output_iteration[record.stable_id] = record.iteration;
    ++written;
  }
  if (!output)
    CCTK_VERROR("AH_analysis could not write output file '%s'",
                filename.c_str());
  const std::streamoff final_position = output.tellp();
  return {written,
          initial_position >= 0 && final_position >= initial_position
              ? static_cast<std::int64_t>(final_position - initial_position)
              : 0};
}

void write_weinberg_header(std::ostream &output) {
  constexpr std::array<const char *, weinberg_columns> columns{{
      "iteration",
      "time",
      "stable_id",
      "requested",
      "valid",
      "invalid_points",
      "weinberg_energy",
      "weinberg_momentum_x",
      "weinberg_momentum_y",
      "weinberg_momentum_z",
      "weinberg_angular_momentum_x",
      "weinberg_angular_momentum_y",
      "weinberg_angular_momentum_z",
  }};
  output << "# schema " << weinberg_schema
         << "\n"
            "# Gauge-consistent full-four-metric Weinberg surface fluxes; "
            "signature (-,+,+,+), G=c=1.\n"
            "# Angular momentum uses the global Cartesian origin. A "
            "requested but invalid row stores valid=0, an explicit positive "
            "invalid_points count, and neutral payload values.\n";
  for (std::size_t column = 0; column < columns.size(); ++column)
    output << "# column " << std::setw(2) << column + 1 << " = "
           << columns[column] << '\n';
  output << '#';
  for (std::size_t column = 0; column < columns.size(); ++column)
    output << ' ' << column + 1 << ':' << columns[column];
  output << '\n';
}

WriteResult write_weinberg_records(
    const std::string &filename,
    const std::vector<ParticleAHFinderX::SurfaceAnalysisRecord> &records,
    Runtime &state) {
  const bool has_pending_record = std::any_of(
      records.begin(), records.end(), [&](const auto &record) {
        if (!record.weinberg_fluxes_requested)
          return false;
        const auto found =
            state.last_weinberg_output_iteration.find(record.stable_id);
        return found == state.last_weinberg_output_iteration.end() ||
               found->second < record.iteration;
      });
  if (!has_pending_record)
    return {};
  const bool write_file_header = file_is_empty(filename);
  std::ofstream output(filename, std::ios::out | std::ios::app);
  if (!output)
    CCTK_VERROR("AH_analysis could not open Weinberg output file '%s'",
                filename.c_str());
  const std::streamoff initial_position = output.tellp();
  if (write_file_header)
    write_weinberg_header(output);
  output << std::setprecision(17);
  std::size_t written = 0;
  for (const auto &record : records) {
    if (!record.weinberg_fluxes_requested)
      continue;
    const auto found =
        state.last_weinberg_output_iteration.find(record.stable_id);
    if (found != state.last_weinberg_output_iteration.end() &&
        found->second >= record.iteration)
      continue;
    output << record.iteration << '\t' << record.time << '\t'
           << record.stable_id << '\t' << record.weinberg_fluxes_requested
           << '\t' << record.weinberg_fluxes_valid << '\t'
           << record.weinberg_flux_invalid_points << '\t'
           << record.weinberg_energy;
    for (const auto value : record.weinberg_momentum)
      output << '\t' << value;
    for (const auto value : record.weinberg_angular_momentum)
      output << '\t' << value;
    output << '\n';
    state.last_weinberg_output_iteration[record.stable_id] =
        record.iteration;
    ++written;
  }
  if (!output)
    CCTK_VERROR("AH_analysis could not write Weinberg output file '%s'",
                filename.c_str());
  const std::streamoff final_position = output.tellp();
  return {written,
          initial_position >= 0 && final_position >= initial_position
              ? static_cast<std::int64_t>(final_position - initial_position)
              : 0};
}

constexpr int ricci_schema = 1;
constexpr int ricci_columns = 16;

OutputRowKey validated_ricci_row(const std::string &line,
                                 const std::string &filename,
                                 const std::int64_t line_number) {
  std::array<std::string, ricci_columns> fields;
  std::istringstream row(line);
  for (auto &field : fields)
    if (!(row >> field))
      CCTK_VERROR("AH_analysis Ricci output '%s' line %lld has fewer than %d "
                  "columns",
                  filename.c_str(), static_cast<long long>(line_number),
                  ricci_columns);
  std::string extra;
  if (row >> extra)
    CCTK_VERROR("AH_analysis Ricci output '%s' line %lld has more than %d "
                "columns",
                filename.c_str(), static_cast<long long>(line_number),
                ricci_columns);

  std::array<CCTK_REAL, ricci_columns> values{};
  for (int column = 0; column < ricci_columns; ++column) {
    std::istringstream parser(fields[column]);
    if (!(parser >> values[column]) || !(parser >> std::ws).eof() ||
        !std::isfinite(values[column]))
      CCTK_VERROR("AH_analysis Ricci output '%s' line %lld has an invalid "
                  "finite value in column %d",
                  filename.c_str(), static_cast<long long>(line_number),
                  column + 1);
  }
  std::int64_t iteration = -1;
  std::int64_t stable_id = 0;
  std::int64_t source = 0;
  std::int64_t valid = 0;
  std::int64_t invalid_points = -1;
  std::istringstream iteration_parser(fields[0]);
  std::istringstream id_parser(fields[2]);
  std::istringstream source_parser(fields[3]);
  std::istringstream valid_parser(fields[4]);
  std::istringstream invalid_parser(fields[5]);
  if (!(iteration_parser >> iteration) ||
      !(iteration_parser >> std::ws).eof() ||
      !(id_parser >> stable_id) || !(id_parser >> std::ws).eof() ||
      !(source_parser >> source) || !(source_parser >> std::ws).eof() ||
      !(valid_parser >> valid) || !(valid_parser >> std::ws).eof() ||
      !(invalid_parser >> invalid_points) ||
      !(invalid_parser >> std::ws).eof() || iteration < 0 ||
      iteration > std::numeric_limits<int>::max() || stable_id <= 0 ||
      source < static_cast<int>(ParticleAHFinderX::SurfaceRicciSource::vacuum) ||
      source > static_cast<int>(ParticleAHFinderX::SurfaceRicciSource::tmunu) ||
      valid < 0 || valid > 1 || invalid_points < 0 ||
      valid != (invalid_points == 0))
    CCTK_VERROR("AH_analysis Ricci output '%s' line %lld has invalid "
                "identity or feature metadata",
                filename.c_str(), static_cast<long long>(line_number));
  if (!valid)
    for (int column = 6; column < ricci_columns; ++column)
      if (values[column] != 0)
        CCTK_VERROR("AH_analysis Ricci output '%s' line %lld has nonzero "
                    "invalid payload in column %d",
                    filename.c_str(), static_cast<long long>(line_number),
                    column + 1);
  if (valid &&
      source ==
          static_cast<int>(ParticleAHFinderX::SurfaceRicciSource::vacuum))
    for (int column = 6; column < ricci_columns; ++column)
      if (values[column] != 0)
        CCTK_VERROR("AH_analysis vacuum Ricci output '%s' line %lld has "
                    "nonzero payload in column %d",
                    filename.c_str(), static_cast<long long>(line_number),
                    column + 1);
  return {static_cast<int>(iteration), stable_id};
}

std::map<std::int64_t, int>
validated_ricci_watermarks(const std::string &filename) {
  std::map<std::int64_t, int> watermarks;
  std::set<std::pair<int, std::int64_t>> keys;
  std::ifstream input(filename);
  if (!input)
    return watermarks;
  bool saw_content = false;
  bool saw_schema = false;
  std::string line;
  std::int64_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty())
      continue;
    saw_content = true;
    if (line[0] == '#') {
      if (line.compare(0, 9, "# schema ") == 0) {
        if (saw_schema)
          CCTK_VERROR("AH_analysis Ricci output '%s' declares its schema "
                      "more than once",
                      filename.c_str());
        std::istringstream schema(line.substr(9));
        int version = -1;
        if (!(schema >> version) || version != ricci_schema)
          CCTK_VERROR("AH_analysis Ricci output '%s' has schema %d; expected "
                      "%d",
                      filename.c_str(), version, ricci_schema);
        saw_schema = true;
      }
      continue;
    }
    const auto key = validated_ricci_row(line, filename, line_number);
    if (!keys.emplace(key.iteration, key.stable_id).second)
      CCTK_VERROR("AH_analysis Ricci output '%s' contains duplicate key "
                  "(%d,%lld)",
                  filename.c_str(), key.iteration,
                  static_cast<long long>(key.stable_id));
    auto &watermark = watermarks[key.stable_id];
    watermark = std::max(watermark, key.iteration);
  }
  if (saw_content && !saw_schema)
    CCTK_VERROR("AH_analysis Ricci output '%s' has no schema declaration",
                filename.c_str());
  return watermarks;
}

void load_ricci_watermarks(const std::string &filename, Runtime &state) {
  state.last_ricci_output_iteration = validated_ricci_watermarks(filename);
  state.loaded_ricci_output = true;
}

void reconcile_ricci_after_recovery(const std::string &filename,
                                    const int recovery_iteration,
                                    Runtime &state) {
  static_cast<void>(validated_ricci_watermarks(filename));
  std::ifstream input(filename);
  if (!input) {
    state.last_ricci_output_iteration.clear();
    state.loaded_ricci_output = true;
    return;
  }
  const std::string temporary = filename + ".recover.tmp";
  std::ofstream output(temporary, std::ios::out | std::ios::trunc);
  if (!output)
    CCTK_VERROR("AH_analysis could not create Ricci recovery file '%s'",
                temporary.c_str());
  std::string line;
  std::int64_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#') {
      output << line << '\n';
      continue;
    }
    const auto key = validated_ricci_row(line, filename, line_number);
    if (key.iteration <= recovery_iteration)
      output << line << '\n';
  }
  output.close();
  input.close();
  if (!output || std::rename(temporary.c_str(), filename.c_str()) != 0)
    CCTK_VERROR("AH_analysis could not reconcile Ricci output file '%s' "
                "after recovery",
                filename.c_str());
  load_ricci_watermarks(filename, state);
}

void write_ricci_header(std::ostream &output) {
  constexpr std::array<const char *, ricci_columns> columns{{
      "iteration",
      "time",
      "stable_id",
      "ricci_source",
      "valid",
      "invalid_points",
      "mean_phi00",
      "mean_phi01_real",
      "mean_phi01_imag",
      "mean_phi02_real",
      "mean_phi02_imag",
      "mean_phi11",
      "mean_phi12_real",
      "mean_phi12_imag",
      "mean_phi22",
      "mean_lambda",
  }};
  output << "# schema " << ricci_schema
         << "\n"
            "# Area-weighted Newman--Penrose Ricci projections; signature "
            "(-,+,+,+), G=c=1. ricci_source: 1=vacuum, 2=TmunuBaseX.\n"
            "# Complex projections retain both components. A requested but "
            "invalid row stores valid=0, a positive invalid_points count, "
            "and neutral payload values.\n";
  for (std::size_t column = 0; column < columns.size(); ++column)
    output << "# column " << std::setw(2) << column + 1 << " = "
           << columns[column] << '\n';
  output << '#';
  for (std::size_t column = 0; column < columns.size(); ++column)
    output << ' ' << column + 1 << ':' << columns[column];
  output << '\n';
}

WriteResult write_ricci_records(
    const std::string &filename,
    const std::vector<ParticleAHFinderX::SurfaceAnalysisRecord> &records,
    Runtime &state) {
  const bool has_pending_record = std::any_of(
      records.begin(), records.end(), [&](const auto &record) {
        if (record.ricci_source == ParticleAHFinderX::SurfaceRicciSource::off)
          return false;
        const auto found =
            state.last_ricci_output_iteration.find(record.stable_id);
        return found == state.last_ricci_output_iteration.end() ||
               found->second < record.iteration;
      });
  if (!has_pending_record)
    return {};
  const bool write_file_header = file_is_empty(filename);
  std::ofstream output(filename, std::ios::out | std::ios::app);
  if (!output)
    CCTK_VERROR("AH_analysis could not open Ricci output file '%s'",
                filename.c_str());
  const std::streamoff initial_position = output.tellp();
  if (write_file_header)
    write_ricci_header(output);
  output << std::setprecision(17);
  std::size_t written = 0;
  for (const auto &record : records) {
    if (record.ricci_source == ParticleAHFinderX::SurfaceRicciSource::off)
      continue;
    const auto found =
        state.last_ricci_output_iteration.find(record.stable_id);
    if (found != state.last_ricci_output_iteration.end() &&
        found->second >= record.iteration)
      continue;
    output << record.iteration << '\t' << record.time << '\t'
           << record.stable_id << '\t'
           << static_cast<int>(record.ricci_source) << '\t'
           << record.ricci_scalars_valid << '\t'
           << record.ricci_invalid_points << '\t'
           << record.mean_ricci_phi00 << '\t'
           << record.mean_ricci_phi01.real << '\t'
           << record.mean_ricci_phi01.imag << '\t'
           << record.mean_ricci_phi02.real << '\t'
           << record.mean_ricci_phi02.imag << '\t'
           << record.mean_ricci_phi11 << '\t'
           << record.mean_ricci_phi12.real << '\t'
           << record.mean_ricci_phi12.imag << '\t'
           << record.mean_ricci_phi22 << '\t'
           << record.mean_ricci_lambda << '\n';
    state.last_ricci_output_iteration[record.stable_id] = record.iteration;
    ++written;
  }
  if (!output)
    CCTK_VERROR("AH_analysis could not write Ricci output file '%s'",
                filename.c_str());
  const std::streamoff final_position = output.tellp();
  return {written,
          initial_position >= 0 && final_position >= initial_position
              ? static_cast<std::int64_t>(final_position - initial_position)
              : 0};
}

} // namespace

extern "C" void AH_analysis_ParamCheck(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  if (compute_multipoles && !compute_invariant_spin)
    CCTK_ERROR("AH_analysis::compute_multipoles requires "
               "AH_analysis::compute_invariant_spin");
  if (CCTK_EQUALS(ricci_source, "tmunu") &&
      !CCTK_IsThornActive("TmunuBaseX"))
    CCTK_PARAMWARN("AH_analysis::ricci_source=tmunu requires the "
                   "TmunuBaseX thorn");
  if (point_output_every > 0 && point_output_every % output_every != 0)
    CCTK_PARAMWARN("AH_analysis::point_output_every must be an integer "
                   "multiple of AH_analysis::output_every so every point "
                   "frame has a committed scalar transaction");
  if (point_output_every > 0 &&
      !ParticleAHFinderX::surface_analysis_point_output_available())
    CCTK_PARAMWARN("AH_analysis point output was requested, but this "
                   "configuration lacks openPMD_api");
}

extern "C" void AH_analysis_Calculate(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  const auto status =
      ParticleAHFinderX::surface_analysis_runtime_status();
  if (!status.ready) {
    if (verbose && !state.reported_not_ready) {
      CCTK_INFO("AH_analysis is waiting for ParticleAHFinderX surface "
                "storage");
      state.reported_not_ready = true;
    }
    return;
  }
  state.reported_not_ready = false;
  if (verbose && !state.reported_transport) {
    const char *const transport =
        !status.accelerator_backend
            ? "host backend"
            : (status.gpu_aware_mpi
                   ? "GPU-aware MPI with direct device records"
                   : "AMReX pinned fallback for compact records");
    CCTK_VINFO("AH_analysis collective transport: %s; angular point data "
               "remain device resident",
               transport);
    state.reported_transport = true;
  }

  const std::string filename = std::string(out_dir) + "/ah_analysis.tsv";
  const std::string weinberg_filename =
      std::string(out_dir) + "/weinberg_fluxes.tsv";
  const std::string ricci_filename =
      std::string(out_dir) + "/ricci_scalars.tsv";
  const auto requested_ricci_source = selected_ricci_source(ricci_source);
  const bool must_reconcile =
      status.recovered && status.recovered_iteration >= 0 &&
      state.reconciled_recovery_iteration != status.recovered_iteration;
  if (must_reconcile) {
    if (CCTK_CreateDirectory(0755, out_dir) < 0)
      CCTK_VERROR("AH_analysis could not create output directory '%s'",
                  out_dir);
    if (CCTK_MyProc(cctkGH) == 0) {
      reconcile_after_recovery(filename, status.recovered_iteration, state);
      if (compute_weinberg_fluxes)
        reconcile_weinberg_after_recovery(
            weinberg_filename, status.recovered_iteration, state);
      if (requested_ricci_source !=
          ParticleAHFinderX::SurfaceRicciSource::off)
        reconcile_ricci_after_recovery(
            ricci_filename, status.recovered_iteration, state);
    }
    CCTK_Barrier(cctkGH);
    if (point_output_every > 0)
      ParticleAHFinderX::reconcile_surface_analysis_point_output(
          cctkGH, point_out_dir, point_out_backend,
          status.recovered_iteration);
    state.reconciled_recovery_iteration = status.recovered_iteration;
  }

  if (cctk_iteration % output_every != 0)
    return;
  ParticleAHFinderX::SurfaceAnalysisOptions options;
  options.compute_invariant_spin = compute_invariant_spin;
  options.compute_multipoles = compute_multipoles;
  options.compute_coordinate_fluxes = compute_coordinate_fluxes;
  options.compute_weyl_curvature = compute_weyl_curvature;
  options.compute_weinberg_fluxes = compute_weinberg_fluxes;
  options.ricci_source = requested_ricci_source;
  options.axial_harmonic_degree = axial_harmonic_degree;
  options.axial_maximum_jacobi_sweeps =
      axial_maximum_jacobi_sweeps;
  options.axial_eigen_tolerance = axial_eigen_tolerance;
  options.axial_minimum_axis_separation =
      axial_minimum_axis_separation;
  ParticleAHFinderX::SurfaceAnalysisPerformance performance;
  const auto records = ParticleAHFinderX::current_surface_analysis(
      cctkGH, cctk_iteration, published_only,
      print_analysis_timing ? &performance : nullptr, options);
  if (records.empty()) {
    if (print_analysis_timing && CCTK_MyProc(cctkGH) == 0 &&
        performance.global_surface_count != 0)
      CCTK_VINFO(
          "AH_analysis performance: iteration=%d time=%.17g surfaces=%d "
          "records=0 total=%.6g s device-kernel/collective/d2h/host-record="
          "%.6g/%.6g/%.6g/%.6g s; "
          "kernels=%d collectives=%d syncs=%d "
          "local-surfaces/points=%d/%lld axial-modes=%d multipole-degree=%d "
          "intrinsic/axial-kernels=%d/%d workspace-intrinsic/axial-point/"
          "axial-projected=%lld/%lld/%lld B "
          "weyl-gather/routing=%.6g/%.6g s weyl-kernels=%d/%d "
          "workspace-weyl-mesh/logical=%lld/%lld B weyl-payload=%lld B "
          "gauge-gather/routing=%.6g/%.6g s gauge-kernels=%d/%d "
          "workspace-gauge-mesh/logical=%lld/%lld B gauge-payload=%lld B "
          "matter-gather/routing=%.6g/%.6g s matter-kernels=%d/%d "
          "workspace-matter-mesh/logical=%lld/%lld B matter-payload=%lld B "
          "compact collective/stage/d2h/h2d=%lld/%lld/%lld/%lld B "
          "collective-stage-copies=%d angular-point-host=0 B "
          "direct-device=%s",
          cctk_iteration, double(cctk_time),
          performance.global_surface_count,
          double(performance.total_seconds),
          double(performance.device_kernel_seconds),
          double(performance.collective_seconds),
          double(performance.compact_device_to_host_seconds),
          double(performance.host_record_seconds),
          performance.device_kernel_launches, performance.device_collectives,
          performance.explicit_device_synchronizations,
          performance.local_surface_count,
          static_cast<long long>(performance.local_angular_points),
          performance.axial_trial_modes,
          performance.multipole_maximum_degree,
          performance.intrinsic_geometry_kernel_launches,
          performance.axial_symmetry_kernel_launches,
          static_cast<long long>(performance.intrinsic_workspace_bytes),
          static_cast<long long>(performance.axial_point_workspace_bytes),
          static_cast<long long>(
              performance.axial_projected_workspace_bytes),
          double(performance.weyl_gather_seconds),
          double(performance.weyl_routing_seconds),
          performance.weyl_gather_kernel_launches,
          performance.weyl_routing_kernel_launches,
          static_cast<long long>(performance.weyl_mesh_workspace_bytes),
          static_cast<long long>(performance.weyl_logical_workspace_bytes),
          static_cast<long long>(performance.weyl_device_payload_bytes),
          double(performance.gauge_gather_seconds),
          double(performance.gauge_routing_seconds),
          performance.gauge_gather_kernel_launches,
          performance.gauge_routing_kernel_launches,
          static_cast<long long>(performance.gauge_mesh_workspace_bytes),
          static_cast<long long>(performance.gauge_logical_workspace_bytes),
          static_cast<long long>(performance.gauge_device_payload_bytes),
          double(performance.matter_gather_seconds),
          double(performance.matter_routing_seconds),
          performance.matter_gather_kernel_launches,
          performance.matter_routing_kernel_launches,
          static_cast<long long>(performance.matter_mesh_workspace_bytes),
          static_cast<long long>(performance.matter_logical_workspace_bytes),
          static_cast<long long>(performance.matter_device_payload_bytes),
          static_cast<long long>(performance.compact_collective_bytes),
          static_cast<long long>(
              performance.compact_collective_host_staging_bytes),
          static_cast<long long>(performance.compact_device_to_host_bytes),
          static_cast<long long>(performance.compact_host_to_device_bytes),
          performance.compact_collective_host_staging_copies,
          performance.direct_device_collective ? "yes" : "no");
    return;
  }
  if (CCTK_CreateDirectory(0755, out_dir) < 0)
    CCTK_VERROR("AH_analysis could not create output directory '%s'",
                out_dir);
  WriteResult output;
  WriteResult weinberg_output;
  WriteResult ricci_output;
  double io_seconds = 0;
  if (CCTK_MyProc(cctkGH) == 0) {
    if (!state.loaded_output)
      load_output_watermarks(filename, state);
    if (compute_weinberg_fluxes && !state.loaded_weinberg_output)
      load_weinberg_watermarks(weinberg_filename, state);
    if (requested_ricci_source !=
            ParticleAHFinderX::SurfaceRicciSource::off &&
        !state.loaded_ricci_output)
      load_ricci_watermarks(ricci_filename, state);
    const auto io_start = std::chrono::steady_clock::now();
    output = write_records(filename, records, state);
    if (compute_weinberg_fluxes)
      weinberg_output =
          write_weinberg_records(weinberg_filename, records, state);
    if (requested_ricci_source !=
        ParticleAHFinderX::SurfaceRicciSource::off)
      ricci_output = write_ricci_records(ricci_filename, records, state);
    const std::chrono::duration<double> io_time =
        std::chrono::steady_clock::now() - io_start;
    io_seconds = io_time.count();
    if (verbose && (output.records != 0 || weinberg_output.records != 0 ||
                    ricci_output.records != 0))
      CCTK_VINFO("AH_analysis wrote %zu dynamic stable-ID surface "
                 "record(s), %zu Weinberg record(s), and %zu Ricci record(s) "
                 "at iteration %d",
                 output.records, weinberg_output.records,
                 ricci_output.records, cctk_iteration);
    if (print_analysis_timing)
      CCTK_VINFO(
          "AH_analysis performance: iteration=%d time=%.17g surfaces=%d "
          "records=%d total=%.6g s device-kernel/collective/d2h/host-record/io="
          "%.6g/%.6g/%.6g/%.6g/%.6g s; kernels=%d collectives=%d syncs=%d "
          "local-surfaces/points=%d/%lld axial-modes=%d multipole-degree=%d "
          "intrinsic/axial-kernels=%d/%d workspace-intrinsic/axial-point/"
          "axial-projected=%lld/%lld/%lld B "
          "weyl-gather/routing=%.6g/%.6g s weyl-kernels=%d/%d "
          "workspace-weyl-mesh/logical=%lld/%lld B weyl-payload=%lld B "
          "gauge-gather/routing=%.6g/%.6g s gauge-kernels=%d/%d "
          "workspace-gauge-mesh/logical=%lld/%lld B gauge-payload=%lld B "
          "matter-gather/routing=%.6g/%.6g s matter-kernels=%d/%d "
          "workspace-matter-mesh/logical=%lld/%lld B matter-payload=%lld B "
          "compact collective/stage/d2h/h2d=%lld/%lld/%lld/%lld B "
          "collective-stage-copies=%d TSV=%lld B angular-point-host=%lld B "
          "direct-device=%s",
          cctk_iteration, double(cctk_time),
          performance.global_surface_count,
          performance.returned_record_count,
          double(performance.total_seconds),
          double(performance.device_kernel_seconds),
          double(performance.collective_seconds),
          double(performance.compact_device_to_host_seconds),
          double(performance.host_record_seconds), io_seconds,
          performance.device_kernel_launches, performance.device_collectives,
          performance.explicit_device_synchronizations,
          performance.local_surface_count,
          static_cast<long long>(performance.local_angular_points),
          performance.axial_trial_modes,
          performance.multipole_maximum_degree,
          performance.intrinsic_geometry_kernel_launches,
          performance.axial_symmetry_kernel_launches,
          static_cast<long long>(performance.intrinsic_workspace_bytes),
          static_cast<long long>(performance.axial_point_workspace_bytes),
          static_cast<long long>(
              performance.axial_projected_workspace_bytes),
          double(performance.weyl_gather_seconds),
          double(performance.weyl_routing_seconds),
          performance.weyl_gather_kernel_launches,
          performance.weyl_routing_kernel_launches,
          static_cast<long long>(performance.weyl_mesh_workspace_bytes),
          static_cast<long long>(performance.weyl_logical_workspace_bytes),
          static_cast<long long>(performance.weyl_device_payload_bytes),
          double(performance.gauge_gather_seconds),
          double(performance.gauge_routing_seconds),
          performance.gauge_gather_kernel_launches,
          performance.gauge_routing_kernel_launches,
          static_cast<long long>(performance.gauge_mesh_workspace_bytes),
          static_cast<long long>(performance.gauge_logical_workspace_bytes),
          static_cast<long long>(performance.gauge_device_payload_bytes),
          double(performance.matter_gather_seconds),
          double(performance.matter_routing_seconds),
          performance.matter_gather_kernel_launches,
          performance.matter_routing_kernel_launches,
          static_cast<long long>(performance.matter_mesh_workspace_bytes),
          static_cast<long long>(performance.matter_logical_workspace_bytes),
          static_cast<long long>(performance.matter_device_payload_bytes),
          static_cast<long long>(performance.compact_collective_bytes),
          static_cast<long long>(
              performance.compact_collective_host_staging_bytes),
          static_cast<long long>(performance.compact_device_to_host_bytes),
          static_cast<long long>(performance.compact_host_to_device_bytes),
          performance.compact_collective_host_staging_copies,
          static_cast<long long>(output.bytes + weinberg_output.bytes +
                                 ricci_output.bytes),
          static_cast<long long>(performance.angular_point_host_bytes),
          performance.direct_device_collective ? "yes" : "no");
  }

  if (point_output_every > 0 &&
      cctk_iteration % point_output_every == 0) {
    // The rank-zero TSV append is the scalar commit decision. All ranks wait
    // for it before collectively publishing the matching point frame.
    CCTK_Barrier(cctkGH);
    std::vector<std::int64_t> stable_ids;
    stable_ids.reserve(records.size());
    for (const auto &record : records)
      stable_ids.push_back(record.stable_id);
    ParticleAHFinderX::SurfaceAnalysisPointOutputOptions point_options;
    point_options.directory = point_out_dir;
    point_options.backend = point_out_backend;
    point_options.chunk_points = point_output_chunk_points;
    point_options.invariant_spin_requested = compute_invariant_spin;
    point_options.weyl_curvature_requested = compute_weyl_curvature;
    ParticleAHFinderX::SurfaceAnalysisPointOutputPerformance point_performance;
    try {
      ParticleAHFinderX::write_current_surface_analysis_points(
          cctkGH, cctk_iteration, cctk_time, stable_ids, point_options,
          (print_analysis_timing || verbose) ? &point_performance : nullptr);
    } catch (const std::exception &error) {
      CCTK_VERROR("AH_analysis point output failed: %s", error.what());
    }
    if (CCTK_MyProc(cctkGH) == 0 && verbose &&
        point_performance.wrote_frame)
      CCTK_VINFO("AH_analysis wrote one GPU-native point frame for %d "
                 "stable-ID surface(s) and %lld point(s) at iteration %d",
                 point_performance.stable_id_count,
                 static_cast<long long>(point_performance.global_point_count),
                 cctk_iteration);
    if (CCTK_MyProc(cctkGH) == 0 && print_analysis_timing)
      CCTK_VINFO(
          "AH_analysis point-output performance: iteration=%d ids=%d "
          "points-local/global=%lld/%lld wrote=%s existing=%s "
          "selection/packing/d2h/output/total=%.6g/%.6g/%.6g/%.6g/%.6g s; "
          "selection/packing-kernels=%d/%d metadata-collectives=%d "
          "syncs=%d d2h-copies=%d workspace=%lld B pinned-host=%lld B "
          "chunk=%d",
          cctk_iteration, point_performance.stable_id_count,
          static_cast<long long>(point_performance.local_point_count),
          static_cast<long long>(point_performance.global_point_count),
          point_performance.wrote_frame ? "yes" : "no",
          point_performance.existing_frame ? "yes" : "no",
          double(point_performance.selection_seconds),
          double(point_performance.packing_seconds),
          double(point_performance.device_to_host_seconds),
          double(point_performance.output_seconds),
          double(point_performance.total_seconds),
          point_performance.selection_kernel_launches,
          point_performance.packing_kernel_launches,
          point_performance.metadata_collectives,
          point_performance.explicit_device_synchronizations,
          point_performance.device_to_host_copies,
          static_cast<long long>(point_performance.device_workspace_bytes),
          static_cast<long long>(point_performance.pinned_host_bytes),
          int(point_output_chunk_points));
  }
}

extern "C" int AH_analysis_Shutdown() {
  runtime() = Runtime{};
  return 0;
}

} // namespace AHAnalysis
