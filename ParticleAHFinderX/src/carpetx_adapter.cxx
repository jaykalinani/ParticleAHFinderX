/**
 * \file carpetx_adapter.cxx
 * \brief Checked CarpetX field resolution, storage, centering, and validity.
 *
 * This compatibility boundary remains separate from the numerical solver so
 * CarpetX-internal changes stay localized.
 */
#include "carpetx_adapter.hxx"

#include <cctk.h>

#include <optional>
#include <set>
#include <stdexcept>

namespace ParticleAHFinderX {

MeshFieldRef resolve_mesh_field(const std::string &name) {
  if (name.empty())
    throw std::runtime_error(
        "ParticleAHFinderX received an empty mesh-variable name");

  const int var_index = CCTK_VarIndex(name.c_str());
  if (var_index < 0)
    throw std::runtime_error("ParticleAHFinderX cannot resolve mesh field \"" +
                             name + "\"");
  const int group_index = CCTK_GroupIndexFromVarI(var_index);
  if (group_index < 0)
    throw std::runtime_error(
        "ParticleAHFinderX cannot resolve the group for \"" + name + "\"");

  cGroup group;
  if (CCTK_GroupData(group_index, &group) != 0 ||
      group.grouptype != CCTK_GF || group.vartype != CCTK_VARIABLE_REAL ||
      group.dim != 3)
    throw std::runtime_error(
        "ParticleAHFinderX mesh field \"" + name +
        "\" must be a three-dimensional CCTK_REAL grid function");

  const int first_var_index = CCTK_FirstVarIndexI(group_index);
  return {var_index, group_index, var_index - first_var_index, name};
}

const CarpetX::GHExt::PatchData::LevelData::GroupData &
group_data(const CarpetX::GHExt::PatchData::LevelData &level,
           const MeshFieldRef &field) {
  if (!field || field.group_index >= static_cast<int>(level.groupdata.size()) ||
      !level.groupdata.at(field.group_index))
    throw std::runtime_error(
        "ParticleAHFinderX mesh storage is unavailable for \"" + field.name +
        "\"");
  return *level.groupdata.at(field.group_index);
}

const amrex::MultiFab &field_mfab(
    const CarpetX::GHExt::PatchData::LevelData &level,
    const MeshFieldRef &field, const int timelevel) {
  const auto &data = group_data(level, field);
  if (timelevel < 0 || timelevel >= static_cast<int>(data.mfab.size()) ||
      !data.mfab.at(timelevel))
    throw std::runtime_error(
        "ParticleAHFinderX time-level storage is unavailable for \"" +
        field.name + "\"");
  return *data.mfab.at(timelevel);
}

FieldCentering field_centering(
    const CarpetX::GHExt::PatchData::LevelData &level,
    const MeshFieldRef &field) {
  const auto &carpetx_centering = group_data(level, field).indextype;
  FieldCentering centering;
  for (int d = 0; d < 3; ++d)
    // CarpetX: 0=vertex, 1=cell. AMReX: 1=node, 0=cell.
    centering.nodal[d] = 1 - carpetx_centering[d];
  return centering;
}

namespace {

bool same_centering(const FieldCentering lhs,
                    const FieldCentering rhs) noexcept {
  for (int d = 0; d < 3; ++d)
    if (lhs.nodal[d] != rhs.nodal[d])
      return false;
  return true;
}

} // namespace

AdmFieldLayout adm_field_layout(
    const CarpetX::GHExt::PatchData::LevelData &level,
    const std::array<MeshFieldRef, 12> &fields) {
  const int metric_group = fields[0].group_index;
  const int curv_group = fields[6].group_index;
  const int metric_component = fields[0].component_index;
  const int curv_component = fields[6].component_index;
  for (int component = 0; component < 6; ++component) {
    if (fields[component].group_index != metric_group ||
        fields[component].component_index != metric_component + component)
      throw std::runtime_error(
          "ParticleAHFinderX requires the six ADM metric variables to be "
          "contiguous components of one CarpetX group");
    if (fields[6 + component].group_index != curv_group ||
        fields[6 + component].component_index != curv_component + component)
      throw std::runtime_error(
          "ParticleAHFinderX requires the six ADM curvature variables to be "
          "contiguous components of one CarpetX group");
  }

  const auto &metric = field_mfab(level, fields[0]);
  const auto &curv = field_mfab(level, fields[6]);
  if (metric.boxArray() != curv.boxArray() ||
      metric.DistributionMap() != curv.DistributionMap())
    throw std::runtime_error(
        "ParticleAHFinderX ADM metric and curvature MultiFabs do not share "
        "one mesh layout");
  if (metric_component < 0 || curv_component < 0 ||
      metric.nComp() < metric_component + 6 ||
      curv.nComp() < curv_component + 6)
    throw std::runtime_error(
        "ParticleAHFinderX ADM MultiFab component layout is incomplete");

  const FieldCentering centering = field_centering(level, fields[0]);
  for (int component = 1; component < 6; ++component)
    if (!same_centering(centering, field_centering(level, fields[component])))
      throw std::runtime_error(
          "ParticleAHFinderX ADM metric components have mixed centerings");
  for (int component = 0; component < 6; ++component)
    if (!same_centering(
            centering, field_centering(level, fields[6 + component])))
      throw std::runtime_error(
          "ParticleAHFinderX ADM metric and curvature centerings differ; "
          "the fused gather will not average either field to another "
          "centering");

  return {&metric, &curv, metric_component, curv_component, centering};
}

GaugeFieldLayout gauge_field_layout(
    const CarpetX::GHExt::PatchData::LevelData &level,
    const std::array<MeshFieldRef, 8> &fields) {
  const int lapse_component = fields[0].component_index;
  const int shift_component = fields[1].component_index;
  const int dt_lapse_component = fields[4].component_index;
  const int dt_shift_component = fields[5].component_index;
  for (int component = 0; component < 3; ++component) {
    if (fields[1 + component].group_index != fields[1].group_index ||
        fields[1 + component].component_index !=
            shift_component + component)
      throw std::runtime_error(
          "ParticleAHFinderX requires the three ADM shift variables to be "
          "contiguous components of one CarpetX group");
    if (fields[5 + component].group_index != fields[5].group_index ||
        fields[5 + component].component_index !=
            dt_shift_component + component)
      throw std::runtime_error(
          "ParticleAHFinderX requires the three ADM shift-time-derivative "
          "variables to be contiguous components of one CarpetX group");
  }

  const auto &lapse = field_mfab(level, fields[0]);
  const auto &shift = field_mfab(level, fields[1]);
  const auto &dt_lapse = field_mfab(level, fields[4]);
  const auto &dt_shift = field_mfab(level, fields[5]);
  const amrex::MultiFab *const multifabs[4]{&lapse, &shift, &dt_lapse,
                                            &dt_shift};
  for (int field = 1; field < 4; ++field)
    if (multifabs[field]->boxArray() != lapse.boxArray() ||
        multifabs[field]->DistributionMap() != lapse.DistributionMap())
      throw std::runtime_error(
          "ParticleAHFinderX ADM gauge fields do not share one mesh layout");
  if (lapse_component < 0 || shift_component < 0 ||
      dt_lapse_component < 0 || dt_shift_component < 0 ||
      lapse.nComp() <= lapse_component ||
      shift.nComp() < shift_component + 3 ||
      dt_lapse.nComp() <= dt_lapse_component ||
      dt_shift.nComp() < dt_shift_component + 3)
    throw std::runtime_error(
        "ParticleAHFinderX ADM gauge MultiFab component layout is "
        "incomplete");

  const FieldCentering centering = field_centering(level, fields[0]);
  for (int field = 1; field < 8; ++field)
    if (!same_centering(centering, field_centering(level, fields[field])))
      throw std::runtime_error(
          "ParticleAHFinderX ADM gauge fields have mixed centerings; the "
          "four-dimensional gather will not average them");

  return {&lapse,          &shift,
          &dt_lapse,       &dt_shift,
          lapse_component, shift_component,
          dt_lapse_component, dt_shift_component,
          centering};
}

MatterFieldLayout matter_field_layout(
    const CarpetX::GHExt::PatchData::LevelData &level,
    const std::array<MeshFieldRef, 10> &fields) {
  const int tt_component = fields[0].component_index;
  const int ti_component = fields[1].component_index;
  const int ij_component = fields[4].component_index;
  for (int component = 0; component < 3; ++component)
    if (fields[1 + component].group_index != fields[1].group_index ||
        fields[1 + component].component_index != ti_component + component)
      throw std::runtime_error(
          "ParticleAHFinderX requires the three TmunuBaseX T_0i variables "
          "to be contiguous components of one CarpetX group");
  for (int component = 0; component < 6; ++component)
    if (fields[4 + component].group_index != fields[4].group_index ||
        fields[4 + component].component_index != ij_component + component)
      throw std::runtime_error(
          "ParticleAHFinderX requires the six TmunuBaseX T_ij variables "
          "to be contiguous components of one CarpetX group");

  const auto &tt = field_mfab(level, fields[0]);
  const auto &ti = field_mfab(level, fields[1]);
  const auto &ij = field_mfab(level, fields[4]);
  const amrex::MultiFab *const multifabs[3]{&tt, &ti, &ij};
  for (int field = 1; field < 3; ++field)
    if (multifabs[field]->boxArray() != tt.boxArray() ||
        multifabs[field]->DistributionMap() != tt.DistributionMap())
      throw std::runtime_error(
          "ParticleAHFinderX TmunuBaseX fields do not share one mesh layout");
  if (tt_component < 0 || ti_component < 0 || ij_component < 0 ||
      tt.nComp() <= tt_component || ti.nComp() < ti_component + 3 ||
      ij.nComp() < ij_component + 6)
    throw std::runtime_error(
        "ParticleAHFinderX TmunuBaseX MultiFab component layout is "
        "incomplete");

  const FieldCentering centering = field_centering(level, fields[0]);
  for (int field = 1; field < 10; ++field)
    if (!same_centering(centering, field_centering(level, fields[field])))
      throw std::runtime_error(
          "ParticleAHFinderX TmunuBaseX fields have mixed centerings; the "
          "matter gather will not average them");

  return {&tt, &ti, &ij, tt_component, ti_component, ij_component,
          centering};
}

void ensure_mesh_fields_ready(const cGH *const cctkGH,
                              const std::vector<MeshFieldRef> &fields) {
  if (fields.empty())
    return;
  if (!CarpetX::active_levels)
    throw std::runtime_error(
        "ParticleAHFinderX has no active CarpetX levels for field sync");

  std::set<int> groups_to_sync;
  for (const auto &field : fields) {
    bool needs_sync = false;
    CarpetX::active_levels->loop_serially([&](const auto &level) {
      const auto &data = group_data(level, field);
      const auto validity =
          data.valid.at(0).at(field.component_index).get();
      if (!validity.valid_int)
        throw std::runtime_error(
            "ParticleAHFinderX interior data are invalid for \"" + field.name +
            "\"");
      needs_sync |= !validity.valid_outer || !validity.valid_ghosts;
    });
    if (needs_sync)
      groups_to_sync.insert(field.group_index);
  }

  if (!groups_to_sync.empty()) {
    const std::vector<int> groups(groups_to_sync.begin(),
                                  groups_to_sync.end());
    if (CarpetX::SyncGroupsByDirI(cctkGH, static_cast<int>(groups.size()),
                                 groups.data(), nullptr) < 0)
      throw std::runtime_error(
          "ParticleAHFinderX could not synchronize an ADM mesh field");
  }
}

ActiveLevelBand active_level_band() {
  if (!CarpetX::active_levels)
    throw std::runtime_error(
        "ParticleAHFinderX analysis has no active CarpetX levels");
  const int minimum_level = CarpetX::active_levels->min_level;
  const int maximum_level = CarpetX::active_levels->max_level;
  if (minimum_level < 0 || maximum_level <= minimum_level ||
      maximum_level > CarpetX::ghext->num_levels())
    throw std::runtime_error(
        "ParticleAHFinderX received an invalid CarpetX active-level band");

  // CarpetX constructs an analysis band only after every member has caught up
  // to one level iteration. Check that invariant explicitly here rather than
  // relying on the driver's debug-only assertion: sampling levels at different
  // logical times would mix geometry from different slices.
  bool visited = false;
  std::optional<CarpetX::rat64> reference_iteration;
  CarpetX::active_levels->loop_serially([&](const auto &level) {
    visited = true;
    if (!reference_iteration)
      reference_iteration = level.iteration;
    else if (level.iteration != *reference_iteration)
      throw std::runtime_error(
          "ParticleAHFinderX active-level band contains levels at different "
          "logical times");
  });
  if (!visited)
    throw std::runtime_error(
        "ParticleAHFinderX found an empty CarpetX active-level band");

  if (!reference_iteration || reference_iteration->den <= 0)
    throw std::runtime_error(
        "ParticleAHFinderX received an invalid CarpetX level iteration");

  return {minimum_level,
          maximum_level,
          minimum_level == 0 &&
              maximum_level == CarpetX::ghext->num_levels() &&
              CarpetX::all_levels_synchronized(),
          reference_iteration->num,
          reference_iteration->den};
}

} // namespace ParticleAHFinderX
