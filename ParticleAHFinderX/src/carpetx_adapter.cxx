/**
 * \file carpetx_adapter.cxx
 * \brief Checked CarpetX field resolution, storage, centering, and validity.
 *
 * This compatibility boundary remains separate from the numerical solver so
 * CarpetX-internal changes stay localized.
 */
#include "carpetx_adapter.hxx"

#include <cctk.h>

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

bool hierarchy_is_time_aligned() {
  return CarpetX::all_levels_synchronized();
}

} // namespace ParticleAHFinderX
