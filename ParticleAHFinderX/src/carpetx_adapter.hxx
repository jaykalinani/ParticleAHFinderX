/**
 * \file carpetx_adapter.hxx
 * \brief Narrow checked access to CarpetX mesh storage.
 */
#ifndef PARTICLEAHFINDERX_CARPETX_ADAPTER_HXX
#define PARTICLEAHFINDERX_CARPETX_ADAPTER_HXX

#include "interpolation.hxx"

#include <driver.hxx>
#include <schedule.hxx>

#include <AMReX_MultiFab.H>

#include <array>
#include <string>
#include <vector>

namespace ParticleAHFinderX {

struct MeshFieldRef {
  int var_index = -1;
  int group_index = -1;
  int component_index = -1;
  std::string name;

  explicit operator bool() const noexcept { return var_index >= 0; }
};

struct AdmFieldLayout {
  const amrex::MultiFab *metric = nullptr;
  const amrex::MultiFab *curv = nullptr;
  int metric_component = 0;
  int curv_component = 0;
  FieldCentering centering;
};

/// Resolve a qualified three-dimensional CCTK_REAL grid function.
MeshFieldRef resolve_mesh_field(const std::string &name);

const CarpetX::GHExt::PatchData::LevelData::GroupData &
group_data(const CarpetX::GHExt::PatchData::LevelData &level,
           const MeshFieldRef &field);

const amrex::MultiFab &field_mfab(
    const CarpetX::GHExt::PatchData::LevelData &level,
    const MeshFieldRef &field, int timelevel = 0);

FieldCentering field_centering(
    const CarpetX::GHExt::PatchData::LevelData &level,
    const MeshFieldRef &field);

/**
 * Resolve the six contiguous metric and curvature components on one level.
 *
 * The fields remain at their native CarpetX centering. Version one's fused
 * gather requires metric and curvature to share that centering; it never
 * creates a vertex-to-cell copy.
 */
AdmFieldLayout adm_field_layout(
    const CarpetX::GHExt::PatchData::LevelData &level,
    const std::array<MeshFieldRef, 12> &fields);

/// Synchronize only groups whose CarpetX validity lacks outer/ghost data.
void ensure_mesh_fields_ready(const cGH *cctkGH,
                              const std::vector<MeshFieldRef> &fields);

bool hierarchy_is_time_aligned();

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_CARPETX_ADAPTER_HXX
