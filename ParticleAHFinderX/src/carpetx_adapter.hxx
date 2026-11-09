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
#include <cstdint>
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

/** Native-centered lapse/shift fields used only by four-dimensional analysis. */
struct GaugeFieldLayout {
  const amrex::MultiFab *lapse = nullptr;
  const amrex::MultiFab *shift = nullptr;
  const amrex::MultiFab *dt_lapse = nullptr;
  const amrex::MultiFab *dt_shift = nullptr;
  int lapse_component = 0;
  int shift_component = 0;
  int dt_lapse_component = 0;
  int dt_shift_component = 0;
  FieldCentering centering;
};

/** Native-centered covariant stress-energy fields used by Ricci analysis. */
struct MatterFieldLayout {
  const amrex::MultiFab *tt = nullptr;
  const amrex::MultiFab *ti = nullptr;
  const amrex::MultiFab *ij = nullptr;
  int tt_component = 0;
  int ti_component = 0;
  int ij_component = 0;
  FieldCentering centering;
};

struct ActiveLevelBand {
  int minimum_level = 0;
  int maximum_level = 0;
  bool complete_hierarchy = false;
  std::int64_t logical_iteration_numerator = 0;
  std::int64_t logical_iteration_denominator = 1;

  bool contains(const int level) const noexcept {
    return level >= minimum_level && level < maximum_level;
  }
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

/** Resolve alpha, beta^i, and their coordinate-time derivatives. */
GaugeFieldLayout gauge_field_layout(
    const CarpetX::GHExt::PatchData::LevelData &level,
    const std::array<MeshFieldRef, 8> &fields);

/** Resolve the ten covariant TmunuBaseX components without recentering. */
MatterFieldLayout matter_field_layout(
    const CarpetX::GHExt::PatchData::LevelData &level,
    const std::array<MeshFieldRef, 10> &fields);

/// Synchronize only groups whose CarpetX validity lacks outer/ghost data.
void ensure_mesh_fields_ready(const cGH *cctkGH,
                              const std::vector<MeshFieldRef> &fields);

/// Return the synchronized CarpetX level band traversing the current schedule.
ActiveLevelBand active_level_band();

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_CARPETX_ADAPTER_HXX
