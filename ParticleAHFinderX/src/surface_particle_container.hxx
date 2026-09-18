/**
 * \file surface_particle_container.hxx
 * \brief Persistent pure-SoA AMReX storage for horizon surface points.
 *
 * AMReX owns spatial position, mesh rank, and refinement-level migration.
 * Surface-specific attributes and lifecycle operations remain in this thorn.
 */
#ifndef PARTICLEAHFINDERX_SURFACE_PARTICLE_CONTAINER_HXX
#define PARTICLEAHFINDERX_SURFACE_PARTICLE_CONTAINER_HXX

#include "interpolation.hxx"

#include <AMReX_AmrParticles.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>

#include <array>
#include <memory>
#include <vector>

namespace ParticleAHFinderX {

enum class ParticleLifecycle : int { candidate = 0, active = 1, retired = 2 };
enum class SurfaceRole : int { individual = 0, common = 1, other = 2 };
enum class ParticleValidity : int {
  valid = 0,
  not_sampled = 1,
  needs_local_redistribute = 2,
  needs_hierarchy_redistribute = 3,
  outside_supported_domain = 4,
  invalid_ghost_data = 5,
  nonfinite = 6
};

enum class AdmGatherPass : int { initial = 0, local_retry = 1, final = 2 };

constexpr amrex::Long id_limb_bits = 30;
constexpr amrex::Long id_limb_mask =
    (amrex::Long(1) << id_limb_bits) - 1;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Long
surface_id_from_limbs(const int low, const int middle, const int high) noexcept {
  return amrex::Long(low) | (amrex::Long(middle) << id_limb_bits) |
         (amrex::Long(high) << (2 * id_limb_bits));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
long_to_limbs(const amrex::Long value, int &low, int &middle,
              int &high) noexcept {
  low = int(value & id_limb_mask);
  middle = int((value >> id_limb_bits) & id_limb_mask);
  high = int(value >> (2 * id_limb_bits));
}

// Pure-SoA positions occupy [0, AMREX_SPACEDIM). The remaining components are
// horizon state and stay in device storage between scheduled searches.
struct RealIdx {
  enum {
    height = AMREX_SPACEDIM,
    relaxation_velocity,
    expansion,
    normal,
    area_weight = normal + 3,
    gamma,
    d_gamma = gamma + 6,
    curv = d_gamma + 18,
    count = curv + 6
  };

  AMREX_GPU_HOST_DEVICE static constexpr int
  gamma_component(const int component) noexcept {
    return gamma + component;
  }

  AMREX_GPU_HOST_DEVICE static constexpr int
  d_gamma_component(const int d, const int component) noexcept {
    return d_gamma + 6 * d + component;
  }

  AMREX_GPU_HOST_DEVICE static constexpr int
  curv_component(const int component) noexcept {
    return curv + component;
  }

  AMREX_GPU_HOST_DEVICE static constexpr int
  normal_component(const int d) noexcept {
    return normal + d;
  }
};

// A stable 64-bit surface ID is stored as three non-negative base-2^30 limbs;
// this avoids assuming that AMReX compile-time integer attributes are 64 bit.
struct IntIdx {
  enum {
    surface_id_low = 0,
    surface_id_middle,
    surface_id_high,
    generation,
    angular_level,
    angular_levels,
    itheta,
    iphi,
    lifecycle,
    role,
    solver_active,
    ntheta,
    nphi,
    validity,
    sampled_level,
    sampled_level_changed,
    logical_owner_rank,
    logical_index_low,
    logical_index_middle,
    logical_index_high,
    count
  };
};

using ParticleType = amrex::SoAParticle<RealIdx::count, IntIdx::count>;
using BaseContainer =
    amrex::AmrParticleContainer_impl<ParticleType, RealIdx::count,
                                     IntIdx::count>;
using ParIter = amrex::ParIterSoA<RealIdx::count, IntIdx::count>;

struct SurfaceSeed {
  amrex::Long surface_id = 0;
  amrex::Long first_particle_id = 0;
  amrex::Long logical_owner_offset = 0;
  int generation = 0;
  int angular_level = 0;
  int angular_levels = 1;
  int logical_owner_rank = -1;
  int ntheta = 0;
  int nphi = 0;
  std::array<amrex::Real, 3> center{{0, 0, 0}};
  amrex::Real radius = 0;
  ParticleLifecycle lifecycle = ParticleLifecycle::active;
  SurfaceRole role = SurfaceRole::individual;

  amrex::Long particle_count() const;
};

struct AdmGatherStats {
  amrex::Long sampled = 0;
  amrex::Long not_sampled = 0;
  amrex::Long needs_local_redistribute = 0;
  amrex::Long needs_hierarchy_redistribute = 0;
  amrex::Long outside_supported_domain = 0;
  amrex::Long invalid_ghost_data = 0;
  amrex::Long nonfinite = 0;
};

struct PhysicalDomainBounds {
  std::array<amrex::Real, 3> lower{{0, 0, 0}};
  std::array<amrex::Real, 3> upper{{0, 0, 0}};
  std::array<int, 3> periodic{{0, 0, 0}};
};

struct LogicalOwnership {
  amrex::Long surface_id = 0;
  amrex::Long owner_offset = 0;
  int owner_rank = -1;
  int generation = 0;
  int angular_level = 0;
  int angular_levels = 1;
  int ntheta = 0;
  int nphi = 0;
};

struct SurfaceCenterUpdate {
  amrex::Long surface_id = 0;
  std::array<amrex::Real, 3> old_center{{0, 0, 0}};
  std::array<amrex::Real, 3> new_center{{0, 0, 0}};
};

struct SurfaceStateUpdate {
  amrex::Long surface_id = 0;
  ParticleLifecycle lifecycle = ParticleLifecycle::candidate;
  SurfaceRole role = SurfaceRole::other;
};

struct SurfaceSphereReset {
  amrex::Long surface_id = 0;
  std::array<amrex::Real, 3> center{{0, 0, 0}};
  amrex::Real radius = 0;
};

struct SurfaceLayerSelection {
  amrex::Long stable_id = 0;
  int angular_level = 0;
};

class SurfaceParticleContainer final : public BaseContainer {
public:
  explicit SurfaceParticleContainer(amrex::AmrCore *core);

  /** Create one new logical surface directly in AMReX device storage.
   *
   * This is a lifecycle operation, called only for an initial surface or a
   * newly born dynamic candidate. Finder iterations must update the resulting
   * particles in place and must never call this method.
   */
  void create_surface(const SurfaceSeed &seed);

  /// Batch a quiescent set of births into one portable device population pass.
  void create_surfaces(const std::vector<SurfaceSeed> &seeds);

  /// Rigidly translate persistent surface populations on the device.
  void update_surface_centers(
      const std::vector<SurfaceCenterUpdate> &updates);
  void update_surface_states(const std::vector<SurfaceStateUpdate> &updates);
  void reset_surface_spheres(const std::vector<SurfaceSphereReset> &resets);

  /// Remove explicitly retired populations; stable IDs remain in metadata.
  amrex::Long retire_surfaces(const std::vector<amrex::Long> &stable_ids);

  /// Rebuild hierarchy metadata without destroying the particle population.
  void resize_after_regrid();
  /// Cache masks that exclude coarse cells covered by the next finer level.
  void rebuild_amr_masks();
  /// Locate every live particle on its finest valid AMR owner after creation.
  void redistribute_global();
  /// Migrate nearby particles after bounded motion at synchronized times.
  void redistribute_local_hierarchy(int max_num_cells_moved);

  /** Replace checkpointed logical-owner metadata for the current MPI layout.
   *
   * Stable particle/surface IDs and immutable angular indices are preserved.
   * Only the rank and owner-local logical offset are decomposition-dependent.
   */
  std::vector<amrex::Long> remap_logical_ownership(
      const std::vector<LogicalOwnership> &ownership);

  /// Select the persistent layer populations used by the current solve pass.
  void set_active_surface_layers(
      const std::vector<SurfaceLayerSelection> &selections);

  /// Mark every particle as awaiting the current synchronized ADM gather.
  void begin_adm_gather();
  /// Start one search-wide AMR routing diagnostic window on the device.
  void begin_sampled_level_tracking();

  /** Sample native-centered ADM data into persistent device particle state. */
  void gather_adm_level(int lev, const amrex::MultiFab &metric,
                        int metric_component, const amrex::MultiFab &curv,
                        int curv_component, FieldCentering centering,
                        int interpolation_order, AdmGatherPass pass);

  /// Compact device reduction of the most recent gather statuses.
  AdmGatherStats local_adm_gather_stats() const;

  /// Maximum error against Cartesian Minkowski ADM data, for validation only.
  amrex::Real local_minkowski_adm_error() const;

  /// Physical level-zero bounds used to preflight rigid surface motion.
  PhysicalDomainBounds physical_domain_bounds() const;

  bool has_level_zero_tile() const;
  amrex::Long local_particle_count() const;
  /// Number of live particles on configured target angular layers.
  amrex::Long local_target_particle_count() const;

private:
  std::vector<std::unique_ptr<amrex::iMultiFab>> fine_masks_;
};

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_SURFACE_PARTICLE_CONTAINER_HXX
