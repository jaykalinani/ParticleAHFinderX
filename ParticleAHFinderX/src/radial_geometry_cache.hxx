/**
 * \file radial_geometry_cache.hxx
 * \brief Owner-GPU cache of frozen-slice geometry along angular rays.
 */
#ifndef PARTICLEAHFINDERX_RADIAL_GEOMETRY_CACHE_HXX
#define PARTICLEAHFINDERX_RADIAL_GEOMETRY_CACHE_HXX

#include "logical_surface_storage.hxx"
#include "surface_router.hxx"

#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <cstddef>
#include <memory>
#include <vector>

namespace ParticleAHFinderX {

class SurfaceParticleContainer;
struct RadialProbeWorkspace;

struct RadialCacheLayer {
  amrex::Long stable_id = 0;
  amrex::Long logical_offset = 0;
  amrex::Long point_count = 0;
  amrex::Long cache_offset = 0;
  int angular_level = 0;
};

struct RadialCacheComparison {
  amrex::Real maximum_absolute_error = 0;
  amrex::Real maximum_scaled_error = 0;
  amrex::Long invalid = 0;
};

/** One native-centered ADM level available to a radial probe batch. */
struct RadialCacheMeshLevel {
  int level = -1;
  const amrex::MultiFab *metric = nullptr;
  const amrex::MultiFab *curv = nullptr;
  int metric_component = 0;
  int curv_component = 0;
  FieldCentering centering;
};

/**
 * Reusable derived geometry for one solver batch.
 *
 * Derived probe records visit the radial nodes during construction without
 * changing the authoritative persistent AMReX particles. The cache then
 * supplies the 30-component ADM record directly on the logical owner while
 * the private pseudo-time state evolves.
 */
class RadialGeometryCache final {
public:
  RadialGeometryCache();
  ~RadialGeometryCache();

  RadialGeometryCache(const RadialGeometryCache &) = delete;
  RadialGeometryCache &operator=(const RadialGeometryCache &) = delete;

  void prepare(const LogicalSurfaceDeviceView &logical,
               const std::vector<RadialCacheLayer> &layers,
               int radial_nodes, amrex::Real half_width_factor,
               amrex::Real level_width_factor,
               amrex::Real minimum_radius_factor,
               amrex::Real maximum_radius_factor);
  /**
   * Sample every angular-ray/radial-node pair in one GPU-aware batch.
   *
   * Probe records are derived accelerator scratch. They are located and
   * routed to mesh-owning GPUs without moving, recreating, or otherwise
   * modifying the authoritative persistent surface particles.
   */
  SurfaceRoutingStats sample_batched(
      const LogicalSurfaceDeviceView &logical,
      SurfaceParticleContainer &container,
      const std::vector<RadialCacheMeshLevel> &levels,
      int minimum_active_level, int maximum_active_level,
      int interpolation_order);
  RadialCacheComparison
  compare_direct(const LogicalSurfaceDeviceView &logical) const;
  void interpolate(const LogicalSurfaceDeviceView &logical) const;
  void invalidate() { valid_ = false; }

  bool valid() const { return valid_; }
  int radial_nodes() const { return radial_nodes_; }
  amrex::Long point_count() const { return point_count_; }
  std::size_t allocated_bytes() const;

private:
  amrex::Gpu::DeviceVector<RadialCacheLayer> layers_;
  amrex::Gpu::DeviceVector<amrex::Long> logical_point_;
  amrex::Gpu::DeviceVector<amrex::Real> lower_radius_;
  amrex::Gpu::DeviceVector<amrex::Real> upper_radius_;
  amrex::Gpu::DeviceVector<amrex::Real> adm_;
  amrex::Gpu::DeviceVector<int> point_valid_;
  std::unique_ptr<RadialProbeWorkspace> probe_workspace_;
  amrex::Long point_count_ = 0;
  int radial_nodes_ = 0;
  bool valid_ = false;
};

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_RADIAL_GEOMETRY_CACHE_HXX
