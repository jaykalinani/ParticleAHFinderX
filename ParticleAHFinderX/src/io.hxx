/**
 * \file io.hxx
 * \brief Restart reconciliation for persistent-particle output.
 */
#ifndef PARTICLEAHFINDERX_IO_HXX
#define PARTICLEAHFINDERX_IO_HXX

namespace ParticleAHFinderX {

/**
 * Remove abandoned temporary frames and committed frames newer than the
 * checkpointed particle-output watermark.
 *
 * Only strictly recognized ParticleAHFinderX frame names are removed.  The
 * operation is collective over the active AMReX communicator.
 */
void reconcile_particle_output(const char *directory, const char *backend,
                               int last_committed_iteration);

} // namespace ParticleAHFinderX

#endif // PARTICLEAHFINDERX_IO_HXX
