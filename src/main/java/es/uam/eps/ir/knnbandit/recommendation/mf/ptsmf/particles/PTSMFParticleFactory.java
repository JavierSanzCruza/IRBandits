package es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles;

import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;

/**
 * Particle factory for the Particle Thompson Sampling algorithm.
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public interface PTSMFParticleFactory<U, I>
{
    public PTSMFParticle<U, I> create(FastUserIndex<U> uIndex, FastItemIndex<I> iIndex);
}
