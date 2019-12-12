package es.uam.eps.ir.knnbandit.recommendation.mf.ictr.particles;

import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;

/**
 * Particle factory for the ICTR algorithm.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public class ICTRParticleFactory<U, I>
{
    /**
     * Creates a new particle.
     *
     * @param uIndex user index.
     * @param iIndex item index.
     * @param K      the number of latent factors for users/items.
     *
     * @return the created particle if everything went OK, null otherwise.
     */
    public ICTRParticle<U, I> create(FastUserIndex<U> uIndex, FastItemIndex<I> iIndex, int K)
    {
        return new ICTRParticle<>(uIndex, iIndex, K);
    }
}
