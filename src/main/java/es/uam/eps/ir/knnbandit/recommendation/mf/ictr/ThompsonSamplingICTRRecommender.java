package es.uam.eps.ir.knnbandit.recommendation.mf.ictr;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.mf.ictr.particles.ICTRParticleFactory;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;

/**
 * Thompson variant of the ICTR algorithm.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public class ThompsonSamplingICTRRecommender<U, I> extends ICTRRecommender<U, I>
{
    /**
     * Constructor.
     *
     * @param uIndex       User index.
     * @param iIndex       Item index.
     * @param prefData     Preference data.
     * @param hasRating    True if we must ignore unknown items when updating.
     * @param K            Number of latent factors to use.
     * @param numParticles Number of particles to use.
     * @param factory      A factory for the particles.
     */
    public ThompsonSamplingICTRRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRating, int K, int numParticles, ICTRParticleFactory<U,I> factory)
    {
        super(uIndex, iIndex, prefData, hasRating, K, numParticles, factory);
    }

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      Preference data.
     * @param hasRating     True if we must ignore unknown items when updating.
     * @param notReciprocal True if reciprocal users can be recommended, false otherwise.
     * @param K             Number of latent factors to use.
     * @param numParticles  Number of particles to use.
     * @param factory       A factory for the particles.
     */
    public ThompsonSamplingICTRRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRating, boolean notReciprocal, int K, int numParticles, ICTRParticleFactory<U,I> factory)
    {
        super(uIndex, iIndex, prefData, hasRating, notReciprocal, K, numParticles, factory);
    }

    @Override
    protected double getEstimatedReward(int uidx, int iidx)
    {
        return this.particles.stream().mapToDouble(particle -> particle.getEstimatedReward(uidx, iidx)).average().getAsDouble();
    }
}
