package es.uam.eps.ir.knnbandit.recommendation.mf.ictr;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.mf.Particle;
import es.uam.eps.ir.knnbandit.recommendation.mf.ictr.particles.ICTRParticle;
import es.uam.eps.ir.knnbandit.recommendation.mf.ictr.particles.ICTRParticleFactory;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;

/**
 * UCB variant of the ICTR recommender.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public class UCBICTRRecommender<U, I> extends ICTRRecommender<U, I>
{
    /**
     * Parameter that indicates the amplitude of the upper confidence bound.
     */
    private final double gamma;

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
     * @param gamma        Estimates the importance of the UCB.
     */
    public UCBICTRRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRating, int K, int numParticles, ICTRParticleFactory<U,I> factory, double gamma)
    {
        super(uIndex, iIndex, prefData, hasRating, K, numParticles, factory);
        this.gamma = gamma;
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
     * @param gamma         Estimates the importance of the UCB.
     */
    public UCBICTRRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRating, boolean notReciprocal, int K, int numParticles, ICTRParticleFactory<U,I> factory, double gamma)
    {
        super(uIndex, iIndex, prefData, hasRating, notReciprocal, K, numParticles, factory);
        this.gamma = gamma;
    }

    @Override
    protected double getEstimatedReward(int uidx, int iidx)
    {
        double average = 0.0;
        double averageVar = 0.0;
        int counter = 0;
        for (Particle<U, I> particle : particles)
        {
            double reward = particle.getEstimatedReward(uidx, iidx);
            double var = ((ICTRParticle<U, I>) particle).getVariance(iidx);
            average += reward;
            averageVar += var;
            ++counter;
        }

        average /= (counter + 0.0);
        averageVar /= (counter + 0.0);

        return average + gamma * Math.sqrt(averageVar);

    }
}
