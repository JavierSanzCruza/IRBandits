/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
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
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
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
     * @param hasRating    True if we must ignore unknown items when updating.
     * @param K            Number of latent factors to use.
     * @param numParticles Number of particles to use.
     * @param factory      A factory for the particles.
     * @param gamma        Estimates the importance of the UCB.
     */
    public UCBICTRRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int K, int numParticles, ICTRParticleFactory<U,I> factory, double gamma)
    {
        super(uIndex, iIndex, hasRating, K, numParticles, factory);
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
