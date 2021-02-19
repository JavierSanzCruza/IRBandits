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
import es.uam.eps.ir.knnbandit.recommendation.mf.ictr.particles.ICTRParticleFactory;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;

import java.util.stream.Stream;

/**
 * Thompson variant of the ICTR algorithm.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ThompsonSamplingICTRRecommender<U, I> extends ICTRRecommender<U, I>
{
    /**
     * Constructor.
     *
     * @param uIndex       User index.
     * @param iIndex       Item index.
     * @param hasRating    True if we must ignore unknown items when updating.
     * @param K            Number of latent factors to use.
     * @param numParticles Number of particles to use.
     * @param factory      A factory for the particles.
     */
    public ThompsonSamplingICTRRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int K, int numParticles, ICTRParticleFactory<U,I> factory)
    {
        super(uIndex, iIndex, hasRating, K, numParticles, factory);
    }

    @Override
    protected double getEstimatedReward(int uidx, int iidx)
    {
        return this.particles.stream().mapToDouble(particle -> particle.getEstimatedReward(uidx, iidx)).average().getAsDouble();
    }


}
