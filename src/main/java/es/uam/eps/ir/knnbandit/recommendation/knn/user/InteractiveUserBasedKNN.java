/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.knn.user;

import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;

import java.util.stream.Stream;

/**
 * Interactive version of user-based kNN algorithm. This is the legacy version,
 * used for offline recommendation with a classical dataset.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class InteractiveUserBasedKNN<U, I> extends AbstractInteractiveUserBasedKNN<U, I>
{
    /**
     * Constructor.
     *
     * @param uIndex      User index.
     * @param iIndex      Item index.
     * @param hasRating   True if we must ignore unknown items when updating.
     * @param ignoreZeros True if we ignore zero ratings when updating.
     * @param k           Number of neighbors to use.
     * @param sim         Updateable similarity
     */
    public InteractiveUserBasedKNN(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, boolean ignoreZeros, int k, UpdateableSimilarity sim)
    {
        super(uIndex, iIndex, hasRating, ignoreZeros, k, sim, SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex));
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        this.sim.updateNorm(uidx, value);
        this.retrievedData.getIidxPreferences(iidx).forEach(vidx -> this.sim.update(uidx, vidx.v1, iidx, value, vidx.v2));
        this.retrievedData.updateRating(uidx, iidx, value);
    }

    @Override
    protected double score(int vidx, double rating)
    {
        return rating;
    }
}
