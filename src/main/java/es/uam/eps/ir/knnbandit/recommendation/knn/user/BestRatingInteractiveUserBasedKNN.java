/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.knn.user;

import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.BestRatingFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;

import java.util.stream.Stream;

/**
 * Interactive version of user-based kNN algorithm.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class BestRatingInteractiveUserBasedKNN<U, I> extends AbstractInteractiveUserBasedKNN<U, I>
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
    public BestRatingInteractiveUserBasedKNN(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, boolean ignoreZeros, int k, UpdateableSimilarity sim)
    {
        super(uIndex, iIndex, hasRating, ignoreZeros, k, sim, BestRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex));
    }

    @Override
    protected double score(int vidx, double rating)
    {
        return rating;
    }
}
