/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.basic;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import org.jooq.lambda.tuple.Tuple3;

import java.util.stream.Stream;

/**
 * Interactive version of a popularity-based algorithm.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class PopularityRecommender<U, I> extends AbstractBasicInteractiveRecommender<U, I>
{
    /**
     * Relevance threshold.
     */
    public final double threshold;

    /**
     * Constructor.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param threshold Relevance threshold
     */
    public PopularityRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, double threshold)
    {
        super(uIndex, iIndex, true);
        this.threshold = threshold;
    }

    @Override
    public void init()
    {
        this.iIndex.getAllIidx().forEach(iidx -> this.values[iidx] = 0.0);
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();
        values.filter(triplet -> triplet.value() >= threshold).forEach(triplet -> ++this.values[triplet.iidx()]);
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        this.values[iidx] += (value >= threshold ? 1.0 : 0.0);
    }
}
