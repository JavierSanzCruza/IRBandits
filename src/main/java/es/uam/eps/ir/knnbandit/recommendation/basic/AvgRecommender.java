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

import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Interactive version of an average rating recommendation algorithm.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class AvgRecommender<U, I> extends AbstractBasicInteractiveRecommender<U, I>
{
    /**
     * Number of times an arm has been selected.
     */
    private double[] numTimes;

    /**
     * Constructor.
     *
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     */
    public AvgRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated)
    {
        super(uIndex, iIndex, ignoreNotRated);
        this.numTimes = new double[iIndex.numItems()];
        IntStream.range(0, iIndex.numItems()).forEach(iidx -> this.numTimes[iidx] = 0);
    }

    @Override
    public void init()
    {
        IntStream.range(0, iIndex.numItems()).forEach(iidx ->
        {
            this.numTimes[iidx] = 0.0;
            this.values[iidx] = 0.0;
        });
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();
        values.forEach(triplet ->
        {
            int iidx = triplet.iidx();
            double oldvalue = this.values[iidx];
            this.values[iidx] = oldvalue + (triplet.value() - oldvalue) / (numTimes[iidx] + 1.0);
            this.numTimes[iidx] += 1.0;
        });
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        double oldValue = values[iidx];
        if (numTimes[iidx] <= 0.0)
        {
            this.values[iidx] = value;
        }
        else
        {
            this.values[iidx] = oldValue + (value - oldValue) / (numTimes[iidx] + 1.0);
        }
        this.numTimes[iidx]++;
    }
}
