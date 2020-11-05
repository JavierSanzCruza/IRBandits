/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.basic;

import es.uam.eps.ir.knnbandit.UntieRandomNumber;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;

import java.util.List;
import java.util.Random;
import java.util.stream.Stream;


/**
 * Interactive version of a random algorithm.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RandomRecommender<U, I> extends InteractiveRecommender<U, I>
{
    /**
     * Random number generator.
     */
    private final Random rng = new Random(UntieRandomNumber.RNG);

    /**
     * Constructor.
     *
     * @param uIndex    user index.
     * @param iIndex    item index.
     */
    public RandomRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex)
    {
        super(uIndex, iIndex, true);
    }

    /**
     * Constructor.
     *
     * @param uIndex    user index.
     * @param iIndex    item index.
     */
    public RandomRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, int rngSeed)
    {
        super(uIndex, iIndex, true, rngSeed);
    }


    @Override
    public void init()
    {
        super.init();
        // It is not necessary to do nothing here.
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        super.init();
    }

    @Override
    public int next(int uidx, IntList availability)
    {
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }
        else
        {
            return availability.get(rng.nextInt(availability.size()));
        }
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {

    }

    @Override
    public void update(List<Tuple3<Integer, Integer, Double>> train)
    {

    }

}
