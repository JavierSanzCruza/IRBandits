/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.bandits;

import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.MultiArmedBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.selector.algorithms.bandit.BanditSupplier;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.stream.Stream;

/**
 * For each user in the system, it uses a different (although context-less) multi-armed bandit
 * for recommending the items in the system.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class UserMultiArmedBanditRecommender<U, I> extends AbstractInteractiveRecommender<U, I>
{
    /**
     * Implementation of an item bandit.
     */
    private final MultiArmedBandit[] multiArmedBandit;
    /**
     * Function for evaluating the value.
     */
    private final ValueFunction valFunc;

    /**
     * Constructor.
     *
     * @param uIndex        User index
     * @param iIndex        Item index.
     * @param ignoreNotRated True if we want to ignore missing ratings when updating, false if we want to count them as failures.
     * @param mabFunc       A function to obtain a multi-armed bandit.
     * @param valFunc       A value function of the reward.
     */
    public UserMultiArmedBanditRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, BanditSupplier mabFunc, ValueFunction valFunc)
    {
        super(uIndex, iIndex, ignoreNotRated);
        int numUsers = uIndex.numUsers();
        this.multiArmedBandit = new MultiArmedBandit[numUsers];
        for(int i = 0; i < numUsers; ++i)
            this.multiArmedBandit[i] = mabFunc.apply(iIndex.numItems());
        this.valFunc = valFunc;
    }

    @Override
    public void init()
    {
        super.init();
        uIndex.getAllUidx().forEach(uidx -> this.multiArmedBandit[uidx].reset());
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();
        values.forEach(triplet -> this.multiArmedBandit[triplet.uidx()].update(triplet.iidx(), triplet.value()));
    }

    @Override
    public void fastUpdate(int uidx, int iidx, double value)
    {
        if(!Double.isNaN(value) && value != Constants.NOTRATEDRATING) // If the (uidx, iidx) pair exists.
        {
            this.multiArmedBandit[uidx].update(iidx, value);
        }
        else if(!this.ignoreNotRated) // If we update the bandit even when the (uidx, iidx) pair does not exist.
        {
            this.multiArmedBandit[uidx].update(iidx, Constants.NOTRATEDNOTIGNORED);
        }
    }

    @Override
    public int next(int uidx, IntList availability)
    {
        return this.multiArmedBandit[uidx].next(availability.toIntArray(), valFunc);
    }

    @Override
    public IntList next(int uidx, IntList availability, int k)
    {
        return this.multiArmedBandit[uidx].next(availability, valFunc, k);
    }
}
