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
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.recommendation.bandits.item.ItemBandit;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;

import java.util.stream.Stream;

/**
 * Simple non-personalized item-based multi-armed bandit recommender.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ItemBanditRecommender<U, I> extends InteractiveRecommender<U, I>
{
    /**
     * Implementation of an item bandit.
     */
    private final ItemBandit<U, I> itemBandit;
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
     * @param itemBandit    An item bandit.
     * @param valFunc       A value function of the reward.
     */
    public ItemBanditRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, ItemBandit<U, I> itemBandit, ValueFunction valFunc)
    {
        super(uIndex, iIndex, ignoreNotRated);
        this.itemBandit = itemBandit;
        this.valFunc = valFunc;
    }

    @Override
    public void init()
    {
        super.init();

        this.itemBandit.reset();
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();
        values.forEach(triplet -> this.itemBandit.update(triplet.iidx(), triplet.value()));
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        if(!Double.isNaN(value)) // If the (uidx, iidx) pair exists.
        {
            this.itemBandit.update(iidx, value);
        }
        else if(!this.ignoreNotRated) // If we update the bandit even when the (uidx, iidx) pair does not exist.
        {
            this.itemBandit.update(iidx, Constants.NOTRATEDNOTIGNORED);
        }
    }

    @Override
    public int next(int uidx, IntList availability)
    {
        return this.itemBandit.next(uidx, availability.toIntArray(), valFunc);
    }
}
