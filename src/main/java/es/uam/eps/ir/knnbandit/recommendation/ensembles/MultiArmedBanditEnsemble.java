/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.ensembles;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.MultiArmedBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.selector.algorithms.bandit.BanditSupplier;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.Map;
import java.util.stream.Stream;

/**
 * Dynamic ensemble that uses multi-armed bandit strategies to decide between recommenders.
 *
 * @param <U> type of the users.
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class MultiArmedBanditEnsemble<U,I> extends AbstractEnsemble<U,I>
{
    /**
     * The multi-armed bandit to select between recommenders.
     */
    private final MultiArmedBandit bandit;
    /**
     * Auxiliar list.
     */
    private final IntList available;
    /**
     * The value function for the bandit
     */
    private final ValueFunction valFunc;

    /**
     * Constructor.
     * @param uIndex the user index.
     * @param iIndex the item index.
     * @param ignoreNotRated true if we want to ignore user-item pairs without rating, false otherwise.
     * @param mabFunc a function for obtaining a multi-armed bandit.
     * @param recs a map, indexed by recommender name, and containing recommender suppliers as values.
     * @param valFunc a value function for the bandit.
     */
    public MultiArmedBanditEnsemble(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, Map<String, InteractiveRecommenderSupplier<U,I>> recs, BanditSupplier mabFunc, ValueFunction valFunc)
    {
        super(uIndex, iIndex, ignoreNotRated, recs);
        bandit = mabFunc.apply(this.getAlgorithmCount());
        this.available = new IntArrayList();
        for(int i = 0; i < this.getAlgorithmCount(); ++i) available.add(i);
        this.valFunc = valFunc;
    }

    /**
     * Constructor.
     * @param uIndex the user index.
     * @param iIndex the item index.
     * @param ignoreNotRated true if we want to ignore user-item pairs without rating, false otherwise.
     * @param rngSeed random number generator seed.
     * @param mabFunc a function for obtaining a multi-armed bandit.
     * @param recs a map, indexed by recommender name, and containing recommender suppliers as values.
     * @param valFunc a value function for the bandit.
     */
    public MultiArmedBanditEnsemble(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, Map<String, InteractiveRecommenderSupplier<U,I>> recs, BanditSupplier mabFunc, ValueFunction valFunc)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed, recs);
        bandit = mabFunc.apply(this.getAlgorithmCount());
        this.available = new IntArrayList();
        for(int i = 0; i < this.getAlgorithmCount(); ++i) available.add(i);
        this.valFunc = valFunc;
    }

    @Override
    public void init()
    {
        super.init();
        bandit.reset();
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        super.init(values);
        bandit.reset();
    }

    @Override
    protected void updateEnsemble(int uidx, int iidx, double value)
    {
        bandit.update(currentAlgorithm, value);
    }

    @Override
    protected int selectAlgorithm(int uidx)
    {
        return bandit.next(available, valFunc);
    }
}