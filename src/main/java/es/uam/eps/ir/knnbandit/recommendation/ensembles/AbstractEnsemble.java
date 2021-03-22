/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.ensembles;

import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.FastInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Abstract implementation of an interactive recommendation ensemble.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class AbstractEnsemble<U,I> extends AbstractInteractiveRecommender<U,I> implements FastEnsemble<U,I>
{
    /**
     * A list of algorithm suppliers.
     */
    protected final List<InteractiveRecommenderSupplier<U,I>> suppliers;
    /**
     * The current list of interactive recommendation algorithms.
     */
    protected final List<FastInteractiveRecommender<U,I>> recommenders;
    /**
     * The names of the interactive recommendation algorithms in the comparison.
     */
    protected final List<String> recNames;

    /**
     * The number of hits of each algorithm.
     */
    protected final IntList hits;
    /**
     * The number of misses of each algorithm.
     */
    protected final IntList misses;
    /**
     * The identifier of the currently used algorithm.
     */
    protected int currentAlgorithm;

    /**
     * Constructor.
     * @param uIndex            user index.
     * @param iIndex            item index.
     * @param ignoreNotRated    true if items not in the definitive dataset are used to update the recommendation algorithms.
     * @param recs              a map, indexed by the recommender name, containing recommender suppliers.
     */
    public AbstractEnsemble(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, Map<String, InteractiveRecommenderSupplier<U,I>> recs)
    {
        super(uIndex, iIndex, ignoreNotRated);

        this.suppliers = new ArrayList<>();
        this.recNames = new ArrayList<>();
        this.recommenders = new ArrayList<>();

        recs.forEach((key, supplier) ->
        {
             this.recNames.add(key);
             this.suppliers.add(supplier);
        });

        this.hits = new IntArrayList();
        this.misses = new IntArrayList();
    }

    /**
     * Constructor.
     * @param uIndex            user index.
     * @param iIndex            item index.
     * @param ignoreNotRated    true if items not in the definitive dataset are used to update the recommendation algorithms.
     * @param rngSeed           the random number generator seed to solve ties.
     * @param recs              a map, indexed by the recommender name, containing recommender suppliers.
     */
    public AbstractEnsemble(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, Map<String, InteractiveRecommenderSupplier<U,I>> recs)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed);

        this.suppliers = new ArrayList<>();
        this.recNames = new ArrayList<>();
        this.recommenders = new ArrayList<>();

        recs.forEach((key, supplier) ->
        {
            this.recNames.add(key);
            this.suppliers.add(supplier);
        });

        this.hits = new IntArrayList();
        this.misses = new IntArrayList();
    }

    @Override
    public void init()
    {
        super.init();
        this.recommenders.clear();
        this.hits.clear();
        this.misses.clear();

        for(InteractiveRecommenderSupplier<U,I> supp : this.suppliers)
        {
            FastInteractiveRecommender<U,I> rec = supp.apply(uIndex, iIndex, rngSeed);
            rec.init();
            this.recommenders.add(rec);
            this.hits.add(0);
            this.misses.add(0);
        }

        this.currentAlgorithm = -1;
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        super.init();
        this.recommenders.clear();
        this.hits.clear();
        this.misses.clear();

        for(InteractiveRecommenderSupplier<U,I> supp : this.suppliers)
        {
            FastInteractiveRecommender<U,I> rec = supp.apply(uIndex, iIndex, rngSeed);
            rec.init(values);
            this.recommenders.add(rec);
            this.hits.add(0);
            this.misses.add(0);
        }

        this.currentAlgorithm = -1;
    }

    @Override
    public int next(int uidx, IntList available)
    {
        this.currentAlgorithm = this.selectAlgorithm(uidx);
        return this.recommenders.get(currentAlgorithm).next(uidx, available);
    }

    @Override
    public IntList next(int uidx, IntList available, int k)
    {
        this.currentAlgorithm = this.selectAlgorithm(uidx);
        return this.recommenders.get(currentAlgorithm).next(uidx, available, k);
    }

    @Override
    public void fastUpdate(int uidx, int iidx, double value)
    {
        double newValue;
        if(!Double.isNaN(value) && value != Constants.NOTRATEDRATING)
        {
            newValue = value;
        }
        else if(!this.ignoreNotRated)
        {
            newValue = Constants.NOTRATEDNOTIGNORED;
        }
        else
        {
            return;
        }

        if(newValue > 0.0) // it is a hit
        {
            int currentHits = this.hits.getInt(currentAlgorithm);
            this.hits.set(currentAlgorithm, currentHits + 1);
        }
        else // it is a miss
        {
            int currentMisses = this.misses.getInt(currentAlgorithm);
            this.misses.set(currentAlgorithm, currentMisses + 1);
        }

        // Update all the recommenders with the current information.
        recommenders.forEach(rec -> rec.fastUpdate(uidx, iidx, value));

        this.updateEnsemble(uidx, iidx, value);
    }

    /**
     * Updates the specific parameters for the ensemble.
     * @param uidx the identifier of the user.
     * @param iidx the identifier of the item.
     * @param value the value given by the user to the item.
     */
    protected abstract void updateEnsemble(int uidx, int iidx, double value);

    @Override
    public int getAlgorithmCount()
    {
        return this.recNames.size();
    }

    @Override
    public List<String> getAlgorithms()
    {
        return this.recNames;
    }

    @Override
    public int getCurrentAlgorithm()
    {
        return this.currentAlgorithm;
    }

    @Override
    public String getAlgorithmName(int idx)
    {
        if(idx < 0 || idx >= this.getAlgorithmCount()) return null;
        return this.recNames.get(idx);
    }

    @Override
    public Pair<Integer> getAlgorithmStats(int idx)
    {
        if(idx < 0 || idx >= this.getAlgorithmCount()) return null;
        return new Pair<>(hits.getInt(idx), misses.getInt(idx));
    }

    /**
     * Obtains the identifier of the next recommendation algorithm to apply.
     * @param uidx the identifier of the user.
     * @return the identifier of the next recommendation algorithm to apply.
     */
    protected abstract int selectAlgorithm(int uidx);

    @Override
    public void setCurrentAlgorithm(int idx)
    {
        this.currentAlgorithm = idx;
    }
}