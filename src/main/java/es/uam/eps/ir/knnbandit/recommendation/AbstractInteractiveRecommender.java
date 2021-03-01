/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.utils.FastRating;

import es.uam.eps.ir.knnbandit.utils.Rating;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Abstract definition of interactive recommendation algorithm.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado Puig (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class AbstractInteractiveRecommender<U, I> implements FastInteractiveRecommender<U,I>
{
    /**
     * User index.
     */
    protected final FastUpdateableUserIndex<U> uIndex;
    /**
     * Item index.
     */
    protected final FastUpdateableItemIndex<I> iIndex;
    /**
     * The random number seed
     */
    protected final int rngSeed;
    /**
     * Random number generator.
     */
    protected Random rng;
    /**
     *
     */
    protected boolean ignoreNotRated;

    /**
     * Constructor.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param ignoreNotRated true if we only consider known ratings, false otherwise.
     */
    public AbstractInteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.rngSeed = 0;
        this.ignoreNotRated = ignoreNotRated;
    }

    /**
     * Constructor.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param ignoreNotRated true if we only consider known ratings, false otherwise.
     * @param rngSeed the random number generator seed.
     */
    public AbstractInteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.rngSeed = rngSeed;
        this.ignoreNotRated = ignoreNotRated;
    }

    @Override
    public void init()
    {
        this.rng = new Random(rngSeed);
    }

    @Override
    public void initialize(Stream<Rating<U, I>> values)
    {
        this.init(values.map(x -> new FastRating(this.uIndex.user2uidx(x.getUser()), this.iIndex.item2iidx(x.getItem()), x.getValue())));
    }

    @Override
    public IntStream getUidx()
    {
        return uIndex.getAllUidx();
    }

    @Override
    public IntStream getIidx()
    {
        return iIndex.getAllIidx();
    }

    @Override
    public Stream<U> getUsers()
    {
        return uIndex.getAllUsers();
    }

    @Override
    public Stream<I> getItems()
    {
        return iIndex.getAllItems();
    }

    @Override
    public int numUsers()
    {
        return uIndex.numUsers();
    }

    @Override
    public int numItems()
    {
        return iIndex.numItems();
    }

    @Override
    public I next(U u, List<I> available)
    {
        IntList availableIds = available.stream().map(iIndex::item2iidx).collect(Collectors.toCollection(IntArrayList::new));
        int uidx = this.uIndex.user2uidx(u);
        int iidx = this.next(uidx, availableIds);
        if(iidx == -1) return null;
        return this.iIndex.iidx2item(iidx);

    }

    @Override
    public List<I> next(U u, List<I> available, int k)
    {
        IntList availableIds = available.stream().map(iIndex::item2iidx).collect(Collectors.toCollection(IntArrayList::new));
        int uidx = this.uIndex.user2uidx(u);
        return this.next(uidx, availableIds, k).stream().map(iIndex::iidx2item).collect(Collectors.toCollection(ArrayList::new));
    }

    @Override
    public void update(U u, I i, double value)
    {
        int uidx = this.uIndex.user2uidx(u);
        int iidx = this.iIndex.item2iidx(i);
        this.fastUpdate(uidx, iidx, value);
    }

    @Override
    public void update(List<Tuple3<U, I, Double>> train)
    {
        train.forEach(tuple -> this.update(tuple.v1, tuple.v2, tuple.v3));
    }

    @Override
    public void fastUpdate(List<Tuple3<Integer, Integer, Double>> train)
    {
        train.forEach(tuple -> this.fastUpdate(tuple.v1, tuple.v2, tuple.v3));
    }

    @Override
    public boolean usesAll()
    {
        return !this.ignoreNotRated;
    }


}
