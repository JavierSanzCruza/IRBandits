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

import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;

import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Abstract definition of interactive recommendation algorithm.
 *
 * @param <U> User type.
 * @param <I> Item type.
 *
 * @author Javier Sanz-Cruzado Puig (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class InteractiveRecommender<U, I>
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
     * True if the algorithm ignores the not rated ratings.
     */
    protected boolean ignoreNotRated;

    /**
     * Constructor.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param ignoreNotRated true if we only consider known ratings, false otherwise.
     */
    public InteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated)
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
    public InteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.rngSeed = rngSeed;
        this.ignoreNotRated = ignoreNotRated;
    }

    /**
     * Initializes the specific variables of a method.
     */
    public void init()
    {
        this.rng = new Random(rngSeed);
    }

    /**
     * Initializes the specific variables of a method, using some information as training data.
     * @param values a stream of (user, item, value) triplets.
     */
    public abstract void init(Stream<FastRating> values);

    /**
     * Obtains the set of identifiers of the users.
     *
     * @return the set of identifiers of the users.
     */
    public IntStream getUidx()
    {
        return uIndex.getAllUidx();
    }

    /**
     * Obtains the set of identifiers of the items.
     *
     * @return the set of identifiers of the items.
     */
    public IntStream getIidx()
    {
        return iIndex.getAllIidx();
    }

    /**
     * Obtains the users.
     *
     * @return the users.
     */
    public Stream<U> getUsers()
    {
        return uIndex.getAllUsers();
    }

    /**
     * Obtains the items.
     *
     * @return the items.
     */
    public Stream<I> getItems()
    {
        return iIndex.getAllItems();
    }

    /**
     * Obtains the number of users.
     *
     * @return the number of users.
     */
    public int numUsers()
    {
        return uIndex.numUsers();
    }

    /**
     * Obtains the number of items.
     *
     * @return the number of items.
     */
    public int numItems()
    {
        return iIndex.numItems();
    }

    /**
     * Given a user, and a list of items, returns the next value.
     * @param uidx user identifier.
     * @param available the list of identifiers of the candidate items.
     * @return the identifier of the recommended item if everything went OK, -1 otherwise i.e. when a user cannot be recommended an item)
     */
    public abstract int next(int uidx, IntList available);

    /**
     * Given a user, and the list of available items, returns a top-k recommendation (when possible).
     * If the algorithm can only recommend l < k items, but there are more available, those are
     * recommended randomly.
     *
     * @param uidx user identifier.
     * @param available the list of identifiers of the candidate items.
     * @param k the number of items to recommend.
     * @return a list of recommended items.
     */
    public abstract IntList next(int uidx, IntList available, int k);

    /**
     * Updates the method.
     *
     * @param uidx  User identifier.
     * @param iidx  Item identifier.
     * @param value The rating uidx provides to iidx.
     */
    public abstract void update(int uidx, int iidx, double value);

    /**
     * Updates the method.
     *
     * @param train Training data.
     */
    public void update(List<Tuple3<Integer, Integer, Double>> train)
    {
        train.forEach(tuple -> this.update(tuple.v1, tuple.v2, tuple.v3));
    }

    /**
     * Checks if the recommender uses all the received information, or only known data.
     *
     * @return true if the recommender uses all the received information, false otherwise.
     */
    public boolean usesAll()
    {
        return !this.ignoreNotRated;
    }
}
