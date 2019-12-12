/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation;

import es.uam.eps.ir.knnbandit.UntieRandomNumber;
import es.uam.eps.ir.knnbandit.data.preference.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Abstract definition of interactive recommendation algorithm.
 *
 * @param <U> User type.
 * @param <I> Item type.
 *
 * @author Javier Sanz-Cruzado Puig (javier.sanz-cruzado@uam.es)
 */
public abstract class InteractiveRecommender<U, I>
{
    protected final FastUpdateableUserIndex<U> uIndex;

    protected final FastUpdateableItemIndex<I> iIndex;
    /**
     * Preference data.
     */
    protected final SimpleFastPreferenceData<U, I> prefData;
    /**
     * A map including which items are recommendable for each user.
     */
    protected final List<IntList> availability;
    /**
     * True if we ignore missing ratings, false if we take them as failures.
     */
    protected final boolean ignoreUnknown;
    /**
     * True if we want to prevent recommending reciprocal links (only people-to-people recommendation in social networks).
     */
    protected final boolean notReciprocal;
    /**
     * Random number generator.
     */
    protected Random rng;
    /**
     * Training data.
     */
    protected SimpleFastUpdateablePreferenceData<U, I> trainData;

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      preference data.
     * @param ignoreUnknown False to treat missing ratings as failures, true otherwise.
     */
    public InteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreUnknown)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.prefData = prefData;
        this.trainData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.availability = new ArrayList<>();
        IntStream.range(0, prefData.numUsers()).forEach(uidx -> availability.add(this.getIidx().boxed().collect(Collectors.toCollection(IntArrayList::new))));
        this.ignoreUnknown = ignoreUnknown;
        this.notReciprocal = false;
        this.rng = new Random(UntieRandomNumber.RNG);
    }

    /**
     * Constructor for people-to-people recommendation.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      preference data.
     * @param ignoreUnknown False to treat missing ratings as failures, true otherwise.
     * @param notReciprocal False to treat missing ratings as failures, true otherwise.
     */
    public InteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreUnknown, boolean notReciprocal)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.prefData = prefData;
        this.trainData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.availability = new ArrayList<>();
        IntStream.range(0, prefData.numUsers()).forEach(uidx ->
        {
            this.availability.add(this.getIidx().filter(iidx -> uidx != iidx).boxed().collect(Collectors.toCollection(IntArrayList::new)));
        });
        this.ignoreUnknown = ignoreUnknown;
        this.notReciprocal = notReciprocal;
        this.rng = new Random(UntieRandomNumber.RNG);
    }

    /**
     * Initializes the recommender (resets it to its basics)
     */
    public void init(boolean contactRec)
    {
        availability.clear();

        this.rng = new Random(UntieRandomNumber.RNG);

        IntStream.range(0, prefData.numUsers()).forEach(uidx ->
        {
            if (contactRec)
            {
                availability.add(this.getIidx().filter(iidx -> uidx != iidx).boxed().collect(Collectors.toCollection(IntArrayList::new)));
            }
            else
            {
                availability.add(this.getIidx().boxed().collect(Collectors.toCollection(IntArrayList::new)));
            }
        });

        this.trainData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);

        this.initializeMethod();
    }

    /**
     * Initializes the recommender with training data.
     *
     * @param train the training data.
     */
    public void init(List<Tuple2<Integer, Integer>> train, boolean contactRec)
    {
        this.init(contactRec);

        // For each element in the training set, add it to the trainData, and update the availabilities
        train.forEach(pair ->
        {
            int uidx = pair.v1;
            int iidx = pair.v2;
            Optional<IdxPref> realvalue = this.prefData.getPreference(uidx, iidx);

            // Update the corresponding value.
            double value = realvalue.isPresent() ? realvalue.get().v2 : 0.0;
            if (!this.ignoreUnknown || realvalue.isPresent())
            {
                this.trainData.updateRating(uidx, iidx, value);
            }
            this.availability.get(uidx).removeInt(this.availability.get(uidx).indexOf(iidx));

            // Update the reciprocal (if we check this).
            if (this.notReciprocal && value > 1.0) // If the link exists...
            {
                realvalue = this.prefData.getPreference(iidx, uidx);
                value = realvalue.isPresent() ? realvalue.get().v2 : 0.0;
                if (!this.ignoreUnknown || realvalue.isPresent())
                {
                    this.trainData.updateRating(iidx, uidx, value);
                }
                this.availability.get(iidx).removeInt(this.availability.get(iidx).indexOf(uidx));
            }
        });

        this.initializeMethod();
    }

    /**
     * Initializes the specific variables of a method, using the information stored as training data.
     */
    protected abstract void initializeMethod();

    /**
     * Obtains the set of identifiers of the users.
     *
     * @return the set of identifiers of the users.
     */
    public IntStream getUidx()
    {
        return prefData.getAllUidx();
    }

    /**
     * Obtains the set of identifiers of the items.
     *
     * @return the set of identifiers of the items.
     */
    public IntStream getIidx()
    {
        return prefData.getAllIidx();
    }

    /**
     * Obtains the users.
     *
     * @return the users.
     */
    public Stream<U> getUsers()
    {
        return prefData.getAllUsers();
    }

    /**
     * Obtains the items.
     *
     * @return the items.
     */
    public Stream<I> getItems()
    {
        return prefData.getAllItems();
    }

    /**
     * Obtains the number of users.
     *
     * @return the number of users.
     */
    public int numUsers()
    {
        return prefData.numUsers();
    }

    /**
     * Obtains the number of items.
     *
     * @return the number of items.
     */
    public int numItems()
    {
        return prefData.numItems();
    }

    /**
     * Given a user, returns the next value.
     *
     * @param uidx User identifier
     *
     * @return the identifier of the recommended item if everything went ok, -1 otherwise (i.e. when a user cannot be recommended).
     */
    public abstract int next(int uidx);

    /**
     * Updates the recommender.
     *
     * @param uidx The target user.
     * @param iidx The recommended item.
     */
    public void update(int uidx, int iidx)
    {
        if (this.prefData.numUsers(iidx) > 0 && this.prefData.numItems(uidx) > 0)
        {
            Optional<IdxPref> realvalue = this.prefData.getPreference(uidx, iidx);
            double value = realvalue.isPresent() ? realvalue.get().v2 : 0.0;
            if (!this.ignoreUnknown || realvalue.isPresent())
            {
                this.updateMethod(uidx, iidx, value);
                this.trainData.updateRating(uidx, iidx, value);
            }

            if (this.notReciprocal && value > 1.0) // If the link exists...
            {
                if (this.prefData.numItems(iidx) > 0)
                {
                    realvalue = this.prefData.getPreference(iidx, uidx);
                    value = realvalue.isPresent() ? realvalue.get().v2 : 0.0;
                    if (!this.ignoreUnknown || realvalue.isPresent())
                    {
                        this.updateMethod(iidx, uidx, value);
                        this.trainData.updateRating(iidx, uidx, value);
                    }
                }
                this.availability.get(iidx).removeInt(this.availability.get(iidx).indexOf(uidx));
            }
        }
        else
        {
            if (!this.ignoreUnknown)
            {
                this.updateMethod(uidx, iidx, 0.0);
                this.trainData.updateRating(uidx, iidx, 0.0);
            }
        }

        if (uidx == -1)
        {
            System.err.println("ERROR: uidx == -1");
        }
        else if (iidx == -1)
        {
            System.err.println("ERROR: iidx == -1");
        }
        else if (this.availability.get(uidx).indexOf(iidx) == -1)
        {
            System.err.println("iidx not available in uidx");
        }

        this.availability.get(uidx).removeInt(this.availability.get(uidx).indexOf(iidx));
    }

    /**
     * Updates the method.
     *
     * @param uidx  User identifier.
     * @param iidx  Item identifier.
     * @param value The rating uidx provides to iidx.
     */
    public abstract void updateMethod(int uidx, int iidx, double value);

    /**
     * Updates the method with training data.
     *
     * @param train The training data.
     */
    public void update(List<Tuple2<Integer, Integer>> train)
    {
        List<Tuple3<Integer, Integer, Double>> tuples = new ArrayList<>();
        for (Tuple2<Integer, Integer> tuple : train)
        {
            int uidx = tuple.v1;
            int iidx = tuple.v2;
            Optional<IdxPref> realvalue = this.prefData.getPreference(uidx, iidx);
            double value = realvalue.isPresent() ? realvalue.get().v2 : 0.0;
            if (!this.ignoreUnknown || realvalue.isPresent())
            {
                tuples.add(new Tuple3<>(uidx, iidx, value));
                this.trainData.updateRating(uidx, iidx, value);
            }

            if (this.availability.get(uidx).contains(iidx))
            {
                this.availability.get(uidx).removeInt(this.availability.get(uidx).indexOf(iidx));
            }

            if (this.notReciprocal)
            {
                if (this.prefData.numItems(iidx) > 0)
                {
                    realvalue = this.prefData.getPreference(iidx, uidx);
                    value = realvalue.isPresent() ? realvalue.get().v2 : 0.0;
                    if (!this.ignoreUnknown || realvalue.isPresent())
                    {
                        tuples.add(new Tuple3<>(iidx, uidx, value));
                        this.trainData.updateRating(iidx, uidx, value);
                    }
                }

                if (this.availability.get(iidx).contains(uidx))
                {
                    this.availability.get(iidx).removeInt(this.availability.get(iidx).indexOf(uidx));
                }
            }
        }

        this.updateMethod(tuples);
    }


    /**
     * Updates the method.
     *
     * @param train Training data.
     */
    public void updateMethod(List<Tuple3<Integer, Integer, Double>> train)
    {
        train.forEach(tuple -> this.updateMethod(tuple.v1, tuple.v2, tuple.v3));
    }

    /**
     * Checks if the recommender uses all the received information, or only known data.
     *
     * @return true if the recommender uses all the received information, false otherwise.
     */
    public boolean usesAll()
    {
        return !this.ignoreUnknown;
    }
}
