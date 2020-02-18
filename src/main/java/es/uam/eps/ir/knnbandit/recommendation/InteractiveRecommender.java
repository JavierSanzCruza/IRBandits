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
    /**
     * User index.
     */
    protected final FastUpdateableUserIndex<U> uIndex;
    /**
     * Item index.
     */
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
        //IntStream.range(0, prefData.numUsers()).forEach(uidx -> availability.add(this.getIidx().boxed().collect(Collectors.toCollection(IntArrayList::new))));
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
        /*IntStream.range(0, prefData.numUsers()).forEach(uidx ->
        {
            this.availability.add(this.getIidx().filter(iidx -> uidx != iidx).boxed().collect(Collectors.toCollection(IntArrayList::new)));
        });*/
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
     * Auxiliar initializer: Initializes only the availability and the training data, not other properties of the method.
     * @param contactRec true if we are using contact recommendation.
     */
    private void auxInit(boolean contactRec)
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
    }

    /**
     * Initializes the recommender.
     * @param availability the list of items which can be recommended to each user.
     */
    public void init(List<IntList> availability)
    {
        this.availability.clear();
        for(IntList items : availability)
        {
            this.availability.add(new IntArrayList(items));
        }
        this.rng = new Random(UntieRandomNumber.RNG);
        this.trainData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
    }

    /**
     * Auxiliar initializer: Initializes the availability, the random number and the training data (empty).
     * @param availability the list of items which can be recommended to each user.
     */
    private void auxInit(List<IntList> availability)
    {
        this.availability.clear();
        for(IntList items : availability)
        {
            this.availability.add(new IntArrayList(items));
        }
        this.rng = new Random(UntieRandomNumber.RNG);
        this.trainData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
    }

    /**
     * Initializer: Initializes all elements of the recommender.
     * @param train the training data.
     * @param availability the list of items which can be recommended to each user.
     * @param contactRec true if we are using contact recommendation.
     */
    public void init(List<Tuple2<Integer, Integer>> train, List<IntList> availability, boolean contactRec)
    {
        // First, we initialize the basic properties (with the availability list already pruned)
        this.auxInit(availability);

        // For each element:
        train.forEach(pair ->
        {
            // Retrieve user and item ids.
            int uidx = pair.v1;
            int iidx = pair.v2;

            // Check whether the element can be retrieved or not.
            double value = 0.0;
            boolean known = false;
            if (prefData.numItems(uidx) > 0 && prefData.numUsers(iidx) > 0)
            {
                Optional<IdxPref> realvalue = this.prefData.getPreference(uidx, iidx);
                if(realvalue.isPresent())
                {
                    value = realvalue.get().v2();
                    known = true;
                }
            }

            // If it is known, update the rating.
            if (!this.ignoreUnknown || known)
            {
                this.trainData.updateRating(uidx, iidx, value);
            }

            // If we are talking about contact recommendation, we cannot recommend reciprocals and the link exists...
            // (or, at least, we know about it...)
            if (contactRec && this.notReciprocal && known)
            {
                value = 0.0;
                known = false;
                if (prefData.numItems(uidx) > 0 && prefData.numUsers(iidx) > 0)
                {
                    Optional<IdxPref> realvalue = this.prefData.getPreference(iidx, uidx);
                    if(realvalue.isPresent())
                    {
                        value = realvalue.get().v2();
                        known = true;
                    }
                }

                if (!this.ignoreUnknown || known)
                {
                    this.trainData.updateRating(iidx, uidx, value);
                }
            }
        });

        // Initialize the specific properties of the method.
        this.initializeMethod();
    }


    /**
     * Initializes the recommender with training data.
     *
     * @param train the training data.
     * @param contactRec true if we are recommending people.
     */
    public void init(List<Tuple2<Integer, Integer>> train, boolean contactRec)
    {
        // First, we initialize the basic properties (with the availability list already pruned)
        this.auxInit(contactRec);

        // For each element in the training set, add it to the trainData, and update the availabilities
        train.forEach(pair ->
        {
            // Retrieve user and item ids.
            int uidx = pair.v1;
            int iidx = pair.v2;

            double value = 0.0;
            boolean known = false;
            if (prefData.numItems(uidx) > 0 && prefData.numUsers(iidx) > 0)
            {
                Optional<IdxPref> realvalue = this.prefData.getPreference(uidx, iidx);
                if(realvalue.isPresent())
                {
                    value = realvalue.get().v2();
                    known = true;
                }
            }

            if (!this.ignoreUnknown || known)
            {
                this.trainData.updateRating(uidx, iidx, value);
            }

            // remove the item from the availability list.
            this.availability.get(uidx).removeInt(this.availability.get(uidx).indexOf(iidx));

            // Update the reciprocal (if we check this).
            if (contactRec && this.notReciprocal && known) // If the link exists...
            {
                known = false;
                value = 0.0;
                if (prefData.numItems(uidx) > 0 && prefData.numUsers(iidx) > 0)
                {
                    Optional<IdxPref> realvalue = this.prefData.getPreference(iidx, uidx);
                    if(realvalue.isPresent())
                    {
                        value = realvalue.get().v2();
                        known = true;
                    }
                }

                if (!this.ignoreUnknown || known)
                {
                    this.trainData.updateRating(iidx, uidx, value);
                }

                // remove the reciprocal pair from the availability list.
                this.availability.get(iidx).removeInt(this.availability.get(iidx).indexOf(uidx));
            }
        });

        // Initialize the method.
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
        double value = 0.0;
        boolean isPresent = false;

        // First, we check if the rating exists.
        if(this.prefData.numUsers(iidx) > 0 && this.prefData.numItems(uidx) > 0)
        {
            Optional<IdxPref> realvalue = this.prefData.getPreference(uidx, iidx);
            if(realvalue.isPresent())
            {
                value = realvalue.get().v2;
                isPresent = true;
            }
        }

        // If the rating exists, or we do want to consider it:
        if(!this.ignoreUnknown || isPresent)
        {
            this.updateMethod(uidx, iidx, value);
            this.trainData.updateRating(uidx, iidx, value);
        }

        // Then, if we are in the contact recommendation case, and the rating exists...
        if(isPresent && this.notReciprocal)
        {
            value = 0.0;
            isPresent = false;
            // First, we check whether the element is in the data.
            if(this.prefData.numItems(iidx) > 0 && this.prefData.numUsers(uidx) > 0)
            {
                Optional<IdxPref> realvalue = this.prefData.getPreference(iidx, uidx);
                if(realvalue.isPresent())
                {
                    value = realvalue.get().v2;
                    isPresent = true;
                }
            }

            // If the rating exists, or we do want to consider it:
            if(!this.ignoreUnknown || isPresent)
            {
                this.updateMethod(iidx, uidx, value);
                this.trainData.updateRating(iidx, uidx, value);
            }

            // Remove uidx from the availability index of iidx, if it has not appeared earlier.
            int index = this.availability.get(iidx).indexOf(uidx);
            if(index > 0) // This pair has not been previously recommended:
                this.availability.get(iidx).removeInt(index);
        }

        // Remove iidx from the availability index of uidx.
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
        // For each tuple...
        for(Tuple2<Integer, Integer> tuple : train)
        {
            int uidx = tuple.v1;
            int iidx = tuple.v2;

            double value = 0.0;
            boolean isPresent = false;

            // First, we check if the rating exists.
            if(this.prefData.numUsers(iidx) > 0 && this.prefData.numItems(uidx) > 0)
            {
                Optional<IdxPref> realvalue = this.prefData.getPreference(uidx, iidx);
                if(realvalue.isPresent())
                {
                    value = realvalue.get().v2;
                    isPresent = true;
                }
            }

            // If the rating exists, or we do want to consider it:
            if(!this.ignoreUnknown || isPresent)
            {
                tuples.add(new Tuple3<>(uidx, iidx, value));
                this.trainData.updateRating(uidx, iidx, value);
            }

            // Then, if we are in the contact recommendation case, and the rating exists...
            if(isPresent && this.notReciprocal)
            {
                value = 0.0;
                isPresent = false;
                // First, we check whether the element is in the data.
                if(this.prefData.numItems(iidx) > 0 && this.prefData.numUsers(uidx) > 0)
                {
                    Optional<IdxPref> realvalue = this.prefData.getPreference(iidx, uidx);
                    if(realvalue.isPresent())
                    {
                        value = realvalue.get().v2;
                        isPresent = true;
                    }
                }

                // If the rating exists, or we do want to consider it:
                if(!this.ignoreUnknown || isPresent)
                {
                    tuples.add(new Tuple3<>(iidx, uidx, value));
                    this.trainData.updateRating(iidx, uidx, value);
                }

                // Remove uidx from the availability index of iidx, if it has not appeared earlier.
                int index = this.availability.get(iidx).indexOf(uidx);
                if(index > 0) // This pair has not been previously recommended:
                    this.availability.get(iidx).removeInt(index);
            }

            // Remove iidx from the availability index of uidx.
            this.availability.get(uidx).removeInt(this.availability.get(uidx).indexOf(iidx));
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
