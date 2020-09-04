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
import es.uam.eps.ir.knnbandit.UntieRandomNumberReader;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.userknowledge.fast.SimpleFastUserKnowledgePreferenceData;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
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
     * Knowledge about whether the user knows about the items or not
     */
    protected final SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData;
    /**
     * A map including which items are recommendable for each user.
     */
    protected final List<IntList> availability;
    /**
     * True if we ignore missing ratings, false if we take them as failures.
     */
    protected final boolean ignoreNotRated;
    /**
     * True if we want to prevent recommending reciprocal links (only people-to-people recommendation in social networks).
     */
    protected final boolean notReciprocal;
    /**
     * In the case of using data with knowledge about whether the user knew about the item before consuming it,
     * identify which information we have to use to update the recommender.
     */
    protected final KnowledgeDataUse dataUse;
    /**
     * Random number seed generator.
     */
    protected final UntieRandomNumberReader rngSeedGen;
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
     * @param uIndex         User index.
     * @param iIndex         Item index.
     * @param prefData       preference data.
     * @param ignoreNotRated true if we have to update the recommender only when the item has test ratings, false otherwise.
     */
    public InteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreNotRated)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.prefData = prefData;
        this.trainData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.availability = new ArrayList<>();
        this.ignoreNotRated = ignoreNotRated;
        this.notReciprocal = false;
        this.knowledgeData = null;
        this.dataUse = KnowledgeDataUse.ALL;
        this.rngSeedGen = new UntieRandomNumberReader();
    }


    /**
     * Constructor.
     *
     * @param uIndex         User index.
     * @param iIndex         Item index.
     * @param prefData       preference data.
     * @param ignoreNotRated true if we have to update the recommender only when the item has test ratings, false otherwise.
     * @param dataUse        configuration for selecting which ratings we have to use to update.
     */
    public InteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData, boolean ignoreNotRated, KnowledgeDataUse dataUse)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.prefData = prefData;
        this.trainData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.availability = new ArrayList<>();
        this.ignoreNotRated = ignoreNotRated;
        this.notReciprocal = false;
        this.knowledgeData = knowledgeData;
        this.dataUse = dataUse;
        this.rngSeedGen = new UntieRandomNumberReader();
    }

    /**
     * Constructor for people-to-people recommendation.
     *
     * @param uIndex         User index.
     * @param iIndex         Item index.
     * @param prefData       preference data.
     * @param ignoreNotRated False to treat missing ratings as failures, true otherwise.
     * @param notReciprocal  False to treat missing ratings as failures, true otherwise.
     */
    public InteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreNotRated, boolean notReciprocal)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.prefData = prefData;
        this.trainData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.availability = new ArrayList<>();
        this.ignoreNotRated = ignoreNotRated;
        this.notReciprocal = notReciprocal;
        this.knowledgeData = null;
        this.dataUse = KnowledgeDataUse.ALL;
        this.rngSeedGen = new UntieRandomNumberReader();
    }

    /**
     * Initializer: Initializes the recommender (resets it to its basics)
     */
    public void init(boolean contactRec)
    {
        this.auxInit(contactRec);
        this.initializeMethod();
    }

    /**
     * Initializer: Initializes the recommender.
     *
     * @param availability the list of items which can be recommended to each user.
     */
    public void init(List<IntList> availability)
    {
        this.auxInit(availability);
        this.initializeMethod();
    }

    /**
     * Initializer: Initializes all elements of the recommender.
     *
     * @param warmup     the warm-up data.
     * @param contactRec true if we are using contact recommendation.
     */
    public void init(Warmup warmup, boolean contactRec)
    {
        List<IntList> availability = warmup.getAvailability();
        // First, we initialize the basic properties (with the availability list already pruned)
        this.auxInit(availability);

        List<Tuple2<Integer, Integer>> train = this.usesAll() ? warmup.getFullTraining() : warmup.getCleanTraining();

        // For each element:
        train.forEach(pair ->
                      {
                          // Retrieve user and item ids.
                          int uidx = pair.v1;
                          int iidx = pair.v2;

                          boolean hasUpdated = this.auxUpdateRating(uidx, iidx, false);
                          // If the (uidx,iidx) pair has been updated (success), then, if it is contact recommendation
                          // and the notReciprocal flag is activated, update the rating, by just adding it.
                          if (hasUpdated && contactRec && this.notReciprocal)
                          {
                              this.auxUpdateRating(iidx, uidx, false);
                          }
                      });

        // Initialize the specific properties of the method.
        this.initializeMethod();
    }

    /**
     * Initializes the recommender with training data.
     *
     * @param train      the training data.
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

            // Update the rating, and modify the corresponding availability.
            boolean didExist = this.auxUpdateRating(uidx, iidx, false);
            this.availability.get(uidx).removeInt(this.availability.get(uidx).indexOf(iidx));
            if (didExist && contactRec && this.notReciprocal)
            {
                int index = this.availability.get(iidx).indexOf(uidx);
                if (index > 0) // It might happen that the user uidx has been previously recommended to iidx (but it was not a hit)
                {
                    this.auxUpdateRating(iidx, uidx, false);
                    this.availability.get(iidx).removeInt(index);
                }
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
        boolean hasUpdated = this.auxUpdateRating(uidx, iidx, true);
        int index = this.availability.get(uidx).indexOf(iidx);
        if(index == -1)
        {
            System.err.println("WHAT?");
        }
        this.availability.get(uidx).removeInt(index);

        if (hasUpdated && this.notReciprocal)
        {
            index = this.availability.get(iidx).indexOf(uidx);
            if (index > 0) // This pair has not been previously recommended:
            {
                this.availability.get(iidx).removeInt(index);
                this.auxUpdateRating(iidx, uidx, true);
            }
        }
    }

    /**
     * Updates the method with training data.
     *
     * @param train The training data.
     */
    public void update(List<Tuple2<Integer, Integer>> train)
    {
        List<Tuple3<Integer, Integer, Double>> tuples = new ArrayList<>();
        // For each tuple...
        for (Tuple2<Integer, Integer> tuple : train)
        {
            int uidx = tuple.v1;
            int iidx = tuple.v2;
            boolean hasUpdated = this.auxUpdateRating(uidx, iidx, false);
            this.availability.get(uidx).removeInt(this.availability.get(uidx).indexOf(iidx));

            if (hasUpdated)
            {
                double value = this.trainData.getPreference(uidx, iidx).get().v2();
                tuples.add(new Tuple3<>(uidx, iidx, value));

                if (this.notReciprocal)
                {
                    int index = this.availability.get(iidx).indexOf(uidx);
                    if (index > 0) // This pair has not been previously recommended:
                    {
                        this.availability.get(iidx).removeInt(index);
                        hasUpdated = this.auxUpdateRating(iidx, uidx, false);
                        if (hasUpdated)
                        {
                            this.availability.get(iidx).removeInt(index);
                            value = this.trainData.getPreference(iidx, uidx).get().v2();
                            tuples.add(new Tuple3<>(iidx, uidx, value));
                        }
                    }
                }
            }
        }

        this.updateMethod(tuples);
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
        return !this.ignoreNotRated;
    }

    /**
     * Auxiliar initializer: Initializes only the availability and the training data, not other properties of the method.
     *
     * @param contactRec true if we are using contact recommendation.
     */
    private void auxInit(boolean contactRec)
    {
        // Initialize the availability lists and the random number generator.
        this.availability.clear();
        this.rng = new Random(rngSeedGen.nextSeed());

        // Fill the availability lists.
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

        // Load the training data (empty data).
        this.trainData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
    }

    /**
     * Auxiliar initializer: Initializes the availability, the random number and the training data (empty).
     *
     * @param availability the list of items which can be recommended to each user.
     */
    private void auxInit(List<IntList> availability)
    {
        this.availability.clear();
        for (IntList items : availability)
        {
            this.availability.add(new IntArrayList(items));
        }
        this.rng = new Random(rngSeedGen.nextSeed());
        this.trainData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
    }


    /**
     * Auxiliar function for updating ratings
     *
     * @param uidx         the identifier of the user.
     * @param iidx         the identifier of the item.
     * @param updateMethod true if the method has to be updated false otherwise.
     * @return true if the rating exists and it has been added to training.
     */
    private boolean auxUpdateRating(int uidx, int iidx, boolean updateMethod)
    {
        double value = 0.0;
        boolean hasRating = false;
        boolean isKnown = false;

        // If the user and the item exist in the preference data.
        if (prefData.numItems(uidx) > 0 && prefData.numUsers(iidx) > 0)
        {
            // Obtain then the possible preference for uidx to iidx.
            Optional<IdxPref> realvalue = this.prefData.getPreference(uidx, iidx);
            if (realvalue.isPresent())
            {
                value = realvalue.get().v2();
                hasRating = true;
            }

            // For datasets with information about whether the user knew about the methods.
            if (hasRating && dataUse != KnowledgeDataUse.ALL)
            {
                realvalue = this.knowledgeData.getKnownPreference(uidx, iidx);
                if (realvalue.isPresent())
                {
                    isKnown = true;
                }
            }
        }

        boolean update = (!this.ignoreNotRated || hasRating);
        switch (dataUse)
        {
            case ONLYKNOWN:
                update = update && isKnown;
                break;
            case ONLYUNKNOWN:
                update = update && !isKnown;
                break;
        }

        // If we have updated the recommender...
        if (update)
        {
            // If we have to update the method, do it.
            if (updateMethod)
            {
                this.updateMethod(uidx, iidx, value);
            }
            // Update the rating in the training.
            this.trainData.updateRating(uidx, iidx, value);
        }

        return hasRating;
    }
}
