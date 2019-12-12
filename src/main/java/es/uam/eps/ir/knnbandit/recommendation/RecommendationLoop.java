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

import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple2;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Class for simulating the recommendation loop.
 *
 * @param <U> User type.
 * @param <I> Item type.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RecommendationLoop<U, I>
{
    /**
     * User index.
     */
    private final FastUserIndex<U> userIndex;
    /**
     * Item index.
     */
    private final FastItemIndex<I> itemIndex;
    /**
     * The recommendation algorithm.
     */
    private final InteractiveRecommender<U, I> recommender;
    /**
     * The metrics we want to find.
     */
    private final Map<String, CumulativeMetric<U, I>> metrics;
    /**
     * The random seed for the random number generator.
     */
    private final int rngSeed;
    /**
     * Total number of iterations.
     */
    private final int nIter;
    /**
     *
     */
    private final boolean notReciprocal;
    /**
     * List of recommendable users.
     */
    private final IntList userList = new IntArrayList();
    private final FastPreferenceData<U, I> prefData;
    /**
     * Random number generator.
     */
    private Random rng;
    /**
     * The number of users with recommendations.
     */
    private int numUsers;
    /**
     * The current iteration number.
     */
    private int iteration;

    /**
     * Constructor. Uses 0 as the default random seed.
     *
     * @param userIndex     Index containing the users.
     * @param itemIndex     Index containing the items.
     * @param recommender   The interactive recommendation algorithm.
     * @param metrics       The map of metrics.
     * @param nIter         Total number of iterations. 0 for iterating until no more recommendations can be done.
     * @param notReciprocal s
     */
    public RecommendationLoop(FastUserIndex<U> userIndex, FastItemIndex<I> itemIndex, FastPreferenceData<U, I> prefData, InteractiveRecommender<U, I> recommender, Map<String, CumulativeMetric<U, I>> metrics, int nIter, boolean notReciprocal)
    {
        this.userIndex = userIndex;
        this.itemIndex = itemIndex;
        this.prefData = prefData;

        this.recommender = recommender;
        this.metrics = metrics;
        this.numUsers = userIndex.numUsers();
        this.rngSeed = 0;
        this.nIter = nIter;

        rng = new Random(rngSeed);
        this.iteration = 0;
        this.notReciprocal = notReciprocal;
    }

    /**
     * Constructor.
     *
     * @param userIndex     Index containing the users.
     * @param itemIndex     Index containing the items.
     * @param recommender   The interactive recommendation algorithm.
     * @param metrics       The map of metrics.
     * @param nIter         Total number of iterations. 0 for iterating until no more recommendations can be done.
     * @param rngSeed       seed for a random number generator.
     * @param notReciprocal d
     */
    public RecommendationLoop(FastUserIndex<U> userIndex, FastItemIndex<I> itemIndex, FastPreferenceData<U, I> prefData, InteractiveRecommender<U, I> recommender, Map<String, CumulativeMetric<U, I>> metrics, int nIter, int rngSeed, boolean notReciprocal)
    {
        this.userIndex = userIndex;
        this.itemIndex = itemIndex;
        this.prefData = prefData;

        this.recommender = recommender;
        this.metrics = metrics;
        this.numUsers = userIndex.numUsers();
        this.rngSeed = 0;
        rng = new Random(rngSeed);
        this.nIter = nIter;
        this.iteration = 0;
        this.notReciprocal = notReciprocal;
    }

    /**
     * Initializes everything without training data.
     */
    public void init(boolean contactRec)
    {
        this.recommender.init(contactRec);
        this.metrics.forEach((name, metric) -> metric.reset());
        this.userList.clear();
        this.prefData.getUidxWithPreferences().forEach(userList::add);
        this.numUsers = this.userList.size();
    }


    /**
     * Given some training data, initializes the recommendation loop.
     *
     * @param train the training data.
     */
    public void init(List<Tuple2<Integer, Integer>> train, boolean contactRec)
    {
        this.recommender.init(train, contactRec);
        this.metrics.forEach((name, metric) -> metric.initialize(train, notReciprocal));
        this.userList.clear();
        this.prefData.getUidxWithPreferences().forEach(userList::add);
        this.numUsers = this.userList.size();
    }

    /**
     * Checks if the loop has ended or not.
     *
     * @return true if the loop has ended, false otherwise.
     */
    public boolean hasEnded()
    {
        if (numUsers == 0)
        {
            return true;
        }
        return nIter > 0 && this.iteration >= nIter;
    }

    /**
     * Recovers previous iterations from a file.
     *
     * @param tuple A tuple containing the user and item to update.
     */
    public void update(Tuple2<Integer, Integer> tuple)
    {
        int uidx = tuple.v1;
        int iidx = tuple.v2;

        this.recommender.update(uidx, iidx);
        this.metrics.forEach((name, metric) -> metric.update(uidx, iidx));
        ++this.iteration;
    }

    /**
     * Obtains the iteration number.
     *
     * @return the iteration number.
     */
    public int getCurrentIteration()
    {
        return this.iteration;
    }

    /**
     * Executes the next iteration of the loop.
     *
     * @return a tuple containing the user and the item selected in the loop. Null if the loop has finished.
     */
    public Tuple2<Integer, Integer> nextIteration()
    {
        // We cannot continue.
        if (this.numUsers == 0)
        {
            return null;
        }

        // Select user and item for this iteration.
        boolean cont = false;
        int uidx;
        int iidx;
        do
        {
            int index = rng.nextInt(numUsers);
            uidx = this.userList.get(index);
            iidx = recommender.next(uidx);
            // If the user cannot be recommended another item.
            if (iidx != -1)
            {
                cont = true;
            }
            else
            {
                this.numUsers--;
                this.userList.remove(index);
            }
        }
        while (!cont && this.numUsers > 0);

        if (this.numUsers == 0)
        {
            return null;
        }

        int defUidx = uidx;
        int defIidx = iidx;
        recommender.update(defUidx, defIidx);
        metrics.forEach((name, metric) -> metric.update(defUidx, defIidx));
        ++this.iteration;
        return new Tuple2<>(uidx, iidx);
    }

    /**
     * Obtains the values for the metrics in the current iteration.
     *
     * @return the values for the metrics in the current iteration.
     */
    public Map<String, Double> getMetrics()
    {
        Map<String, Double> values = new HashMap<>();
        this.metrics.forEach((name, metric) -> values.put(name, metric.compute()));
        return values;
    }
}
