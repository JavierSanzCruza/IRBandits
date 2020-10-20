/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.metrics;

import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.novdiv.distance.FeatureItemDistanceModel;
import it.unimi.dsi.fastutil.ints.*;
import org.jooq.lambda.tuple.Tuple2;

import java.util.List;


// TODO Pensar la versión con descuentos / para solo relevantes (fijarse en Vargas et al. 2011)

/**
 * Cumulative version of the intra-list diversity metric. Measures how different are the items
 * recommended to a single user over time.
 *
 * @param <U> the type of the users.
 * @param <I> the type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class CumulativeILD<U, I, F, V> implements CumulativeMetric<U, I>
{
    /**
     * Number of users of the system.
     */
    private final int numUsers;
    /**
     * Number of items of the system.
     */
    private final int numItems;
    /**
     * Model for the distance between different items. We assume that this is static, i.e. it depends on external properties
     * of the items, such as features.
     */
    private final FeatureItemDistanceModel<I, F, V> distModel;
    /**
     * Index for translating from items to indexes and viceversa.
     */
    private final FastItemIndex<I> index;
    /**
     * Global ILD sum.
     */
    double sum;
    /**
     * Hashmap containing the set of items of each user
     */
    private Int2ObjectMap<IntSet> usersItemsSets;
    /**
     * Hashmap containing the sets of items recommended to each user.
     */
    private Int2IntMap usersItemsCount;
    /**
     * Hashmap containing the ILD main term sums for each user.
     */
    private Int2DoubleMap sums;


    /**
     * Constructor. This metric assumes that the distance model is static
     * (i.e. it does not change with time). Therefore, it does not depend on
     * the ratings: only on external features.
     *
     * @param numUsers number of users in the system.
     * @param numItems number of items in the system.
     */
    public CumulativeILD(int numUsers, int numItems, FeatureItemDistanceModel<I, F, V> distanceModel, FastItemIndex<I> index)
    {
        // Store the number of users and items.
        this.numUsers = numUsers;
        this.numItems = numItems;

        // Create the user-related elements.
        this.usersItemsSets = new Int2ObjectOpenHashMap<IntSet>();
        this.usersItemsCount = new Int2IntOpenHashMap();
        this.sums = new Int2DoubleOpenHashMap();

        // Initialize the global sum.
        this.sum = 0.0;

        // Initialize the different users.
        for (int i = 0; i < numUsers; ++i)
        {
            this.usersItemsSets.put(i, new IntOpenHashSet());
            this.usersItemsCount.put(i, 0);
            this.sums.put(i, 0.0);
        }

        this.distModel = distanceModel;
        this.index = index;
    }

    @Override
    public void initialize(List<FastRating> train, boolean notReciprocal)
    {

    }

    @Override
    public double compute()
    {
        if (this.numUsers <= 0)
        {
            return Double.NaN;
        }
        return sum / (this.numUsers + 0.0);
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        // Check that the input is OK.
        if (uidx < 0 || uidx >= numUsers || iidx < 0 || iidx >= numItems)
        {
            System.err.println("Something failed while updating cumulative EILD");
            return;
        }

        // First, remove the current value for the user from the expanded sum.
        // We suppose that ILD(u) = 0 if the system has only recommended one item to the user.
        int count = this.usersItemsCount.get(uidx);
        double userSum = this.sums.get(uidx);
        if (count >= 2) // First, we remove the value from the expanded sum.
        {
            sum -= count * (count - 1.0) * userSum;
        }

        I newItem = index.iidx2item(iidx);

        // Now, we compute the increment on the metric.
        double addition = 0.0;
        for (int itemId : this.usersItemsSets.getOrDefault(uidx, new IntOpenHashSet()))
        {
            I oldItem = index.iidx2item(itemId);
            addition += this.distModel.dist(newItem, oldItem) + this.distModel.dist(oldItem, newItem);
        }

        userSum += addition;

        // Update the different values
        if (count >= 1)
        {
            sum += userSum / (count * (count + 1.0)); // the global sum, only if the user has been recommended more than one item.
        }
        this.sums.put(uidx, userSum); // the individual one.
        this.usersItemsCount.put(uidx, count + 1); // the number of items recommended to the user.
        this.usersItemsSets.get(uidx).add(iidx); // the set of items recommended to the user.
    }

    @Override
    public void reset()
    {
        this.sum = 0.0;
        this.sums.clear();
        this.usersItemsCount.clear();
        this.usersItemsSets.clear();

        for (int i = 0; i < this.numUsers; ++i)
        {
            this.sums.put(i, 0.0);
            this.usersItemsCount.put(i, 0);
            this.usersItemsSets.put(i, new IntOpenHashSet());
        }
    }
}
