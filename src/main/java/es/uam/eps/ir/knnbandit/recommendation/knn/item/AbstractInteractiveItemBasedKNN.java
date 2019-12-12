/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.knn.item;

import es.uam.eps.ir.knnbandit.data.preference.fast.TransposedUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.*;

/**
 * Abstract version of an interactive item-based kNN algorithm
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class AbstractInteractiveItemBasedKNN<U, I> extends InteractiveRecommender<U, I>
{
    /**
     * Updateable similarity.
     */
    protected final UpdateableSimilarity sim;
    /**
     * Random number generator to untie neighbors.
     */
    private final Random neighborUntie = new Random();

    /**
     * Number of rated items of the user to pick
     */
    private final int userK;

    /**
     * Number of neighbors of the item to pick
     */
    private final int itemK;

    /**
     * Neighbor comparator.
     */
    private final Comparator<Tuple2id> comp;
    /**
     * Shuffled list of users.
     */
    private final IntList itemList;

    private final boolean ignoreZeros;

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      Preference data.
     * @param ignoreUnknown True if we must ignore unknown items when updating.
     * @param ignoreZeros   True if we ignore zero ratings when updating.
     * @param userK         Number of items rated by the user to pick.
     * @param itemK         Number of users rated by the item to pick.
     * @param sim           Updateable similarity
     */
    public AbstractInteractiveItemBasedKNN(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreUnknown, boolean ignoreZeros, int userK, int itemK, UpdateableSimilarity sim)
    {
        super(uIndex, iIndex, prefData, ignoreUnknown);
        this.sim = sim;
        this.userK = userK;
        this.itemK = (itemK > 0) ? itemK : prefData.numItems();
        this.itemList = new IntArrayList();
        uIndex.getAllUidx().forEach(uidx -> itemList.add(uidx));
        this.comp = (Tuple2id x, Tuple2id y) ->
        {
            int value = (int) Math.signum(x.v2 - y.v2);
            if (value == 0)
            {
                return itemList.indexOf(x.v1) - itemList.indexOf(y.v1);
            }
            return value;
        };
        this.ignoreZeros = ignoreZeros;
    }

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      Preference data.
     * @param ignoreUnknown True if we must ignore unknown items when updating.
     * @param ignoreZeros   True if we ignore zero ratings when updating.
     * @param notReciprocal True if we do not recommend reciprocal social links, false otherwise.
     * @param userK         Number of items rated by the user to pick.
     * @param itemK         Number of users rated by the item to pick.
     * @param sim           Updateable similarity
     */
    public AbstractInteractiveItemBasedKNN(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreUnknown, boolean ignoreZeros, boolean notReciprocal, int userK, int itemK, UpdateableSimilarity sim)
    {
        super(uIndex, iIndex, prefData, ignoreUnknown, notReciprocal);
        this.sim = sim;
        this.userK = userK;
        this.itemK = (itemK > 0) ? itemK : prefData.numItems();
        this.itemList = new IntArrayList();
        iIndex.getAllIidx().forEach(uidx -> itemList.add(uidx));
        this.comp = (Tuple2id x, Tuple2id y) ->
        {
            int value = (int) Math.signum(x.v2 - y.v2);
            if (value == 0)
            {
                return itemList.indexOf(x.v1) - itemList.indexOf(y.v1);
            }
            return value;
        };
        this.ignoreZeros = ignoreZeros;
    }

    @Override
    protected void initializeMethod()
    {
        this.sim.initialize(new TransposedUpdateablePreferenceData<>(this.trainData));
    }

    @Override
    public int next(int uidx)
    {
        IntList list = this.availability.get(uidx);
        if (list == null || list.isEmpty())
        {
            return -1;
        }

        // Shuffle the order of users.
        Collections.shuffle(itemList, neighborUntie);

        PriorityQueue<Tuple2id> firstHeap = new PriorityQueue<>(this.numItems(), comp);

        if (this.userK == 0)
        {
            this.trainData.getUidxPreferences(uidx).filter(iv -> !ignoreZeros || iv.v2 > 0).forEach(iv -> firstHeap.add(iv));
        }
        else
        {
            this.trainData.getUidxPreferences(uidx).forEach(iv ->
            {
                if(!this.ignoreZeros || iv.v2 > 0.0)
                {
                    if (firstHeap.size() == userK)
                    {
                        firstHeap.add(iv);
                        firstHeap.poll();
                    }
                    else
                    {
                        firstHeap.add(iv);
                    }
                }
            });
        }

        if (firstHeap.isEmpty())  // If the user has not rated any item, return an item at random.
        {
            int idx = rng.nextInt(list.size());
            return list.get(idx);
        }

        Int2DoubleOpenHashMap map = new Int2DoubleOpenHashMap();
        map.defaultReturnValue(0.0);
        firstHeap.forEach(iv ->
        {
            int jidx = iv.v1;
            double ruj = iv.v2;

            PriorityQueue<Tuple2id> heap = new PriorityQueue<>(itemK, comp);
            this.sim.similarElems(jidx).forEach(jv ->
            {
                if(list.contains(jv.v1))
                {
                    if (heap.size() == itemK)
                    {
                        heap.add(jv);
                        heap.poll();
                    }
                    else
                    {
                        heap.add(jv);
                    }
                }
            });

            for (Tuple2id jv : heap)
            {
                map.addTo(jv.v1, jv.v2 * ruj);
            }
        });

        // Select the best item.
        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();

        for (int iidx : map.keySet())
        {
            double val = map.get(iidx);
            if (!list.contains(iidx))
            {
                continue;
            }

            if (top.isEmpty() || val > max)
            {
                top = new IntArrayList();
                max = val;
                top.add(iidx);
            }
            else if (val == max)
            {
                top.add(iidx);
            }
        }

        int topSize = top.size();
        if (top.isEmpty())
        {
            return list.get(rng.nextInt(list.size()));
        }
        else if (topSize == 1)
        {
            return top.get(0);
        }
        return top.get(rng.nextInt(topSize));
    }

    /**
     * Scoring function.
     *
     * @param vidx   Identifier of the neighbor user.
     * @param rating The rating value.
     *
     * @return
     */
    protected abstract double score(int vidx, double rating);

    @Override
    public void updateMethod(List<Tuple3<Integer, Integer, Double>> train)
    {
        this.sim.initialize(new TransposedUpdateablePreferenceData<>(this.trainData));
    }
}
