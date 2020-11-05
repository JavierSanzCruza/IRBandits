/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.knn.item;

import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AbstractSimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.FastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.TransposedUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.*;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

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
    /**
     * Preference data.
     */
    protected final AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData;

    private final boolean ignoreZeros;

    /**
     * Constructor.
     *
     * @param uIndex      User index.
     * @param iIndex      Item index.
     * @param hasRating   True if we must ignore unknown items when updating.
     * @param ignoreZeros True if we ignore zero ratings when updating.
     * @param userK       Number of items rated by the user to pick.
     * @param itemK       Number of users rated by the item to pick.
     * @param sim         Updateable similarity
     */
    public AbstractInteractiveItemBasedKNN(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, boolean ignoreZeros, int userK, int itemK, UpdateableSimilarity sim, AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData)
    {
        super(uIndex, iIndex, hasRating);
        this.sim = sim;
        this.userK = userK;
        this.itemK = (itemK > 0) ? itemK : iIndex.numItems();
        this.itemList = new IntArrayList();
        uIndex.getAllUidx().forEach(itemList::add);
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
        this.retrievedData = retrievedData;
    }

    /**
     * Constructor.
     *
     * @param uIndex      User index.
     * @param iIndex      Item index.
     * @param hasRating   True if we must ignore unknown items when updating.
     * @param ignoreZeros True if we ignore zero ratings when updating.
     * @param userK       Number of items rated by the user to pick.
     * @param itemK       Number of users rated by the item to pick.
     * @param sim         Updateable similarity
     */
    public AbstractInteractiveItemBasedKNN(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int rngSeed, boolean ignoreZeros, int userK, int itemK, UpdateableSimilarity sim, AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData)
    {
        super(uIndex, iIndex, hasRating, rngSeed);
        this.sim = sim;
        this.userK = userK;
        this.itemK = (itemK > 0) ? itemK : iIndex.numItems();
        this.itemList = new IntArrayList();
        uIndex.getAllUidx().forEach(itemList::add);
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
        this.retrievedData = retrievedData;
    }

    @Override
    public void init()
    {
        super.init();
        this.sim.initialize();
        this.retrievedData.clear();
    }

    /*@Override
    public void init(FastPreferenceData<U,I> prefData)
    {
        this.retrievedData.clear();
        prefData.getUidxWithPreferences().forEach(uidx -> prefData.getUidxPreferences(uidx).forEach(i -> retrievedData.updateRating(uidx, i.v1, i.v2)));
        this.sim.initialize(new TransposedUpdateablePreferenceData<>(retrievedData));
    }*/

    @Override
    public void init(Stream<FastRating> values)
    {
        this.retrievedData.clear();
        values.forEach(triplet -> this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), triplet.value()));
        this.sim.initialize(new TransposedUpdateablePreferenceData<>(retrievedData));
    }


    @Override
    public int next(int uidx, IntList availability)
    {
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }

        // Shuffle the order of the items.
        Collections.shuffle(itemList, neighborUntie);
        PriorityQueue<Tuple2id> firstHeap = new PriorityQueue<>(this.numItems(), comp);

        if (this.userK == 0)
        {
            this.retrievedData.getUidxPreferences(uidx).filter(iv -> !ignoreZeros || iv.v2 > 0).forEach(firstHeap::add);
        }
        else
        {
            this.retrievedData.getUidxPreferences(uidx).forEach(iv ->
            {
                if (!this.ignoreZeros || iv.v2 > 0.0)
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
            int idx = rng.nextInt(availability.size());
            return availability.get(idx);
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
                if (availability.contains(jv.v1))
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
            if (!availability.contains(iidx))
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
            return availability.get(rng.nextInt(availability.size()));
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
     * @return the score value.
     */
    protected abstract double score(int vidx, double rating);

    @Override
    public void update(int uidx, int iidx, double value)
    {
        boolean hasRating = false;
        double oldValue = 0;

        if(this.retrievedData.numItems(uidx) > 0 && this.retrievedData.numUsers(iidx) > 0)
        {
            Optional<IdxPref> opt = this.retrievedData.getPreference(uidx, iidx);
            hasRating = opt.isPresent();
            if(hasRating)
            {
                oldValue = opt.get().v2();
            }
        }

        if(!hasRating)
        {
            this.retrievedData.getUidxPreferences(uidx).forEach(jidx -> this.sim.update(iidx, jidx.v1, uidx, value, jidx.v2));
            this.sim.updateNorm(iidx, value);
            this.retrievedData.updateRating(uidx, iidx, value);
        }
        else
        {
            if(this.retrievedData.updateRating(uidx, iidx, value))
            {
                Optional<IdxPref> opt = this.retrievedData.getPreference(uidx, iidx);
                if(opt.isPresent())
                {
                    double newValue = opt.get().v2;

                    this.sim.updateNormDel(iidx, oldValue);
                    this.sim.updateNorm(iidx, newValue);

                    double finalOldValue = oldValue;
                    this.retrievedData.getUidxPreferences(uidx).filter(jidx -> jidx.v1 != iidx).forEach(jidx ->
                    {
                        this.sim.updateDel(iidx, jidx.v1, uidx, finalOldValue, jidx.v2);
                        this.sim.update(iidx, jidx.v1, uidx, newValue, jidx.v2);
                    });
                }
            }
        }
    }
}
