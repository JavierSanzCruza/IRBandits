/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.knn.user;

import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AbstractSimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import it.unimi.dsi.fastutil.ints.*;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.*;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

/**
 * Abstract version of an interactive user-based kNN algorithm
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class AbstractInteractiveUserBasedKNN<U, I> extends InteractiveRecommender<U, I>
{
    /**
     * Updateable similarity.
     */
    protected final UpdateableSimilarity sim;

    /**
     * Preference data.
     */
    protected final AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData;

    /**
     * Random number generator to untie neighbors.
     */
    private final Random neighborUntie = new Random();
    /**
     * Number of neighbors to use.
     */
    private final int k;
    /**
     * Neighbor comparator.
     */
    private final Comparator<Tuple2id> comp;
    /**
     * Shuffled list of users.
     */
    private final IntList userList;

    /**
     * This variable gives more importance to irrelevant items for the final item selection
     * than to unknown items when it is false. Otherwise, they will be given the same
     * importance.
     */
    private final boolean ignoreZeros;

    /**
     * Constructor.
     *
     * @param uIndex      User index.
     * @param iIndex      Item index.
     * @param hasRating   True if we must ignore unknown items when updating.
     * @param ignoreZeros True if we ignore zero ratings when updating.
     * @param k           Number of neighbors to use.
     * @param sim         Updateable similarity
     */
    public AbstractInteractiveUserBasedKNN(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, boolean ignoreZeros, int k, UpdateableSimilarity sim, AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData)
    {
        super(uIndex, iIndex, hasRating);

        // Store the similarity we want to use.
        this.sim = sim;

        // Fix the number of neighbors to take
        this.k = (k > 0) ? k : uIndex.numUsers();
        this.userList = new IntArrayList();

        // Fix a preference order between the users.
        uIndex.getAllUidx().forEach(userList::add);
        this.comp = (Tuple2id x, Tuple2id y) ->
        {
            int value = (int) Math.signum(x.v2 - y.v2);
            if (value == 0)
            {
                return userList.indexOf(x.v1) - userList.indexOf(y.v1);
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
     * @param k           Number of neighbors to use.
     * @param sim         Updateable similarity
     */
    public AbstractInteractiveUserBasedKNN(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int rngSeed, boolean ignoreZeros, int k, UpdateableSimilarity sim, AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData)
    {
        super(uIndex, iIndex, hasRating, rngSeed);

        // Store the similarity we want to use.
        this.sim = sim;

        // Fix the number of neighbors to take
        this.k = (k > 0) ? k : uIndex.numUsers();
        this.userList = new IntArrayList();

        // Fix a preference order between the users.
        uIndex.getAllUidx().forEach(userList::add);
        this.comp = (Tuple2id x, Tuple2id y) ->
        {
            int value = (int) Math.signum(x.v2 - y.v2);
            if (value == 0)
            {
                return userList.indexOf(x.v1) - userList.indexOf(y.v1);
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
        this.retrievedData.clear();
        this.sim.initialize();
    }

    /*@Override
    public void init(FastPreferenceData<U,I> prefData)
    {
        this.retrievedData.clear();
        prefData.getUidxWithPreferences().forEach(uidx -> prefData.getUidxPreferences(uidx).forEach(i -> retrievedData.updateRating(uidx, i.v1, i.v2)));
        this.sim.initialize(retrievedData);
    }*/

    @Override
    public void init(Stream<FastRating> values)
    {
        super.init();
        this.retrievedData.clear();
        values.forEach(triplet -> this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), triplet.value()));
        this.sim.initialize(retrievedData);
    }

    @Override
    public int next(int uidx, IntList availability)
    {
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }

        // Shuffle the order of users.
        Collections.shuffle(userList, neighborUntie);

        // Obtain the top-k best neighbors for user uidx.
        PriorityQueue<Tuple2id> neighborHeap = new PriorityQueue<>(k, comp);
        this.sim.similarElems(uidx).forEach(vidx ->
        {
            double s = vidx.v2;
            if (neighborHeap.size() < k)
            {
                neighborHeap.add(new Tuple2id(vidx.v1, vidx.v2));
            }
            else
            {
                assert neighborHeap.peek() != null;
                if (neighborHeap.peek().v2 <= s)
                {
                    neighborHeap.poll();
                    neighborHeap.add(new Tuple2id(vidx.v1, s));
                }
            }
        });

        // If no neighbor has been selected, then, select an item at random.
        if (neighborHeap.isEmpty())
        {
            return availability.get(rng.nextInt(availability.size()));
        }

        // Generate the scores for the different items.
        Int2DoubleOpenHashMap itemScores = new Int2DoubleOpenHashMap();
        itemScores.defaultReturnValue(0.0);
        IntSet availableItems = new IntOpenHashSet(availability);

        // Then, generate scores for the different items.
        while (!neighborHeap.isEmpty())
        {
            Tuple2id neigh = neighborHeap.poll();

            retrievedData.getUidxPreferences(neigh.v1).filter(vs -> availableItems.contains(vs.v1)).forEach(vs ->
            {
                double p = neigh.v2 * this.score(neigh.v1, vs.v2);
                if (!ignoreZeros || p > 0)
                {
                    itemScores.addTo(vs.v1, p);
                }
            });
        }

        // Select the best item.
        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();

        for (int iidx : itemScores.keySet())
        {
            double val = itemScores.get(iidx);

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

    @Override
    public IntList next(int uidx, IntList availability, int k)
    {
        if (availability == null || availability.isEmpty())
        {
            return new IntArrayList();
        }

        IntList top = new IntArrayList();
        int num = Math.min(availability.size(), k);

        // Shuffle the order of users.
        Collections.shuffle(userList, neighborUntie);

        // Obtain the top-k best neighbors for user uidx.
        PriorityQueue<Tuple2id> neighborHeap = new PriorityQueue<>(k, comp);
        this.sim.similarElems(uidx).forEach(vidx ->
        {
            double s = vidx.v2;
            if (neighborHeap.size() < k)
            {
                neighborHeap.add(new Tuple2id(vidx.v1, vidx.v2));
            }
            else
            {
                assert neighborHeap.peek() != null;
                if (neighborHeap.peek().v2 <= s)
                {
                    neighborHeap.poll();
                    neighborHeap.add(new Tuple2id(vidx.v1, s));
                }
            }
        });

        // If no neighbor has been selected, then, select an item at random.
        if (!neighborHeap.isEmpty())
        {
            // Generate the scores for the different items.
            Int2DoubleOpenHashMap itemScores = new Int2DoubleOpenHashMap();
            itemScores.defaultReturnValue(0.0);
            IntSet availableItems = new IntOpenHashSet(availability);

            // Then, generate scores for the different items.
            while (!neighborHeap.isEmpty())
            {
                Tuple2id neigh = neighborHeap.poll();

                retrievedData.getUidxPreferences(neigh.v1).filter(vs -> availableItems.contains(vs.v1)).forEach(vs ->
                {
                    double p = neigh.v2 * this.score(neigh.v1, vs.v2);
                    if (!ignoreZeros || p > 0)
                    {
                        itemScores.addTo(vs.v1, p);
                    }
                });
            }

            PriorityQueue<Tuple2id> queue = new PriorityQueue<>(num, Comparator.comparingDouble(x -> x.v2));
            for(int iidx : itemScores.keySet())
            {
                double val = itemScores.get(iidx);
                if(queue.size() < num)
                {
                    queue.add(new Tuple2id(iidx, val));
                }
                else
                {
                    Tuple2id newTuple = new Tuple2id(iidx, val);
                    if(queue.comparator().compare(queue.peek(), newTuple) < 0)
                    {
                        queue.poll();
                        queue.add(newTuple);
                    }
                }
            }

            while(!queue.isEmpty())
            {
                top.add(0, queue.poll().v1);
            }
        }

        while(top.size() < num)
        {
            int idx = rng.nextInt(availability.size());
            int item = availability.get(idx);
            if(!top.contains(item)) top.add(item);
        }

        return top;
    }

    /**
     * Scoring function.
     *
     * @param vidx   Identifier of the neighbor user.
     * @param rating The rating value.
     * @return the score.
     */
    protected abstract double score(int vidx, double rating);

    @Override
    public void update(int uidx, int iidx, double value)
    {
        double newValue;
        if(!Double.isNaN(value))
            newValue = value;
        else if(!this.ignoreNotRated)
            newValue = Constants.NOTRATEDNOTIGNORED;
        else
            return;


        boolean hasRating = false;
        double oldValue = 0;
        // First, we find whether we have a rating or not:
        if(this.retrievedData.numItems(uidx) > 0 && this.retrievedData.numUsers(iidx) > 0)
        {
            Optional<IdxPref> opt = this.retrievedData.getPreference(uidx, iidx);
            hasRating = opt.isPresent();
            if(hasRating)
            {
                oldValue = opt.get().v2;
            }
        }

        if(!hasRating)
        {
            this.retrievedData.updateRating(uidx, iidx, newValue);
            this.retrievedData.getIidxPreferences(iidx).forEach(vidx -> this.sim.update(uidx, vidx.v1, iidx, newValue, vidx.v2));
            this.sim.updateNorm(uidx, newValue);
        }
        else
        {
            if(this.retrievedData.updateRating(uidx, iidx, newValue))
            {
                Optional<IdxPref> opt = this.retrievedData.getPreference(uidx, iidx);
                if(opt.isPresent())
                {
                    double auxNewValue = opt.get().v2;
                    this.sim.updateNormDel(uidx, oldValue);
                    this.sim.updateNorm(uidx, auxNewValue);

                    double finalOldValue = oldValue;
                    this.retrievedData.getIidxPreferences(iidx).filter(vidx -> vidx.v1 != uidx).forEach(vidx ->
                    {
                        this.sim.updateDel(uidx, vidx.v1, iidx, finalOldValue, vidx.v2);
                        this.sim.update(uidx, vidx.v1, iidx, auxNewValue, vidx.v2);
                    });
                }
            }
        }
    }
}
