/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.wisdom;

import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.function.DoublePredicate;
import java.util.stream.Stream;

/**
 * Recommender that uses the average distance between pairs of users who have rated
 * the item to compute the recommendation.
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class AverageCosineUserDistance<U,I> extends AbstractInteractiveRecommender<U,I>
{
    /**
     * The cosine similarity between users.
     */
    private final VectorCosineSimilarity sim;
    /**
     * The scores for each item.
     */
    private final Int2DoubleMap itemScores;

    /**
     * Predicate for checking the relevance of a rating.
     */
    private final DoublePredicate relevanceChecker;

    /**
     * Preference data.
     */
    protected final SimpleFastUpdateablePreferenceData<U,I> retrievedData;

    /**
     * Constructor.
     * @param uIndex user index.
     * @param iIndex item index.
     * @param relevanceChecker checks the relevance of a rating.
     */
    public AverageCosineUserDistance(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, DoublePredicate relevanceChecker)
    {
        super(uIndex, iIndex, true);
        this.sim = new VectorCosineSimilarity(uIndex.numUsers());
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.relevanceChecker = relevanceChecker;
        this.itemScores = new Int2DoubleOpenHashMap();
        this.itemScores.defaultReturnValue(0.0);
    }

    /**
     * Constructor.
     * @param uIndex user index.
     * @param iIndex item index.
     * @param rngSeed random number generator seed.
     * @param relevanceChecker checks the relevance of a rating.
     */
    public AverageCosineUserDistance(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, int rngSeed, DoublePredicate relevanceChecker)
    {
        super(uIndex, iIndex, true, rngSeed);
        this.sim = new VectorCosineSimilarity(uIndex.numUsers());
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.relevanceChecker = relevanceChecker;
        this.itemScores = new Int2DoubleOpenHashMap();
        this.itemScores.defaultReturnValue(0.0);
    }

    @Override
    public void init()
    {
        this.retrievedData.clear();
        this.sim.initialize();
        this.itemScores.clear();
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.retrievedData.clear();
        this.itemScores.clear();

        values.filter(triplet -> relevanceChecker.test(triplet.value())).forEach(triplet ->
            this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), triplet.value()));
        this.sim.initialize(retrievedData);

        // Now, we initialize the values for the items.
        this.retrievedData.getIidxWithPreferences().forEach(iidx ->
        {
            double iidxVal = 2.0*this.retrievedData.getIidxPreferences(iidx).map(u -> u.v1).mapToDouble(uidx1 ->
                this.retrievedData.getIidxPreferences(iidx).filter(u -> uidx1 < u.v1).map(u -> u.v1).mapToDouble(uidx2 ->
                    1.0-this.sim.similarity(uidx1, uidx2)
                ).sum()
            ).sum();

            this.itemScores.put(iidx, iidxVal);
        });
    }

    @Override
    public int next(int uidx, IntList availability)
    {
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }
        else
        {
            double val = Double.NEGATIVE_INFINITY;
            IntList top = new IntArrayList();

            for (int item : availability)
            {
                double value = itemScores.getOrDefault(item, itemScores.defaultReturnValue());
                int size = retrievedData.numUsers(item);
                if(size > 1)
                    value /= (size + 0.0)*(size + 1.0);

                if (value > val)
                {
                    val = value;
                    top = new IntArrayList();
                    top.add(item);
                }
                else if (value == val)
                {
                    top.add(item);
                }
            }

            int nextItem;
            int size = top.size();
            if (size == 1)
            {
                nextItem = top.get(0);
            }
            else
            {
                nextItem = top.get(rng.nextInt(size));
            }

            return nextItem;
        }
    }

    @Override
    public IntList next(int uidx, IntList availability, int k)
    {
        if (availability == null || availability.isEmpty())
        {
            return new IntArrayList();
        }
        else
        {
            IntList top = new IntArrayList();

            int num = Math.min(availability.size(), k);
            PriorityQueue<Tuple2id> queue = new PriorityQueue<>(num, Comparator.comparingDouble(x -> x.v2));

            for (int iidx : availability)
            {

                double value = itemScores.getOrDefault(iidx, itemScores.defaultReturnValue());
                int size = retrievedData.numUsers(iidx);
                if (size > 1)
                    value /= (size + 0.0) * (size + 1.0);

                if (queue.size() < num)
                {
                    queue.add(new Tuple2id(iidx, value));
                }
                else
                {
                    Tuple2id newTuple = new Tuple2id(iidx, value);
                    if (queue.comparator().compare(queue.peek(), newTuple) < 0)
                    {
                        queue.poll();
                        queue.add(newTuple);
                    }
                }
            }

            while (!queue.isEmpty())
            {
                top.add(0, queue.poll().v1);
            }

            return top;
        }
    }

    @Override
    public void fastUpdate(int uidx, int iidx, double value)
    {
        if(!relevanceChecker.test(value)) return;

        // Step 1: For all the items which have been positively rated by uidx,
        // we update the value by substracting the distances between uidx and the rest.
        this.retrievedData.getUidxPreferences(uidx).forEach(i ->
        {
            int jidx = i.v1;
            double sub = this.retrievedData.getIidxPreferences(jidx).map(v -> v.v1).mapToDouble(vidx -> 1.0 - sim.similarity(uidx, vidx)).sum();
            this.itemScores.put(jidx, this.itemScores.get(jidx) - sub);
        });

        // Step 2: Update the similarity and the preference data:
        this.retrievedData.updateRating(uidx, iidx, value);
        this.sim.updateNorm(uidx, value);
        this.retrievedData.getIidxPreferences(iidx).forEach(vidx -> this.sim.update(uidx, vidx.v1, iidx, value, vidx.v2));

        // Step 3: For all the items which have been positively rated by uidx (including iidx),
        // we update their value by adding the distances between uidx and the rest.
        this.retrievedData.getUidxPreferences(uidx).forEach(i ->
        {
            int jidx = i.v1;
            double add = this.retrievedData.getIidxPreferences(jidx).map(v -> v.v1).mapToDouble(vidx -> 1.0 - sim.similarity(uidx, vidx)).sum();
            this.itemScores.put(jidx, this.itemScores.get(jidx) + add);
        });
    }
}