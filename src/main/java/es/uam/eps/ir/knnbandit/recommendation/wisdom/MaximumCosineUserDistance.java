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
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import it.unimi.dsi.fastutil.ints.*;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.function.DoublePredicate;
import java.util.stream.Stream;

/**
 * Recommender that uses the maximum distance between pairs of users who have rated
 * the item to compute the recommendation.
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class MaximumCosineUserDistance<U,I> extends AbstractInteractiveRecommender<U,I>
{
    /**
     * The scores for each item.
     */
    private final Int2DoubleMap itemScores;
    /**
     * The actual similarities.
     */
    private final Int2ObjectMap<Int2DoubleOpenHashMap> sims;
    /**
     * The norms of the users.
     */
    private final Int2DoubleMap userNorms;


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
    public MaximumCosineUserDistance(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, DoublePredicate relevanceChecker)
    {
        super(uIndex, iIndex, true);
        this.sims = new Int2ObjectOpenHashMap<>();
        sims.defaultReturnValue(new Int2DoubleOpenHashMap());
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.userNorms = new Int2DoubleOpenHashMap();
        userNorms.defaultReturnValue(0.0);
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
    public MaximumCosineUserDistance(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, int rngSeed, DoublePredicate relevanceChecker)
    {
        super(uIndex, iIndex, true, rngSeed);
        this.sims = new Int2ObjectOpenHashMap<>();
        sims.defaultReturnValue(new Int2DoubleOpenHashMap());
        this.userNorms = new Int2DoubleOpenHashMap();
        userNorms.defaultReturnValue(0.0);
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.relevanceChecker = relevanceChecker;
        this.itemScores = new Int2DoubleOpenHashMap();
        this.itemScores.defaultReturnValue(0.0);
    }

    @Override
    public void init()
    {
        this.retrievedData.clear();
        this.sims.clear();
        this.itemScores.clear();
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.retrievedData.clear();
        this.itemScores.clear();

        values.filter(triplet -> relevanceChecker.test(triplet.value())).forEach(triplet ->
            this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), triplet.value()));

        // Initialize the cosine similarity between users:
        this.retrievedData.getAllUidx().forEach(uidx ->
        {
            Int2DoubleOpenHashMap map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.sims.put(uidx, map);

            double uNorm = retrievedData.getUidxPreferences(uidx).mapToDouble(iidx ->
            {
                retrievedData.getIidxPreferences(iidx.v1).forEach(vidx -> this.sims.get(uidx).addTo(vidx.v1,iidx.v2*vidx.v2));
                return iidx.v2*iidx.v2;
            }).sum();

            this.userNorms.put(uidx, uNorm);
         });

        // Now, we initialize the scores for the different items.
        this.retrievedData.getIidxWithPreferences().forEach(iidx ->
        {
            IntIterator iterator1 = this.retrievedData.getIidxUidxs(iidx);

            double value = 0.0;
            while(iterator1.hasNext())
            {
                IntIterator iterator2 = this.retrievedData.getIidxUidxs(iidx);
                int uidx = iterator1.nextInt();
                while(iterator2.hasNext())
                {
                    int vidx = iterator2.nextInt();
                    if(vidx <= uidx) continue;

                    double val = 1.0 - this.sims.get(uidx).getOrDefault(vidx, 0.0) / Math.sqrt(this.userNorms.get(uidx)*this.userNorms.get(vidx));
                    if(val >= value)
                        value = val;
                }
                this.itemScores.put(iidx, value);
            }
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

        double userNorm = this.userNorms.getOrDefault(uidx, 0.0);
        double oldVal = this.retrievedData.getPreference(uidx, iidx).orElse(new IdxPref(iidx, 0.0)).v2;
        double newNorm = userNorm + 2*oldVal*value + value*value;

        // Now, we update the corresponding similarities:
        // We do only have to update the similarities between uidx and the items who have rated iidx, so:

        this.retrievedData.getIidxPreferences(iidx).filter(v -> v.v1 != uidx).forEach(v ->
        {
            int vidx = v.v1;
            double sim = this.sims.get(vidx).getOrDefault(uidx, 0.0);
            sim += v.v2*value;

            this.sims.get(vidx).put(uidx, sim);
            this.sims.get(uidx).put(vidx, sim);
        });

        this.userNorms.put(uidx, newNorm);
        this.retrievedData.updateRating(uidx, iidx, value);

        // Now, we find the maximum values for the items who uidx has rated (plus iidx):
        this.retrievedData.getUidxPreferences(uidx).forEach(j ->
        {
            int jidx = j.v1;
            IntIterator iterator1 = this.retrievedData.getIidxUidxs(jidx);

            double v = 0.0;
            while(iterator1.hasNext())
            {
                IntIterator iterator2 = this.retrievedData.getIidxUidxs(jidx);
                int widx = iterator1.nextInt();
                while(iterator2.hasNext())
                {
                    int vidx = iterator2.nextInt();
                    if(vidx <= widx) continue;

                    double val = 1.0 - this.sims.get(widx).getOrDefault(vidx, 0.0) / Math.sqrt(this.userNorms.get(widx)*this.userNorms.get(vidx));
                    if(val >= v)
                        v = val;
                }
                this.itemScores.put(jidx, v);
            }
        });
    }
}