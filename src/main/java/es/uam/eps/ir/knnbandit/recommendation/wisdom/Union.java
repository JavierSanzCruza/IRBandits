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
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.*;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.function.DoublePredicate;
import java.util.stream.Stream;

/**
 * Recommends items according to the number of different items liked by the
 * users that have enjoyed the candidate item.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class Union<U,I> extends AbstractInteractiveRecommender<U,I>
{
    /**
     * The recommendation scores for each of the items.
     */
    private final Int2DoubleMap itemScores;
    /**
     * The set of items at distance two of the candidate item in the user-item bipartite graph.
     */
    private final Int2ObjectMap<IntSet> relatedItems;
    /**
     * Preference data.
     */
    protected final SimpleFastUpdateablePreferenceData<U,I> retrievedData;
    /**
     * Checks whether a rating is positive or not.
     */
    private final DoublePredicate relevanceChecker;

    /**
     * Constructor.
     * @param uIndex           the user index.
     * @param iIndex           the item index.
     * @param relevanceChecker checks whether a rating is relevant or not.
     */
    public Union(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, DoublePredicate relevanceChecker)
    {
        super(uIndex, iIndex, true);
        this.itemScores = new Int2DoubleOpenHashMap();
        this.relatedItems = new Int2ObjectOpenHashMap<>();
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.relevanceChecker = relevanceChecker;
    }

    /**
     * Constructor.
     * @param uIndex           the user index.
     * @param iIndex           the item index.
     * @param relevanceChecker checks whether a rating is relevant or not.
     * @param rngSeed          random number generator seed.
     */
    public Union(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, int rngSeed, DoublePredicate relevanceChecker)
    {
        super(uIndex, iIndex, true, rngSeed);
        this.itemScores = new Int2DoubleOpenHashMap();
        this.relatedItems = new Int2ObjectOpenHashMap<>();
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.relevanceChecker = relevanceChecker;
    }

    @Override
    public void init()
    {
        super.init();

        this.retrievedData.clear();
        this.itemScores.clear();
        this.relatedItems.clear();
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();

        // We do first store the retrieved values:
        values.filter(triplet -> relevanceChecker.test(triplet.value())).forEach(triplet ->
            this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), triplet.value()));

        // Then:
        retrievedData.getIidxWithPreferences().forEach(iidx ->
        {
            IntSet set = new IntOpenHashSet();
            retrievedData.getIidxPreferences(iidx).map(u -> u.v1).forEach(uidx ->
                retrievedData.getUidxPreferences(uidx).map(j -> j.v1).forEach(set::add)
            );
            this.relatedItems.put(iidx, set);
            this.itemScores.put(iidx, set.size() + 0.0);
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
                double value = this.itemScores.getOrDefault(item, 0.0);

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
            else if(size == 0)
            {
                nextItem = availability.get(rng.nextInt(availability.size()));
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
                double value = this.itemScores.getOrDefault(iidx, 0.0);

                if(queue.size() < num)
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
        // We do not need to update anything: it already has.
        if(this.retrievedData.getPreference(uidx, iidx).isPresent()) return;

        // Initialize the set of related items for the iidx element.
        if(!this.relatedItems.containsKey(iidx))
        {
            this.relatedItems.put(iidx, new IntOpenHashSet());
        }

        IntSet iItems = this.relatedItems.get(iidx);

        // Update the scores:
        double score = this.retrievedData.getUidxPreferences(uidx).mapToDouble(j ->
        {
            int jidx = j.v1;

            // First, update the score for the item j.
            if(!relatedItems.get(jidx).contains(iidx))
            {
                relatedItems.get(jidx).add(iidx);
                double jScore = itemScores.get(jidx) + 1.0;
                itemScores.put(jidx, jScore);
            }

            // Then, update the score for item i.
            if(!iItems.contains(jidx))
            {
                iItems.add(jidx);
                return 1.0;
            }

            return 0.0;
        }).sum();

        // As uidx has not previously rated iidx, then, we add it.
        iItems.add(iidx);
        score += 1.0;

        double old = this.itemScores.getOrDefault(iidx, 0.0);
        this.itemScores.put(iidx, old + score);
        this.retrievedData.updateRating(uidx, iidx, value);
    }
}