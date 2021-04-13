/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.wisdom;

import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AdditiveRatingFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import it.unimi.dsi.fastutil.ints.*;
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
public class ItemCentroidDistance<U,I> extends AbstractInteractiveRecommender<U,I>
{
    /**
     * The norm of the users.
     */
    private final Int2DoubleMap userNorm;
    /**
     * The norm of the item centroids.
     */
    private final Int2DoubleMap itemNorm;
    /**
     * The item centroids.
     */
    private final Int2ObjectMap<Int2DoubleMap> itemVectors;
    /**
     * Similarity between users and items.
     */
    private final Int2ObjectMap<Int2DoubleMap> item2userSimilarity;
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
    protected final AdditiveRatingFastUpdateablePreferenceData<U,I> retrievedData;

    /**
     * Constructor.
     * @param uIndex user index.
     * @param iIndex item index.
     * @param relevanceChecker checks the relevance of a rating.
     */
    public ItemCentroidDistance(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, DoublePredicate relevanceChecker)
    {
        super(uIndex, iIndex, true);
        this.retrievedData = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.relevanceChecker = relevanceChecker;
        this.itemScores = new Int2DoubleOpenHashMap();
        this.itemScores.defaultReturnValue(0.0);

        this.userNorm = new Int2DoubleOpenHashMap();
        this.userNorm.defaultReturnValue(0.0);
        this.itemNorm = new Int2DoubleOpenHashMap();
        this.itemNorm.defaultReturnValue(0.0);

        this.itemVectors = new Int2ObjectOpenHashMap<>();
        this.itemVectors.defaultReturnValue(new Int2DoubleOpenHashMap());

        this.item2userSimilarity = new Int2ObjectOpenHashMap<>();
        this.item2userSimilarity.defaultReturnValue(new Int2DoubleOpenHashMap());
    }

    /**
     * Constructor.
     * @param uIndex user index.
     * @param iIndex item index.
     * @param rngSeed random number generator seed.
     * @param relevanceChecker checks the relevance of a rating.
     */
    public ItemCentroidDistance(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, int rngSeed, DoublePredicate relevanceChecker)
    {
        super(uIndex, iIndex, true, rngSeed);
        this.retrievedData = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.relevanceChecker = relevanceChecker;
        this.itemScores = new Int2DoubleOpenHashMap();
        this.itemScores.defaultReturnValue(0.0);

        this.userNorm = new Int2DoubleOpenHashMap();
        this.userNorm.defaultReturnValue(0.0);
        this.itemNorm = new Int2DoubleOpenHashMap();
        this.itemNorm.defaultReturnValue(0.0);

        this.itemVectors = new Int2ObjectOpenHashMap<>();
        this.itemVectors.defaultReturnValue(new Int2DoubleOpenHashMap());

        this.item2userSimilarity = new Int2ObjectOpenHashMap<>();
        this.item2userSimilarity.defaultReturnValue(new Int2DoubleOpenHashMap());
    }

    @Override
    public void init()
    {
        super.init();

        this.retrievedData.clear();
        this.itemScores.clear();
        this.userNorm.clear();
        this.itemNorm.clear();
        this.itemVectors.clear();
        this.item2userSimilarity.clear();
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();

        values.filter(triplet -> relevanceChecker.test(triplet.value())).forEach(triplet ->
            this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), triplet.value()));

        // First, we compute the modules for the users.
        this.retrievedData.getUidxWithPreferences().forEach(uidx -> userNorm.put(uidx, retrievedData.getUidxPreferences(uidx).mapToDouble(i -> i.v2*i.v2).sum()));

        retrievedData.getIidxWithPreferences().forEach(iidx ->
        {
            // Step 1: Find the item centroid
            Int2DoubleOpenHashMap centroid = new Int2DoubleOpenHashMap();
            centroid.defaultReturnValue(0.0);
            long count = retrievedData.getIidxPreferences(iidx).mapToDouble(u ->
            {
                retrievedData.getUidxPreferences(u.v1).forEach(j -> centroid.addTo(j.v1, u.v2*j.v2));
                return 1.0;
            }).count();

            this.itemVectors.put(iidx, centroid);
            this.item2userSimilarity.put(iidx, new Int2DoubleOpenHashMap());

            if(count == 0) this.itemScores.put(iidx, 0.0);
            else
            {
                double modI = centroid.values().stream().mapToDouble(aDouble -> aDouble * aDouble).sum();
                this.itemNorm.put(iidx, modI);

                double score = retrievedData.getIidxPreferences(iidx).mapToDouble(u ->
                {
                    double dotPr = retrievedData.getUidxPreferences(u.v1).mapToDouble(j -> j.v2*centroid.get(j.v1)).sum();
                    double modU = this.userNorm.get(u.v1);
                    this.item2userSimilarity.get(iidx).put(u.v1, dotPr);
                    return (dotPr/Math.sqrt(modU));
                }).sum();
                this.itemScores.put(iidx, score);
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
                double itemNorm = Math.sqrt(this.itemNorm.getOrDefault(item, 0.0));
                int size = retrievedData.numUsers(item);
                double value = this.itemScores.getOrDefault(item, 0.0);
                if(size >= 1)
                {
                    value = 1.0 - value/(itemNorm*size);
                }

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
                double itemNorm = Math.sqrt(this.itemNorm.getOrDefault(iidx, 0.0));
                int size = retrievedData.numUsers(iidx);
                double value = this.itemScores.getOrDefault(iidx, 0.0);
                if(size >= 1)
                {
                    value = 1.0 - value/(itemNorm*size);
                }

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
        // First: if the rating is not relevant, ignore.
        if(!relevanceChecker.test(value)) return;

        // We obtain the users who have rated iidx (excepting uidx)
        IntSet set = new IntOpenHashSet();
        this.retrievedData.getIidxPreferences(iidx).map(v -> v.v1).forEach(set::add);
        set.remove(uidx);

        if(!itemVectors.containsKey(iidx))
        {
            itemVectors.put(iidx, new Int2DoubleOpenHashMap());
            item2userSimilarity.put(iidx, new Int2DoubleOpenHashMap());
        }

        // Step 1: obtain the data to update the user vector and norm:
        double rui = this.retrievedData.getPreference(uidx, iidx).orElse(new IdxPref(iidx, 0.0)).v2;
        double userNorm = this.userNorm.getOrDefault(uidx, 0.0);
        double origSimIU = this.item2userSimilarity.get(iidx).getOrDefault(uidx, 0.0);
        double squareValue = value*value;
        double incrUNorm = 2*rui*value + squareValue;

        // Now, we run over the different items that user uidx has rated in the past:
        this.retrievedData.getUidxPreferences(uidx).forEach(j ->
        {
            int jidx = j.v1;
            double ruj = j.v2;

            double ijIncr = ruj*value;

            // Step 2: we run over the users who have rated both i and j (except for uidx).
            this.retrievedData.getIidxPreferences(jidx).filter(pref -> set.contains(pref.v1)).forEach(v ->
            {
                int vidx = v.v1;
                double rvj = j.v2;
                double rvi = this.retrievedData.getPreference(vidx, iidx).orElse(new IdxPref(iidx, 0.0)).v2;

                // First, we do have to update the item vectors for i and j
                // the new i_j = i_j + r_u(j) * value
                // the new j_i = j_i + r_u(j) * value
                // We update the similarity between v and i, and the similarity between v and j:
                double simIV = item2userSimilarity.get(iidx).getOrDefault(vidx, 0.0) + rvj*ijIncr;
                double simJV = item2userSimilarity.get(jidx).getOrDefault(vidx, 0.0) + rvi*ijIncr;

                item2userSimilarity.get(iidx).put(vidx, simIV);
                item2userSimilarity.get(jidx).put(vidx, simJV);

                // Update the scores for both i and j items
                double vMod = this.userNorm.get(vidx);
                double jScore = this.itemScores.get(jidx);
                jScore += rvi*ijIncr/vMod;
                this.itemScores.put(jidx, jScore);

                double iScore = this.itemScores.get(iidx);
                iScore += rvj*ijIncr/vMod;
                this.itemScores.put(iidx, iScore);
            });

            // Next step: update the scores for the j item:

            // Step 3: Update the similarity between vectors u and j:
            double ji = this.itemVectors.getOrDefault(jidx, new Int2DoubleOpenHashMap()).getOrDefault(iidx, 0.0);
            double simJU = item2userSimilarity.get(jidx).getOrDefault(uidx, 0.0);

            double jScore = this.itemScores.get(jidx);
            if(userNorm != 0.0)
            {
                jScore -= simJU/Math.sqrt(userNorm);
            }

            simJU += rui*ijIncr + ruj*squareValue + ji*value;
            jScore += simJU/Math.sqrt(userNorm + incrUNorm);
            item2userSimilarity.get(jidx).put(uidx, simJU);
            this.itemScores.put(jidx, jScore);

            // Step 4: Update the similarity between vectors u and i:
            double simIU = item2userSimilarity.get(iidx).getOrDefault(uidx, 0.0) + ruj*ijIncr;
            item2userSimilarity.get(iidx).put(uidx, simIU);

            // Step 5: update the module and vector for item j:
            double modJ = this.itemNorm.getOrDefault(jidx, 0.0);
            modJ += 2*ji*ijIncr + ijIncr*ijIncr;
            ji += ruj*value;
            this.itemVectors.get(jidx).put(iidx, ji);
            this.itemNorm.put(jidx, modJ);

            // Step 6: update the module and vector for item i:
            double modI = this.itemNorm.getOrDefault(iidx, 0.0);
            double ij = this.itemVectors.get(iidx).getOrDefault(jidx, 0.0);
            modI += 2*ji*ijIncr + ijIncr*ijIncr;
            ij += ijIncr;
            this.itemVectors.get(iidx).put(jidx, ij);
            this.itemNorm.put(iidx, modI);
        });

        // Step 7: update the similarity between u and i
        double ii = this.itemVectors.get(iidx).getOrDefault(iidx, 0.0);
        double iiIncr = 2*rui*value + squareValue;

        double simIU = this.item2userSimilarity.get(iidx).getOrDefault(uidx, 0.0) + ii*value + 2*rui*rui*value + 3*rui*squareValue + value*squareValue;
        this.item2userSimilarity.get(iidx).put(uidx, simIU);

        // Step 8: update the norm for item i:
        double modI = this.itemNorm.getOrDefault(iidx, 0.0);
        modI += 2*ii*iiIncr + iiIncr*iiIncr;
        this.itemNorm.put(iidx, modI);

        ii += iiIncr;
        this.itemVectors.get(iidx).put(iidx, ii);

        // Step 9: finish updating the similarities between i and the rest of items that scored him:
        set.forEach(vidx ->
        {
            double iScore = this.itemScores.getOrDefault(iidx, 0.0);
            double vNorm = this.userNorm.get(vidx);


            double rvi = this.retrievedData.getPreference(vidx, iidx).orElse(new IdxPref(iidx, 0.0)).v2;
            double simIV = this.item2userSimilarity.get(iidx).getOrDefault(vidx, 0.0);
            simIV += rvi*iiIncr;
            iScore += rvi*iiIncr/vNorm;

            this.item2userSimilarity.get(iidx).put(vidx.intValue(), simIV);
            this.itemScores.put(iidx, iScore);
        });

        double iScore = this.itemScores.getOrDefault(iidx, 0.0);
        if(userNorm != 0.0)
        {
            iScore -= origSimIU/Math.sqrt(userNorm);
        }
        iScore += this.item2userSimilarity.get(iidx).get(uidx)/Math.sqrt(userNorm+incrUNorm);
        this.itemScores.put(iidx, iScore);

        // Step 10: finally, update the user vector:
        this.userNorm.put(uidx, userNorm + incrUNorm);
        retrievedData.updateRating(uidx, iidx, value);

        // And the norm of the user.
        this.userNorm.put(uidx, userNorm + incrUNorm);
        retrievedData.updateRating(uidx, iidx, value);
    }
}