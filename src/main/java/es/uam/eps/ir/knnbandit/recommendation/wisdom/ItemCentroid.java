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
public class ItemCentroid<U,I> extends AbstractInteractiveRecommender<U,I>
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
    public ItemCentroid(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, DoublePredicate relevanceChecker)
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
    public ItemCentroid(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, int rngSeed, DoublePredicate relevanceChecker)
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
                double value = itemScores.getOrDefault(item, itemScores.defaultReturnValue());
                int size = retrievedData.numUsers(item);
                if(size >= 1)
                    value = 1.0 - this.itemNorm.get(item)*(size + 1.0);

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
                if(size >= 1)
                    value = 1.0 - this.itemNorm.get(iidx)*(size + 1.0);

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

        double oldVal = this.retrievedData.getPreference(uidx, iidx).orElse(new IdxPref(iidx, 0.0)).v2();
        double userNorm = this.userNorm.getOrDefault(uidx, 0.0);
        double incrUNorm = 2*oldVal*value + value*value;

        // We run over the different items rated by uidx in the past.
        this.retrievedData.getUidxPreferences(uidx).forEach(j ->
        {
            int jidx = j.v1;
            double ruj = j.v2;

            // Step 1: we run over the users who have rated both i and j (except for uidx).
            this.retrievedData.getIidxPreferences(jidx).filter(pref -> set.contains(pref.v1)).forEach(v ->
            {
                int vidx = v.v1;
                double rvj = j.v2;

                // We have to update the dot product values between iidx, jidx and vidx:
                // For this, we consider that the i_j and j_i indexes are modified:
                // the new i_j = i_j + r_u(j)
                // the new j_i = j_i + value

                // so, the dot product between i and the user v is increased by r_u(j)*r_v(j):
                double dotProdI = ruj*rvj;
                // and, the dot product between j and user v is increased by r_v(i)*value:
                double dotProdJ = this.retrievedData.getPreference(vidx, iidx).orElse(new IdxPref(iidx, 0.0)).v2 * value;

                // Now that we have this increment, we can then update
                // a) the dot product between iidx,jidx and vidx:
                double simIV = item2userSimilarity.get(iidx).getOrDefault(vidx, 0.0);
                double simJV = item2userSimilarity.get(jidx).get(vidx);

                item2userSimilarity.get(iidx).put(vidx, simIV + dotProdI);
                item2userSimilarity.get(jidx).put(vidx, simJV + dotProdJ);

                // b) the item scores: in this case, as the norm of vidx is not modified, it is enough to sum up the
                // increment in the dot product divided by the norm of the user.
                double vidxNorm = this.userNorm.get(vidx);
                double scoreI = this.itemScores.getOrDefault(iidx, 0.0);
                double scoreJ = this.itemScores.get(jidx);

                itemScores.put(iidx, scoreI + dotProdI/vidxNorm);
                itemScores.put(jidx, scoreJ + dotProdJ/vidxNorm);
            });

            // Step 2: Now, we update the distance between jidx and uidx:
            double ji = this.itemVectors.get(jidx).getOrDefault(iidx, 0.0);
            double dotProdJ = value*value + ji*value + oldVal*value;
            double distJU = this.item2userSimilarity.get(jidx).get(uidx);

            // Update the similarity:
            item2userSimilarity.get(jidx).put(uidx, distJU + dotProdJ);
            itemScores.put(jidx, itemScores.get(jidx) - distJU/userNorm + (distJU + dotProdJ)/(userNorm + incrUNorm));
            itemNorm.put(jidx, itemNorm.get(jidx) + 2.0*ji*value + value*value);
            itemVectors.get(jidx).put(iidx, ji + value);

            // Step 3: Now, we update the vector and norm
            double ij = this.itemVectors.get(iidx).getOrDefault(jidx, 0.0);
            double dotProdI = ruj*ruj;
            double distIU = this.item2userSimilarity.get(iidx).getOrDefault(uidx, 0.0);
            item2userSimilarity.get(iidx).put(uidx, distIU + dotProdI);
            itemScores.put(iidx, itemScores.getOrDefault(iidx, 0.0) - distIU/userNorm + (distIU + dotProdI)/(userNorm + incrUNorm));
            itemNorm.put(iidx, itemNorm.getOrDefault(iidx, 0.0) + 2.0*ij*ruj + ruj*ruj);
            itemVectors.get(iidx).put(jidx, ij + ruj);
        });

        // Finally, we update the i_i value:
        double dist = this.item2userSimilarity.get(iidx).getOrDefault(uidx, 0.0);
        double ii = this.itemVectors.get(iidx).getOrDefault(iidx, 0.0);
        double dotProd = value*value + ii*value+ value*oldVal;
        item2userSimilarity.get(iidx).put(uidx, dist + dotProd);
        if(userNorm > 0.0)
            itemScores.put(iidx, itemScores.getOrDefault(iidx, 0.0) - dist/userNorm + (dist + dotProd)/(userNorm + incrUNorm));
        else // in this case, dist == 0 by default..
            itemScores.put(iidx, itemScores.getOrDefault(iidx, 0.0) + dotProd/incrUNorm);
        itemNorm.put(iidx, itemNorm.getOrDefault(iidx,0.0) + 2.0*ii*value + value*value);
        itemVectors.get(iidx).put(iidx, ii + value );

        // And the norm of the user.
        this.userNorm.put(uidx, userNorm + incrUNorm);
        retrievedData.updateRating(uidx, iidx, value);
    }
}