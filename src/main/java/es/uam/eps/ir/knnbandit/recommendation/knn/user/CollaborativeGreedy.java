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
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.RestrictedVectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.*;

import java.util.Collections;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Implementation of the kNN-based collaborative-greedy algorithm.
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * <b>Reference:</b> G. Bresler, G.H. Chen, D. Shah. A latent source model for online collaborative filtering. 28th Conference on Neural Information Processing Systems (NeurIPS 2014). Montréal, Canada (2014).
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 */
public class CollaborativeGreedy<U,I> extends InteractiveRecommender<U, I>
{
    /**
     * A list containing a random order of items.
     */
    private final IntList jointList;
    /**
     * For each users, we indicate which items have been jointly explored.
     */
    private final Int2ObjectMap<IntSet> jointExpl;

    /**
     * Relation between users and jointly explored objects (including ratings)
     */
    SimpleFastUpdateablePreferenceData<U, I> jointData;

    SimpleFastUpdateablePreferenceData<U,I> retrievedData;
    /**
     * Number of times each user has been recommended an item.
     */
    private final Int2IntMap times = new Int2IntOpenHashMap();
    /**
     * The position of the jointList for each user.
     */
    private final Int2IntMap jointIndex = new Int2IntOpenHashMap();

    /**
     * Similarity threshold to appear in a neighborhood
     */
    private final double threshold;
    /**
     * Parameter for the time decay of the probability of joint exploration. Between 0 and 4/7
     */
    private final double alpha;

    /**
     * Random number to select the epsilon value.
     */
    private final Random partrng = new Random();
    /**
     * Similarity.
     */
    private RestrictedVectorCosineSimilarity sim;

    /**
     * Constructor.
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param threshold Similarity threshold. Two users are similar if their similarity is smaller than this value.
     * @param alpha Value for determining the exploration probability.
     */
    public CollaborativeGreedy(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, double threshold, double alpha)
    {
        super(uIndex, iIndex, ignoreNotRated);

        this.jointList = new IntArrayList();
        jointExpl = new Int2ObjectOpenHashMap<>();
        this.jointData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.threshold = threshold;
        this.alpha = alpha;
        this.sim = new RestrictedVectorCosineSimilarity(numUsers());
    }

    /**
     * Constructor.
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param threshold Similarity threshold. Two users are similar if their similarity is smaller than this value.
     * @param alpha Value for determining the exploration probability.
     */
    public CollaborativeGreedy(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, double threshold, double alpha)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed);

        this.jointList = new IntArrayList();
        jointExpl = new Int2ObjectOpenHashMap<>();
        this.jointData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.threshold = threshold;
        this.alpha = alpha;
        this.sim = new RestrictedVectorCosineSimilarity(numUsers());
    }

    @Override
    public void init()
    {
        super.init();

        // Sort the list containing random orders of the items.
        jointList.clear();
        this.getIidx().forEach(jointList::add);
        Collections.shuffle(jointList);

        jointExpl.clear();
        this.times.clear();
        this.jointIndex.clear();

        this.jointData.clear();
        this.retrievedData.clear();

        this.sim.initialize();
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();
        values.forEach(t -> jointData.updateRating(t.uidx(), t.iidx(), t.value() == 0 ? -1.0 : 1.0));
        this.sim.initialize(jointData);
    }

    /*@Override
    public void init(FastPreferenceData<U, I> prefData)
    {
        jointList.clear();
        this.getIidx().forEach(jointList::add);
        Collections.shuffle(jointList);

        jointExpl.clear();
        this.times.clear();
        jointIndex.clear();

        this.jointData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);

        prefData.getAllUidx().forEach(uidx ->
            prefData.getUidxPreferences(uidx).forEach(iidx ->
                this.jointData.update(prefData.uidx2user(uidx), prefData.iidx2item(iidx.v1),iidx.v2 == 0 ? -1 : iidx.v2)));

        this.sim.initialize(jointData);
    }*/

    @Override
    public int next(int uidx, IntList availability)
    {
        // We check whether there is an available item to recommend
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }

        // Find the probabilities of exploring / joint exploring
        double probExpl = 1.0/Math.pow(numUsers(), alpha);
        double probJointExpl = 1.0/Math.pow(this.times.get(uidx), alpha);


        double next = partrng.nextDouble();
        if(next < probExpl) // Return at random
        {
            return availability.getInt(rng.nextInt(availability.size()));
        }
        else if(next < probExpl + probJointExpl) // return the next element by joint exploration.
        {
            int index = this.jointIndex.getOrDefault(uidx, 0);
            int iidx = this.jointList.getInt(index);

            while(!availability.contains(iidx))
            {
                ++index;
                this.jointIndex.put(uidx, index);
                iidx = this.jointList.getInt(index);
            }

            if(!this.jointExpl.containsKey(uidx)) this.jointExpl.put(uidx, new IntOpenHashSet());
            this.jointExpl.get(uidx).add(iidx);
            return iidx;
        }

        // Otherwise: exploit
        Int2DoubleOpenHashMap itemScores = new Int2DoubleOpenHashMap();
        Int2DoubleOpenHashMap itemDen = new Int2DoubleOpenHashMap();
        itemScores.defaultReturnValue(1.0);
        itemDen.defaultReturnValue(2.0);

        // Compute the recommendation scores.
        this.sim.similarElems(uidx).forEach(sim ->
        {
            if(sim.v2 > threshold)
            {
                int vidx = sim.v1;
                this.retrievedData.getUidxPreferences(vidx).forEach(iidx ->
                {
                    if(itemScores.containsKey(iidx.v1))
                    {
                        if(iidx.v2 > 0) itemScores.addTo(iidx.v1, 1.0);
                        itemDen.addTo(iidx.v1, 1.0);
                    }
                    else
                    {
                        if(iidx.v2 > 0) itemScores.put(iidx.v1, 1.0);
                        itemDen.put(iidx.v1, 1.0);
                    }
                });
            }
        });

        // Select the best item:
        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();

        for(int iidx : availability)
        {
            double val = itemScores.getOrDefault(iidx, itemScores.defaultReturnValue());
            double count = itemDen.getOrDefault(iidx, itemDen.defaultReturnValue());

            val = val/count;

            if(top.isEmpty() || val > max)
            {
                top = new IntArrayList();
                max = val;
                top.add(iidx);
            }
            else if(val == max)
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
    public void update(int uidx, int iidx, double value)
    {
        double newValue;
        if(!Double.isNaN(value))
            newValue = value;
        else if(!this.ignoreNotRated)
            newValue = Constants.NOTRATEDNOTIGNORED;
        else
            return;


        if(this.retrievedData.updateRating(uidx, iidx, newValue))
        {
            // Update the number of times that the user u has been recommended.
            this.times.put(uidx, this.times.getOrDefault(uidx, 1) + 1);

            double auxvalue = value > 0 ? 1 : -1;

            // If the item has been explored through joint exploration, update the similarities.
            if (this.jointExpl.get(uidx).contains(iidx))
            {
                this.jointData.update(this.uIndex.uidx2user(uidx), this.iIndex.iidx2item(iidx), auxvalue);
                this.jointData.getIidxPreferences(iidx).forEach(vidx -> this.sim.update(uidx, vidx.v1, iidx, auxvalue, vidx.v2));
            }

            // Update the index for the joint exploration list.
            int index = this.jointIndex.get(uidx);
            if (index < this.retrievedData.numItemsWithPreferences() && iidx == this.jointList.get(index))
            {
                ++index;
                while (index < this.jointList.size() && this.retrievedData.getPreference(uidx, this.jointList.get(index)).isPresent())
                {
                    ++index;
                }
                this.jointIndex.put(uidx, index);
            }
        }
    }
}
