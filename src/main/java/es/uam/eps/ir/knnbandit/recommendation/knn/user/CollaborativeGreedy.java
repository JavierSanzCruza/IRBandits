package es.uam.eps.ir.knnbandit.recommendation.knn.user;

import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.FastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.userknowledge.fast.SimpleFastUserKnowledgePreferenceData;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.RestrictedVectorCosineSimilarity;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.*;
import org.jooq.lambda.tuple.Tuple3;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Implementation of the kNN-based collaborative-greedy algorithm.
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * Bresler, G., Chen, George H., Shah, D.: A latent source model for online collaborative filtering. NIPS 2014.
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
    FastUpdateablePreferenceData<U, I> jointData;
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

    public CollaborativeGreedy(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreNotRated, double threshold, double alpha)
    {
        super(uIndex, iIndex, prefData, ignoreNotRated);
        this.jointList = new IntArrayList();
        jointExpl = new Int2ObjectOpenHashMap<>();
        this.jointData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.threshold = threshold;
        this.alpha = alpha;
        this.sim = new RestrictedVectorCosineSimilarity(numUsers());
    }

    public CollaborativeGreedy(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData, boolean ignoreNotRated, KnowledgeDataUse dataUse, double threshold, double alpha)
    {
        super(uIndex, iIndex, prefData, knowledgeData, ignoreNotRated, dataUse);
        this.jointList = new IntArrayList();
        jointExpl = new Int2ObjectOpenHashMap<>();
        this.jointData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.threshold = threshold;
        this.alpha = alpha;
        this.sim = new RestrictedVectorCosineSimilarity(numUsers());

    }

    public CollaborativeGreedy(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreNotRated, boolean notReciprocal, double threshold, double alpha)
    {
        super(uIndex, iIndex, prefData, ignoreNotRated, notReciprocal);
        this.jointList = new IntArrayList();
        jointExpl = new Int2ObjectOpenHashMap<>();
        this.jointData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.threshold = threshold;
        this.alpha = alpha;
        this.sim = new RestrictedVectorCosineSimilarity(numUsers());

    }

    @Override
    protected void initializeMethod()
    {
        jointList.clear();
        this.getIidx().forEach(jointList::add);
        Collections.shuffle(jointList);

        jointExpl.clear();
        this.times.clear();
        this.getUidx().forEach(uidx ->
        {
            this.jointExpl.put(uidx, new IntOpenHashSet());
            this.times.put(uidx, trainData.numItems(uidx)+1);
            this.jointIndex.put(uidx, 0);
        });

        this.jointData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.sim.initialize(trainData);

        trainData.getAllUidx().forEach(uidx ->
           trainData.getUidxPreferences(uidx).forEach(iidx ->
              this.jointData.update(trainData.uidx2user(uidx), trainData.iidx2item(iidx.v1),iidx.v2 == 0 ? -1 : iidx.v2)));
    }

    @Override
    public int next(int uidx)
    {
        IntList list = this.availability.get(uidx);
        if (list == null || list.isEmpty())
        {
            return -1;
        }

        double probExpl = 1.0/Math.pow(numUsers(), alpha);
        double probJointExpl = 1.0/Math.pow(this.times.get(uidx), alpha);

        double next = partrng.nextDouble();
        if(next < probExpl) // Return at random
        {
            return list.getInt(rng.nextInt(list.size()));
        }
        else if(next < probExpl + probJointExpl) // return the next element by joint exploration.
        {
            int index = this.jointIndex.get(uidx);
            int iidx = this.jointList.getInt(index);

            while(!this.availability.get(uidx).contains(iidx))
            {
                ++index;
                this.jointIndex.put(uidx, index);
                iidx = this.jointList.getInt(index);
            }

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
                this.trainData.getUidxPreferences(vidx).forEach(iidx ->
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

        for(int iidx : list)
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
            return list.get(rng.nextInt(list.size()));
        }
        else if (topSize == 1)
        {
            return top.get(0);
        }
        return top.get(rng.nextInt(topSize));
    }

    @Override
    public void updateMethod(int uidx, int iidx, double value)
    {
        // Update the number of times that the user u has been recommended.
        this.times.put(uidx, this.times.get(uidx)+1);

        double auxvalue = value > 0 ? 1 : -1;

        // If the item has been explored through joint exploration, update the similarities.
        if(this.jointExpl.get(uidx).contains(iidx))
        {
            this.jointData.update(this.uIndex.uidx2user(uidx), this.iIndex.iidx2item(iidx), auxvalue);
            this.jointData.getIidxPreferences(iidx).forEach(vidx -> this.sim.update(uidx, vidx.v1, iidx, value, vidx.v2));
        }

        // Update the index for the joint exploration list.
        int index = this.jointIndex.get(uidx);
        if(index < this.prefData.numItemsWithPreferences() && iidx == this.jointList.get(index))
        {
            ++index;
            while(index < this.jointList.size() && !this.availability.get(uidx).contains(this.jointList.get(index)))
            {
                ++index;
            }
            this.jointIndex.put(uidx, index);
        }

    }



    @Override
    public void updateMethod(List<Tuple3<Integer, Integer, Double>> train)
    {


        this.sim.initialize(this.trainData);

    }
}
