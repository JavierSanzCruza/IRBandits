package es.uam.eps.ir.knnbandit.recommendation.wisdom;

import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AbstractSimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.*;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.function.DoublePredicate;
import java.util.stream.Stream;

public class AverageCosineUserDistance<U,I> extends AbstractInteractiveRecommender<U,I>
{
    private final VectorCosineSimilarity cosineSimilarity;
    private final Int2DoubleMap itemScores;
    private final Int2ObjectMap<IntSet> itemUsers;
    private final DoublePredicate relevanceChecker;
    /**
     * Preference data.
     */
    protected final SimpleFastUpdateablePreferenceData<U,I> retrievedData;

    public AverageCosineUserDistance(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, DoublePredicate relevanceChecker)
    {
        super(uIndex, iIndex, true);
        this.cosineSimilarity = new VectorCosineSimilarity(uIndex.numUsers());
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.relevanceChecker = relevanceChecker;
        this.itemScores = new Int2DoubleOpenHashMap();
        this.itemScores.defaultReturnValue(0.0);
        this.itemUsers = new Int2ObjectOpenHashMap<>();
        this.itemUsers.defaultReturnValue(new IntOpenHashSet());
    }

    public AverageCosineUserDistance(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, int rngSeed, DoublePredicate relevanceChecker)
    {
        super(uIndex, iIndex, true, rngSeed);
        this.cosineSimilarity = new VectorCosineSimilarity(uIndex.numUsers());
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.relevanceChecker = relevanceChecker;
        this.itemScores = new Int2DoubleOpenHashMap();
        this.itemScores.defaultReturnValue(0.0);
        this.itemUsers = new Int2ObjectOpenHashMap<>();
        this.itemUsers.defaultReturnValue(new IntOpenHashSet());
    }

    @Override
    public void init()
    {
        this.retrievedData.clear();
        this.cosineSimilarity.initialize();
        this.itemUsers.clear();
        this.itemScores.clear();
        this.getIidx().forEach(iidx -> this.itemUsers.put(iidx, new IntOpenHashSet()));
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.retrievedData.clear();
        this.itemUsers.clear();
        this.itemScores.clear();
        this.getIidx().forEach(iidx -> this.itemUsers.put(iidx, new IntOpenHashSet()));

        values.filter(triplet -> relevanceChecker.test(triplet.value())).forEach(triplet ->
        {
            this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), triplet.value());
            this.itemUsers.get(triplet.iidx()).add(triplet.uidx());
        });
        this.cosineSimilarity.initialize(retrievedData);

        // Now, we initialize the values for the items.
        this.itemUsers.forEach((iidx, uidxs) ->
        {
            double iidxVal = 0.0;
            for(int uidx1 : uidxs)
            {
                for(int uidx2 : uidxs)
                {
                    if(uidx1 >= uidx2)
                    {
                        continue;
                    }

                    double val = this.cosineSimilarity.similarity(uidx1, uidx2);
                    if(!Double.isNaN(val)) iidxVal += 1.0-val;
                }
            }

            iidxVal *= 2.0;
            this.itemScores.put(iidx.intValue(), iidxVal);
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
                int size = itemUsers.getOrDefault(item, itemUsers.defaultReturnValue()).size();
                if(size > 1);
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
                int size = itemUsers.getOrDefault(iidx, itemUsers.defaultReturnValue()).size();
                if (size > 1) ;
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

        // Then:
        // TODO: FINISH:
        // if the item receives a new user, update and add the differences.
        // then, for all users which have received the item, update.
        // NOTE: We shall need an inverse map (or just use the preference data).





    }
}