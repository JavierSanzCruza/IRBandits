package es.uam.eps.ir.knnbandit.recommendation.reranker;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.FastInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.stream.Stream;

/**
 * Given two interactive recommenders, this algorithm uses the first to obtain a top-k recommendation,
 * which is later reranked using the second algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 */
public class RankingCombiner<U,I> extends AbstractInteractiveRecommender<U, I>
{
    /**
     * The base recommender.
     */
    private final FastInteractiveRecommender<U,I> baseRec;
    /**
     * The reranking recommender.
     */
    private final FastInteractiveRecommender<U,I> rerRec;
    /**
     * The top-k of the recommendation.
     */
    private final int topK;
    /**
     * Indicates the trade-off between the original recommender (when lambda == 0) and the
     * remaining one (when lambda == 1).
     */
    private final double lambda;


    /**
     * Constructor.
     * @param uIndex            user index
     * @param iIndex            item index
     * @param ignoreNotRated    true if we ignore not rated items, true otherwise.
     * @param baseRec           base recommender.
     * @param rerRec            reranking recommender.
     * @param topK              the number of elements to retrieve from the base recommender.
     * @param lambda            trade-off betweeen the original recommendation and the reranking one.
     */
    public RankingCombiner(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, FastInteractiveRecommender<U,I> baseRec, FastInteractiveRecommender<U,I> rerRec, int topK, double lambda)
    {
        super(uIndex, iIndex, ignoreNotRated);
        this.baseRec = baseRec;
        this.rerRec = rerRec;
        this.topK = topK;
        this.lambda = lambda;
    }

    /**
     * Constructor.
     * @param uIndex            user index
     * @param iIndex            item index
     * @param ignoreNotRated    true if we ignore not rated items, true otherwise.
     * @param rngSeed           random number generator seed.
     * @param baseRec           base recommender.
     * @param rerRec            reranking recommender.
     * @param topK              the number of elements to retrieve from the base recommender.
     * @param lambda            trade-off betweeen the original recommendation and the reranking one.
     */
    public RankingCombiner(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, FastInteractiveRecommender<U,I> baseRec, FastInteractiveRecommender<U,I> rerRec, int topK, double lambda)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed);
        this.baseRec = baseRec;
        this.rerRec = rerRec;
        this.topK = topK;
        this.lambda = lambda;
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        List<FastRating> aux = new ArrayList<>();
        values.forEach(aux::add);
        this.baseRec.init(aux.stream());
        this.rerRec.init(aux.stream());
    }

    @Override
    public int next(int uidx, IntList available)
    {
        // First, we obtain the top k recommendation:
        IntList fastBaseRec = baseRec.next(uidx, available, topK);
        // and store the corresponding values
        Int2DoubleMap map = new Int2DoubleOpenHashMap();
        int size = fastBaseRec.size();
        for(int i = 0; i < size; ++i)
        {
            map.put(fastBaseRec.get(i).intValue(), (size - i + 0.0)/(size+0.0));
        }

        // Then, we obtain the recommendation ranking for those elements, given the second recommender
        IntList fastRerRec = rerRec.next(uidx, fastBaseRec, topK);

        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();

        int i = 0;
        size = fastRerRec.size();

        // ranking combination using the rank-sim normalization.
        for(int iidx : fastRerRec)
        {
            double value = (1.0-lambda)*map.get(fastRerRec) + lambda*(size - i + 0.0)/(size + 0.0);
            if(value > max)
            {
                top.clear();
                max = value;
                top.add(iidx);
            }
            else if(value == max)
            {
                top.add(iidx);
            }

            ++i;
        }

        int topSize = top.size();
        if(top.isEmpty())
        {
            return available.get(rng.nextInt(available.size()));
        }
        if(topSize == 1)
            return top.get(0);
        else
            return top.get(rng.nextInt(topSize));
    }

    @Override
    public IntList next(int uidx, IntList available, int k)
    {
        // First, we obtain the top k recommendation:
        IntList fastBaseRec = baseRec.next(uidx, available, topK);
        Int2DoubleMap map = new Int2DoubleOpenHashMap();
        int size = fastBaseRec.size();
        for(int i = 0; i < size; ++i)
        {
            map.put(fastBaseRec.get(i).intValue(), (size - i + 0.0)/(size+0.0));
        }

        IntList fastRerRec = rerRec.next(uidx, fastBaseRec, topK);
        int i = 0;
        size = fastRerRec.size();

        IntList top = new IntArrayList();
        int num = Math.min(k, available.size());
        PriorityQueue<Tuple2id> queue = new PriorityQueue<>(num, Comparator.comparingDouble(x -> x.v2));
        for(int iidx : fastRerRec)
        {
            double val = (1.0-lambda)*map.get(fastRerRec) + lambda*(size - i + 0.0)/(size + 0.0);
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

            ++i;
        }

        while(!queue.isEmpty())
        {
            top.add(0, queue.poll().v1);
        }

        while(top.size() < num)
        {
            int idx = rng.nextInt(available.size());
            int item = available.get(idx);
            if(!top.contains(item)) top.add(item);
        }

        return top;
    }

    @Override
    public void fastUpdate(int uidx, int iidx, double value)
    {
        this.baseRec.fastUpdate(uidx, iidx, value);
        this.rerRec.fastUpdate(uidx, iidx, value);
    }
}
