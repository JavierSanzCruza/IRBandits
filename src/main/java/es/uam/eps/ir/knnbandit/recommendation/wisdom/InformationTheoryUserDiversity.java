package es.uam.eps.ir.knnbandit.recommendation.wisdom;

import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AbstractSimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AdditiveRatingFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.function.DoublePredicate;
import java.util.stream.Stream;

public class InformationTheoryUserDiversity<U,I> extends InteractiveRecommender<U, I>
{
    protected final Int2DoubleOpenHashMap num;
    protected final Int2DoubleOpenHashMap den;
    /**
     * Preference data.
     */
    protected final AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData;

    protected final DoublePredicate predicate;


    public InformationTheoryUserDiversity(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, DoublePredicate predicate)
    {
        super(uIndex, iIndex, ignoreNotRated);
        retrievedData = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.num = new Int2DoubleOpenHashMap();
        this.den = new Int2DoubleOpenHashMap();
        this.predicate = predicate;
    }

    public InformationTheoryUserDiversity(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, DoublePredicate predicate)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed);
        retrievedData = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.num = new Int2DoubleOpenHashMap();
        this.den = new Int2DoubleOpenHashMap();
        this.predicate = predicate;
    }

    @Override
    public void init()
    {
        super.init();
        this.retrievedData.clear();
        this.num.clear();
        this.den.clear();
        this.uIndex.getAllUidx().forEach(uidx ->
        {
             num.put(uidx, 0.0);
             den.put(uidx, 0.0);
        });
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        super.init();
        this.retrievedData.clear();
        values.forEach(triplet -> this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), predicate.test(triplet.value()) ? 1.0 : 0.0));
        this.num.clear();
        this.den.clear();
        values.forEach(triplet ->
        {
           this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), predicate.test(triplet.value()) ? 1.0 : 0.0);
           if(predicate.test(triplet.value())) num.addTo(triplet.uidx(), 1.0);
           den.addTo(triplet.uidx(), 1.0);
        });
    }

    @Override
    public int next(int uidx, IntList available)
    {
        if(available == null || available.isEmpty()) return -1;
        if(available.size() == 1) return available.get(0);

        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();

        for(int iidx : available)
        {
            double value = this.retrievedData.getIidxPreferences(iidx).filter(u -> predicate.test(u.v2())).mapToDouble(u -> Math.log(den.get(u.v1)) - Math.log(num.get(u.v1))).sum();
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
        if(available == null || available.isEmpty()) return new IntArrayList();

        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();

        int n = Math.min(available.size(), k);
        PriorityQueue<Tuple2id> queue = new PriorityQueue<>(n, Comparator.comparingDouble(x -> x.v2));
        for(int iidx : available)
        {
            double val = this.retrievedData.getIidxPreferences(iidx).filter(u -> predicate.test(u.v2())).mapToDouble(u -> Math.log(den.get(u.v1)) - Math.log(num.get(u.v1))).sum();
            if(queue.size() < n)
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

        return top;
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
        this.retrievedData.updateRating(uidx, iidx, predicate.test(newValue) ? 1.0 : 0.0);
        this.num.addTo(uidx, predicate.test(newValue) ? 1.0 : 0.0);
        this.den.addTo(uidx, 1.0);
    }
}
