package es.uam.eps.ir.knnbandit.recommendation.wisdom;

import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AbstractSimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AdditiveRatingFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.Optional;
import java.util.stream.Stream;

public class InformationTheoryUserDiversity<U,I> extends InteractiveRecommender<U, I>
{
    protected final Int2DoubleOpenHashMap num;
    protected final Int2DoubleOpenHashMap den;
    /**
     * Preference data.
     */
    protected final AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData;

    protected final double threshold;

    public InformationTheoryUserDiversity(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, double threshold)
    {
        super(uIndex, iIndex, ignoreNotRated);
        retrievedData = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.num = new Int2DoubleOpenHashMap();
        this.den = new Int2DoubleOpenHashMap();
        this.threshold = threshold;
    }

    public InformationTheoryUserDiversity(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, double threshold)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed);
        retrievedData = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        this.num = new Int2DoubleOpenHashMap();
        this.den = new Int2DoubleOpenHashMap();
        this.threshold = threshold;

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
        values.forEach(triplet -> this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), triplet.value() >= threshold ? 1.0 : 0.0));
        this.num.clear();
        this.den.clear();
        values.forEach(triplet ->
        {
           this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), triplet.value());
           if(triplet.value() >= threshold) num.addTo(triplet.uidx(), 1.0);
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
            double value = this.retrievedData.getIidxPreferences(iidx).filter(u -> u.v1 > 0).mapToDouble(u -> Math.log(den.get(u.v1)) - Math.log(num.get(u.v1))).sum();
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
    public void update(int uidx, int iidx, double value)
    {
        boolean hasRating = false;
        double oldValue = 0;
        // First, we find whether we have a rating or not:
        if(this.retrievedData.numItems(uidx) > 0 && this.retrievedData.numUsers(iidx) > 0)
        {
            Optional<IdxPref> opt = this.retrievedData.getPreference(uidx, iidx);
            hasRating = opt.isPresent();
            if(hasRating)
            {
                oldValue = opt.get().v2;
            }
        }

        if(!hasRating)
        {
            this.retrievedData.updateRating(uidx, iidx, value);
            this.retrievedData.getIidxPreferences(iidx).forEach(vidx -> this.sim.update(uidx, vidx.v1, iidx, value, vidx.v2));
        }
        else
        {
            if(this.retrievedData.updateRating(uidx, iidx, value))
            {
                Optional<IdxPref> opt = this.retrievedData.getPreference(uidx, iidx);
                if(opt.isPresent())
                {
                    double newValue = opt.get().v2;
                    this.sim.updateNormDel(uidx, oldValue);
                    this.sim.updateNorm(uidx, newValue);

                    double finalOldValue = oldValue;
                    this.retrievedData.getIidxPreferences(iidx).filter(vidx -> vidx.v1 != uidx).forEach(vidx ->
                                                                                                        {
                                                                                                            this.sim.updateDel(uidx, vidx.v1, iidx, finalOldValue, vidx.v2);
                                                                                                            this.sim.update(uidx, vidx.v1, iidx, newValue, vidx.v2);
                                                                                                        });
                }
            }
        }
    }
}
