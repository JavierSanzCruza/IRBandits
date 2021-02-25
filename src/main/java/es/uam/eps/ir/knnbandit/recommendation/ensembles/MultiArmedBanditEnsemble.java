package es.uam.eps.ir.knnbandit.recommendation.ensembles;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.MultiArmedBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.selector.algorithms.bandit.BanditSupplier;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Dynamic ensemble that uses multi-armed bandit strategies to decide between recommenders.
 *
 * @param <U> type of the users.
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class MultiArmedBanditEnsemble<U,I> extends InteractiveRecommender<U,I> implements Ensemble<U,I>
{
    /**
     * The list of recommenders in the ensemble.
     */
    private final List<InteractiveRecommender<U,I>> recommenders;
    /**
     * The names of the recommenders.
     */
    private final List<String> recNames;
    /**
     * The multi-armed bandit to select between recommenders.
     */
    private final MultiArmedBandit bandit;
    /**
     * Auxiliar list.
     */
    private final IntList available;
    /**
     * The value function for the bandit
     */
    private final ValueFunction valFunc;
    /**
     * The last used recommendation algorithm.
     */
    int lastRec;

    /**
     * Constructor.
     * @param uIndex the user index.
     * @param iIndex the item index.
     * @param ignoreNotRated true if we want to ignore user-item pairs without rating, false otherwise.
     * @param mabFunc a function for obtaining a multi-armed bandit.
     * @param recs a map, indexed by recommender name, and containing recommender suppliers as values.
     * @param valFunc a value function for the bandit.
     */
    public MultiArmedBanditEnsemble(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, BanditSupplier mabFunc, Map<String, InteractiveRecommenderSupplier<U,I>> recs, ValueFunction valFunc)
    {
        super(uIndex, iIndex, ignoreNotRated);
        this.recommenders = new ArrayList<>();
        this.recNames = new ArrayList<>();
        recs.forEach((name, rec) ->
        {
            recNames.add(name);
            recommenders.add(rec.apply(uIndex, iIndex));
        });

        bandit = mabFunc.apply(recNames.size());
        int lastRec = -1;
        this.available = new IntArrayList();
        for(int i = 0; i< recNames.size(); ++i) available.add(i);
        this.valFunc = valFunc;
    }

    /**
     * Constructor.
     * @param uIndex the user index.
     * @param iIndex the item index.
     * @param ignoreNotRated true if we want to ignore user-item pairs without rating, false otherwise.
     * @param rngSeed random number generator seed.
     * @param mabFunc a function for obtaining a multi-armed bandit.
     * @param recs a map, indexed by recommender name, and containing recommender suppliers as values.
     * @param valFunc a value function for the bandit.
     */
    public MultiArmedBanditEnsemble(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, BanditSupplier mabFunc, Map<String, InteractiveRecommenderSupplier<U,I>> recs, ValueFunction valFunc)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed);
        this.recommenders = new ArrayList<>();
        this.recNames = new ArrayList<>();
        recs.forEach((name, rec) ->
        {
            recNames.add(name);
            recommenders.add(rec.apply(uIndex, iIndex));
        });

        bandit = mabFunc.apply(recNames.size());
        int lastRec = -1;
        this.available = new IntArrayList();
        for(int i = 0; i< recNames.size(); ++i) available.add(i);
        this.valFunc = valFunc;
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.recommenders.forEach(rec -> rec.init(values));
    }

    @Override
    public int next(int uidx, IntList available)
    {
        this.lastRec = bandit.next(available, valFunc);
        return this.recommenders.get(lastRec).next(uidx, available);
    }

    @Override
    public IntList next(int uidx, IntList available, int k)
    {
        this.lastRec = bandit.next(this.available, valFunc);
        return this.recommenders.get(lastRec).next(uidx, available, k);
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        bandit.update(lastRec, value);
        recommenders.forEach(rec -> rec.update(uidx, iidx, value));
    }

    @Override
    public int getCurrentAlgorithm()
    {
        return lastRec;
    }

    @Override
    public String getAlgorithmName(int idx)
    {
        return recNames.get(idx);
    }

    @Override
    public Pair<Integer> getAlgorithmStats(int idx)
    {
        return bandit.getStats(idx);
    }
}