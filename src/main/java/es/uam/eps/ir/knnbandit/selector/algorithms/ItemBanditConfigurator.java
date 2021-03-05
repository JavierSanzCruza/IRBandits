package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.bandits.ItemBanditRecommender;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunctions;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.algorithms.bandit.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class ItemBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    private static final String BANDIT = "bandit";
    private static final String NAME = "name";
    private static final String PARAMS = "params";

    @Override
    public List<InteractiveRecommenderSupplier<U, I>> getAlgorithms(JSONArray array)
    {
        List<InteractiveRecommenderSupplier<U,I>> list = new ArrayList<>();
        int numConfigs = array.length();
        for(int i = 0; i < numConfigs; ++i)
        {
            JSONObject object = array.getJSONObject(i);

            boolean ignoreUnknown = true;
            if(object.has(IGNOREUNKNOWN))
            {
                ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
            }

            JSONObject bandit = object.getJSONObject(BANDIT);
            String name = bandit.getString(NAME);
            BanditConfigurator banditConfigurator = this.selectBanditConfigurator(name);
            if(banditConfigurator == null) return null;

            List<BanditSupplier> banditSuppliers = banditConfigurator.getBandits(bandit.getJSONArray(PARAMS));
            for(BanditSupplier supplier : banditSuppliers)
            {
                list.add(new ItemBanditInteractiveRecommenderSupplier<>(supplier, ignoreUnknown));
            }
        }
        return list;
    }

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        JSONObject bandit = object.getJSONObject(BANDIT);
        String name = bandit.getString(NAME);
        BanditConfigurator banditConfigurator = this.selectBanditConfigurator(name);
        if(banditConfigurator == null) return null;

        BanditSupplier banditSupplier = banditConfigurator.getBandit(bandit.getJSONObject(PARAMS));
        return new ItemBanditInteractiveRecommenderSupplier<>(banditSupplier, ignoreUnknown);
    }

    protected BanditConfigurator selectBanditConfigurator(String name)
    {
        switch(name)
        {
            case ItemBanditIdentifiers.EGREEDY:
                return new EpsilonGreedyConfigurator<>();
            case ItemBanditIdentifiers.ETGREEDY:
                return new EpsilonTGreedyConfigurator<>();
            case ItemBanditIdentifiers.UCB1:
                return new UCB1Configurator<>();
            case ItemBanditIdentifiers.UCB1TUNED:
                return new UCB1TunedConfigurator<>();
            case ItemBanditIdentifiers.THOMPSON:
                return new ThompsonSamplingConfigurator<>();
            case ItemBanditIdentifiers.DELAYTHOMPSON:
                return new DelayThompsonSamplingConfigurator<>();
            case ItemBanditIdentifiers.MLEPOP:
                return new MLECategoricalItemBanditConfigurator<>();
            case ItemBanditIdentifiers.MLEAVG:
                return new MLECategoricalAverageItemBanditConfigurator<>();
            default:
                return null;
        }
    }


    private static class ItemBanditInteractiveRecommenderSupplier<U,I> implements InteractiveRecommenderSupplier<U,I>
    {
        BanditSupplier banditSupplier;
        boolean ignoreUnknown;

        public ItemBanditInteractiveRecommenderSupplier(BanditSupplier supplier, boolean ignoreUnknown)
        {
            this.banditSupplier = supplier;
            this.ignoreUnknown = ignoreUnknown;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            ValueFunction valueFunction = ValueFunctions.identity();
            return new ItemBanditRecommender<>(userIndex, itemIndex, ignoreUnknown, banditSupplier, valueFunction);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return this.apply(userIndex, itemIndex);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.ITEMBANDIT + "-" + banditSupplier.getName() + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
