package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.mf.AdditiveRatingInteractiveMF;
import es.uam.eps.ir.knnbandit.recommendation.mf.BestRatingInteractiveMF;
import es.uam.eps.ir.knnbandit.recommendation.mf.InteractiveMF;
import es.uam.eps.ir.knnbandit.recommendation.mf.LastRatingInteractiveMF;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.KNNBanditIdentifiers;
import es.uam.eps.ir.knnbandit.selector.algorithms.factorizer.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class MFConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    private static final String FACTORIZER = "factorizer";
    private static final String K = "k";
    private static final String VARIANT = "variant";
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

            int k = object.getInt(K);
            String variant = object.getString(VARIANT);

            JSONObject bandit = object.getJSONObject(FACTORIZER);
            String name = bandit.getString(NAME);
            FactorizerConfigurator<U,I> factorizerConfigurator = this.selectFactorizerConfigurator(name);
            if(factorizerConfigurator == null) return null;

            List<FactorizerSupplier<U,I>> factorizerSuppliers = factorizerConfigurator.getFactorizers(bandit.getJSONArray(PARAMS));
            for(FactorizerSupplier<U,I> supplier : factorizerSuppliers)
            {
                list.add(new MFInteractiveRecommenderSupplier<>(supplier, k, ignoreUnknown, variant));
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
        int k = object.getInt(K);
        String variant = object.getString(VARIANT);

        JSONObject bandit = object.getJSONObject(FACTORIZER);
        String name = bandit.getString(NAME);
        FactorizerConfigurator<U,I> factorizerConfigurator = this.selectFactorizerConfigurator(name);
        if(factorizerConfigurator == null) return null;

        FactorizerSupplier<U,I>supplier = factorizerConfigurator.getFactorizer(bandit.getJSONObject(PARAMS));
        return new MFInteractiveRecommenderSupplier<>(supplier, k,  ignoreUnknown, variant);
    }

    protected FactorizerConfigurator<U,I> selectFactorizerConfigurator(String name)
    {
        switch(name)
        {
            case FactorizerIdentifiers.IMF:
                return new HKVFactorizerConfigurator<>();
            case FactorizerIdentifiers.FASTIMF:
                return new PZTFactorizerConfigurator<>();
            case FactorizerIdentifiers.PLSA:
                return new PLSAFactorizerConfigurator<>();
            default:
                return null;
        }
    }


    private static class MFInteractiveRecommenderSupplier<U,I> implements InteractiveRecommenderSupplier<U,I>
    {
        FactorizerSupplier<U,I> supplier;
        int k;
        boolean ignoreUnknown;
        String variant;

        public MFInteractiveRecommenderSupplier(FactorizerSupplier<U,I> supplier, int k, boolean ignoreUnknown, String variant)
        {
            this.supplier = supplier;
            this.k = k;
            this.ignoreUnknown = ignoreUnknown;
            this.variant = variant;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            switch(this.variant)
            {
                case KNNBanditIdentifiers.BASIC:
                    return new InteractiveMF<>(userIndex, itemIndex, ignoreUnknown, k, supplier.apply());
                case KNNBanditIdentifiers.BEST:
                    return new BestRatingInteractiveMF<>(userIndex, itemIndex, ignoreUnknown, k, supplier.apply());
                case KNNBanditIdentifiers.LAST:
                    return new LastRatingInteractiveMF<>(userIndex, itemIndex, ignoreUnknown, k, supplier.apply());
                case KNNBanditIdentifiers.ADDITIVE:
                    return new AdditiveRatingInteractiveMF<>(userIndex, itemIndex, ignoreUnknown, k, supplier.apply());
                default:
                    return null;
            }
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            switch(this.variant)
            {
                case KNNBanditIdentifiers.BASIC:
                    return new InteractiveMF<>(userIndex, itemIndex, ignoreUnknown, rngSeed, k, supplier.apply());
                case KNNBanditIdentifiers.BEST:
                    return new BestRatingInteractiveMF<>(userIndex, itemIndex, ignoreUnknown, rngSeed, k, supplier.apply());
                case KNNBanditIdentifiers.LAST:
                    return new LastRatingInteractiveMF<>(userIndex, itemIndex, ignoreUnknown, rngSeed, k, supplier.apply());
                case KNNBanditIdentifiers.ADDITIVE:
                    return new AdditiveRatingInteractiveMF<>(userIndex, itemIndex, ignoreUnknown, rngSeed, k, supplier.apply());
                default:
                    return null;
            }

        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.MF + "-" + variant + "-" + k + "-" + supplier.getName() + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
