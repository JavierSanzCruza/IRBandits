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

/**
 * Class for configuring an matrix factorization algorithm.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.mf.InteractiveMF
 * @see es.uam.eps.ir.knnbandit.recommendation.mf.LastRatingInteractiveMF
 * @see es.uam.eps.ir.knnbandit.recommendation.mf.BestRatingInteractiveMF
 * @see es.uam.eps.ir.knnbandit.recommendation.mf.AdditiveRatingInteractiveMF
 */
public class MatrixFactorizationConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for selecting whether the algorithm is updated with items unknown by the system or not.
     */
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    /**
     * Identifier for selecting the factorization approach.
     */
    private static final String FACTORIZER = "factorizer";
    /**
     * Identifier for selecting the number of latent factors.
     */
    private static final String K = "k";
    /**
     * Identifier for selecting the matrix factorization variant (i.e. how to update the ratings).
     */
    private static final String VARIANT = "variant";

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
            FactorizerSelector<U,I> selector = new FactorizerSelector<>();
            List<FactorizerSupplier<U,I>> factorizerSuppliers = selector.getFactorizers(bandit);
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
        FactorizerSelector<U,I> selector = new FactorizerSelector<>();
        FactorizerSupplier<U,I>supplier = selector.getFactorizer(bandit);
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
