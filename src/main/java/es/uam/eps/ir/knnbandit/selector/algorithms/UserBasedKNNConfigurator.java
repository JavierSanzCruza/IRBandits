package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.AdditiveRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.BestRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.InteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.LastRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.KNNBanditIdentifiers;
import org.json.JSONObject;

public class UserBasedKNNConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    private static final String IGNOREZEROES = "ignoreZeroes";
    private static final String K = "k";
    private static final String VARIANT = "variant";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        boolean ignoreZeroes = true;
        if(object.has(IGNOREZEROES))
        {
            ignoreZeroes = object.getBoolean(IGNOREZEROES);
        }

        String variant = KNNBanditIdentifiers.BASIC;
        if(object.has(VARIANT))
        {
            variant = object.getString(VARIANT);
        }

        int k = object.getInt(K);

        return new UserBasedInteractiveRecommenderSupplier(k, ignoreZeroes, ignoreUnknown, variant);
    }

    private class UserBasedInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        private final boolean ignoreUnknown;
        private final boolean ignoreZeroes;
        private final int k;
        private final String variant;

        public UserBasedInteractiveRecommenderSupplier(int k, boolean ignoreZeroes, boolean ignoreUnknown, String variant)
        {
            this.ignoreUnknown = ignoreUnknown;
            this.ignoreZeroes = ignoreZeroes;
            this.k = k;
            this.variant = variant;
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            UpdateableSimilarity sim = new VectorCosineSimilarity(userIndex.numUsers());

            switch(this.variant)
            {
                case KNNBanditIdentifiers.BASIC:
                    return new InteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.BEST:
                    return new BestRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.LAST:
                    return new LastRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.ADDITIVE:
                    return new AdditiveRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, ignoreZeroes, k, sim);
                default:
                    return null;
            }
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            UpdateableSimilarity sim = new VectorCosineSimilarity(userIndex.numUsers());

            switch(this.variant)
            {
                case KNNBanditIdentifiers.BASIC:
                    return new InteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, rngSeed, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.BEST:
                    return new BestRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, rngSeed, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.LAST:
                    return new LastRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, rngSeed, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.ADDITIVE:
                    return new AdditiveRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, rngSeed, ignoreZeroes, k, sim);
                default:
                    return null;
            }
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.USERBASEDKNN + "-" + variant + "-" + k + "-" + (ignoreZeroes ? "ignore" : "all") + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
