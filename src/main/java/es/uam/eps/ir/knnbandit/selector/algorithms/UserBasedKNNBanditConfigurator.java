package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.stochastic.BetaStochasticSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.AdditiveRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.BestRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.InteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.LastRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.KNNBanditIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

public class UserBasedKNNBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String ALPHA = "alpha";
    private static final String BETA = "beta";
    private static final String K = "k";
    private static final String VARIANT = "variant";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        int k = object.getInt(K);
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);

        String variant = KNNBanditIdentifiers.BASIC;
        if(object.has(VARIANT))
        {
            variant = object.getString(VARIANT);
        }

        return new UserBasedKNNBanditInteractiveRecommenderSupplier(k, alpha, beta, variant);
    }

    private class UserBasedKNNBanditInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        private final double alpha;
        private final double beta;
        private final int k;
        private final String variant;

        public UserBasedKNNBanditInteractiveRecommenderSupplier(int k,  double alpha, double beta, String variant)
        {
            this.alpha = alpha;
            this.beta = beta;
            this.k = k;
            this.variant = variant;
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            UpdateableSimilarity sim = new BetaStochasticSimilarity(userIndex.numUsers(), alpha, beta);

            switch(this.variant)
            {
                case KNNBanditIdentifiers.BASIC:
                    return new InteractiveUserBasedKNN<>(userIndex, itemIndex, true, true, k, sim);
                case KNNBanditIdentifiers.BEST:
                    return new BestRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, true, k, sim);
                case KNNBanditIdentifiers.LAST:
                    return new LastRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, true, k, sim);
                case KNNBanditIdentifiers.ADDITIVE:
                    return new AdditiveRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, true, k, sim);
                default:
                    return null;
            }
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            UpdateableSimilarity sim = new BetaStochasticSimilarity(userIndex.numUsers(), alpha, beta);

            switch(this.variant)
            {
                case KNNBanditIdentifiers.BASIC:
                    return new InteractiveUserBasedKNN<>(userIndex, itemIndex, true, rngSeed, true, k, sim);
                case KNNBanditIdentifiers.BEST:
                    return new BestRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, rngSeed,true, k, sim);
                case KNNBanditIdentifiers.LAST:
                    return new LastRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, rngSeed,true, k, sim);
                case KNNBanditIdentifiers.ADDITIVE:
                    return new AdditiveRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, rngSeed, true, k, sim);
                default:
                    return null;
            }
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.UBBANDIT + "-" + variant + "-" + k + "-" + alpha + "-" + beta;
        }
    }
}
