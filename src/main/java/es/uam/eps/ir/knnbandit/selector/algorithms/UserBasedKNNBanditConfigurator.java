package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.stochastic.BetaStochasticSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.InteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

public class UserBasedKNNBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String ALPHA = "alpha";
    private static final String BETA = "beta";
    private static final String K = "k";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        int k = object.getInt(K);
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);
        return new UserBasedKNNBanditInteractiveRecommenderSupplier(k, alpha, beta);
    }

    private class UserBasedKNNBanditInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        private final double alpha;
        private final double beta;
        private final int k;

        public UserBasedKNNBanditInteractiveRecommenderSupplier(int k,  double alpha, double beta)
        {
            this.alpha = alpha;
            this.beta = beta;
            this.k = k;
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            UpdateableSimilarity sim = new BetaStochasticSimilarity(userIndex.numUsers(), alpha, beta);
            return new InteractiveUserBasedKNN<>(userIndex, itemIndex, true, true, k, sim);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.UBBANDIT + "-" + k + "-" + alpha + "-" + beta;
        }
    }
}
