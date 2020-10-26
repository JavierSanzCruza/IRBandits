package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.knn.item.InteractiveItemBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.stochastic.BetaStochasticSimilarity;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

public class ItemBasedKNNBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String ALPHA = "alpha";
    private static final String BETA = "beta";
    private static final String USERK = "userK";
    private static final String ITEMK = "itemK";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        int userK = object.getInt(USERK);
        int itemK = object.getInt(ITEMK);
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);
        return new ItemBasedKNNBanditInteractiveRecommenderSupplier(userK, itemK, alpha, beta);
    }

    private class ItemBasedKNNBanditInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        private final double alpha;
        private final double beta;
        private final int userK;
        private final int itemK;

        public ItemBasedKNNBanditInteractiveRecommenderSupplier(int userK, int itemK,  double alpha, double beta)
        {
            this.alpha = alpha;
            this.beta = beta;
            this.userK = userK;
            this.itemK = itemK;
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            UpdateableSimilarity sim = new BetaStochasticSimilarity(userIndex.numUsers(), alpha, beta);
            return new InteractiveItemBasedKNN<>(userIndex, itemIndex, true, true, userK, itemK, sim);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.IBBANDIT + "-" + userK + "-" + itemK + "-" + alpha + "-" + beta;
        }
    }
}
