package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.clusters.club.CLUBComplete;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.stochastic.BetaStochasticSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.InteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

public class CLUBConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String ALPHA = "alpha";
    private static final String ALPHA2 = "alpha2";
    private static final String IGNOREUNKNOWN = "ignoreUnknown";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }


        double alpha = object.getDouble(ALPHA);
        double alpha2 = object.getDouble(ALPHA2);
        return new CLUBInteractiveRecommenderSupplier<>(alpha, alpha2, ignoreUnknown);
    }

    private static class CLUBInteractiveRecommenderSupplier<U,I> implements InteractiveRecommenderSupplier<U,I>
    {
        private final double alpha;
        private final double alpha2;
        private final boolean ignoreUnknown;

        public CLUBInteractiveRecommenderSupplier(double alpha, double alpha2, boolean ignoreUnknown)
        {
            this.alpha = alpha;
            this.alpha2 = alpha2;
            this.ignoreUnknown = ignoreUnknown;
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new CLUBComplete<>(userIndex, itemIndex, ignoreUnknown, alpha, alpha2);
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new CLUBComplete<>(userIndex, itemIndex, ignoreUnknown, rngSeed, alpha, alpha2);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.CLUB + "-" + alpha + "-" + alpha2 + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
