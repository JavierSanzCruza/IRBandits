package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.CollaborativeGreedy;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.InteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

public class CollabGreedyConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    private static final String THRESHOLD = "threshold";
    private static final String ALPHA = "alpha";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        double threshold = object.getDouble(THRESHOLD);
        double alpha = object.getDouble(ALPHA);
        return new CollabGreedyInteractiveRecommenderSupplier<>(alpha, threshold, ignoreUnknown);
    }

    private static class CollabGreedyInteractiveRecommenderSupplier<U,I> implements InteractiveRecommenderSupplier<U,I>
    {
        private final double alpha;
        private final double threshold;
        private final boolean ignoreUnknown;

        public CollabGreedyInteractiveRecommenderSupplier(double alpha, double threshold, boolean ignoreUnknown)
        {
            this.alpha = alpha;
            this.ignoreUnknown = ignoreUnknown;
            this.threshold = threshold;
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new CollaborativeGreedy<>(userIndex, itemIndex, ignoreUnknown, threshold, alpha);
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new CollaborativeGreedy<>(userIndex, itemIndex, ignoreUnknown, rngSeed, threshold, alpha);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.COLLABGREEDY + "-" + threshold + "-" + alpha + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
