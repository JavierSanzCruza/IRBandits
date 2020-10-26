package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.InteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.algorithms.AbstractAlgorithmConfigurator;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

public class UserBasedKNNConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    private static final String IGNOREZEROES = "ignoreZeroes";
    private static final String K = "k";

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

        int k = object.getInt(K);
        return new UserBasedInteractiveRecommenderSupplier(k, ignoreZeroes, ignoreUnknown);
    }

    private class UserBasedInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        private final boolean ignoreUnknown;
        private final boolean ignoreZeroes;
        private final int k;

        public UserBasedInteractiveRecommenderSupplier(int k, boolean ignoreZeroes, boolean ignoreUnknown)
        {
            this.ignoreUnknown = ignoreUnknown;
            this.ignoreZeroes = ignoreZeroes;
            this.k = k;
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            UpdateableSimilarity sim = new VectorCosineSimilarity(userIndex.numUsers());
            return new InteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, ignoreZeroes, k, sim);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.USERBASEDKNN + "-" + k + "-" + (ignoreZeroes ? "ignore" : "all") + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
