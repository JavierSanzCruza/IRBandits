package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.knn.item.InteractiveItemBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

public class ItemBasedKNNConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
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
        return new ItemBasedInteractiveRecommenderSupplier(k, ignoreZeroes, ignoreUnknown);
    }

    private class ItemBasedInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        private final boolean ignoreUnknown;
        private final boolean ignoreZeroes;
        private final int k;

        public ItemBasedInteractiveRecommenderSupplier(int k, boolean ignoreZeroes, boolean ignoreUnknown)
        {
            this.ignoreUnknown = ignoreUnknown;
            this.ignoreZeroes = ignoreZeroes;
            this.k = k;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            UpdateableSimilarity sim = new VectorCosineSimilarity(itemIndex.numItems());
            return new InteractiveItemBasedKNN<>(userIndex, itemIndex, ignoreUnknown, ignoreZeroes, 0, k, sim);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            UpdateableSimilarity sim = new VectorCosineSimilarity(itemIndex.numItems());
            return new InteractiveItemBasedKNN<>(userIndex, itemIndex, ignoreUnknown, rngSeed, ignoreZeroes, 0, k, sim);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.ITEMBASEDKNN + "-" + k + "-" + (ignoreZeroes ? "ignore" : "all") + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
