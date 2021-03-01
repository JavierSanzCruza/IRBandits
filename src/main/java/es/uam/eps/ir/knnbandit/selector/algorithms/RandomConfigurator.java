package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.basic.RandomRecommender;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

public class RandomConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        return new RandomInteractiveRecommenderSupplier<>();
    }

    private static class RandomInteractiveRecommenderSupplier<U,I> implements InteractiveRecommenderSupplier<U,I>
    {

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new RandomRecommender<>(userIndex, itemIndex);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new RandomRecommender<>(userIndex, itemIndex, rngSeed);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.RANDOM;
        }
    }
}
