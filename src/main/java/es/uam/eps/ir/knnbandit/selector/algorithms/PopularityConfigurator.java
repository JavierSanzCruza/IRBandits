package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.basic.PopularityRecommender;
import es.uam.eps.ir.knnbandit.recommendation.basic.RandomRecommender;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoublePredicate;

public class PopularityConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private final DoublePredicate predicate;

    public PopularityConfigurator(DoublePredicate predicate)
    {
        this.predicate = predicate;
    }

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        return new PopularityInteractiveRecommenderSupplier();
    }

    private class PopularityInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new PopularityRecommender<>(userIndex, itemIndex, predicate);
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new PopularityRecommender<>(userIndex, itemIndex, rngSeed, predicate);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.POP;
        }
    }
}
