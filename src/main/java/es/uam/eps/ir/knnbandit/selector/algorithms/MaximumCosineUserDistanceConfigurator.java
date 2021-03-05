package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.wisdom.AverageCosineUserDistance;
import es.uam.eps.ir.knnbandit.recommendation.wisdom.MaximumCosineUserDistance;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

public class MaximumCosineUserDistanceConfigurator<U,I> extends AbstractAlgorithmConfigurator<U, I>
{
    private final DoublePredicate relevanceChecker;
    public MaximumCosineUserDistanceConfigurator(DoublePredicate relevanceChecker)
    {
        this.relevanceChecker = relevanceChecker;
    }

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        return new MaximumCosineUserDistanceInteractiveRecommenderSupplier();
    }

    private class MaximumCosineUserDistanceInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new MaximumCosineUserDistance<>(userIndex, itemIndex, relevanceChecker);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new MaximumCosineUserDistance<>(userIndex, itemIndex, rngSeed, relevanceChecker);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.MAXUSERCOS;
        }
    }
}