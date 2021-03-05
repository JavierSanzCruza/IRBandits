package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;

import es.uam.eps.ir.knnbandit.recommendation.wisdom.AverageCosineUserDistance;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;


import java.util.function.DoublePredicate;

public class AverageCosineUserDistanceConfigurator<U,I> extends AbstractAlgorithmConfigurator<U, I>
{
    private final DoublePredicate relevanceChecker;
    public AverageCosineUserDistanceConfigurator(DoublePredicate relevanceChecker)
    {
        this.relevanceChecker = relevanceChecker;
    }

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        return new AverageCosineUserDistanceInteractiveRecommenderSupplier();
    }

    private class AverageCosineUserDistanceInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new AverageCosineUserDistance<>(userIndex, itemIndex, relevanceChecker);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new AverageCosineUserDistance<>(userIndex, itemIndex, rngSeed, relevanceChecker);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.AVGUSERCOS;
        }
    }
}