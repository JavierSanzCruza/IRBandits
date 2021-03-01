package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.wisdom.InformationTheoryUserDiversity;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

public class InformationTheoryUserDiversityConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    private final DoublePredicate predicate;
    public InformationTheoryUserDiversityConfigurator(DoublePredicate predicate)
    {
        this.predicate = predicate;
    }

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }
        return new AverageInteractiveRecommenderSupplier(ignoreUnknown);
    }

    private class AverageInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        private final boolean ignoreUnknown;
        public AverageInteractiveRecommenderSupplier(boolean ignoreUnknown)
        {
            this.ignoreUnknown = ignoreUnknown;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new InformationTheoryUserDiversity<>(userIndex, itemIndex, ignoreUnknown, predicate);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new InformationTheoryUserDiversity<>(userIndex, itemIndex, ignoreUnknown, rngSeed, predicate);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.INFTHEOR + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
