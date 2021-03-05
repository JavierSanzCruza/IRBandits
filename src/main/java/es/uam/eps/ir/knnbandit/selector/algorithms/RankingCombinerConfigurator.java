package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.reranker.RankingCombiner;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import org.json.JSONObject;

/**
 * Class for configuring a RankingCombiner algorithm.
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RankingCombinerConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    private static final String FIRST = "first";
    private static final String SECOND = "second";
    private static final String K = "k";
    private static final String LAMBDA = "lambda";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        AlgorithmSelector<U,I> selector = new AlgorithmSelector<>();

        JSONObject firstAlgorithm = object.getJSONObject(FIRST);
        InteractiveRecommenderSupplier<U,I> first = selector.getAlgorithm(firstAlgorithm);

        JSONObject secondAlgorithm = object.getJSONObject(SECOND);
        InteractiveRecommenderSupplier<U,I> second = selector.getAlgorithm(secondAlgorithm);

        int k = object.getInt(K);
        double lambda = object.getDouble(LAMBDA);

        return new RankingCombinerRecommenderSupplier<>(first, second, ignoreUnknown, k, lambda);
    }

    /**
     * Supplier for RankingCombiner algorithms.
     * @param <U> type of the users.
     * @param <I> type of the items.
     */
    private static class RankingCombinerRecommenderSupplier<U,I> implements InteractiveRecommenderSupplier<U,I>
    {
        InteractiveRecommenderSupplier<U,I> firstAlg;
        InteractiveRecommenderSupplier<U,I> secondAlg;
        boolean ignoreUnknown;
        int k;
        double lambda;

        /**
         * Constructor.
         * @param firstAlg a supplier for the first algorithm (the base algorithm).
         * @param secondAlg a supplier for the second algorithm (the reranking algorithm).
         * @param ignoreUnknown true if we ignore the unknown ratings, false otherwise.
         * @param k the cutoff we consider for the recommendation.
         * @param lambda the trade-off between the base algorithm and the reranking one.
         */
        public RankingCombinerRecommenderSupplier(InteractiveRecommenderSupplier<U,I> firstAlg, InteractiveRecommenderSupplier<U,I> secondAlg, boolean ignoreUnknown, int k, double lambda)
        {
            this.firstAlg = firstAlg;
            this.secondAlg = secondAlg;
            this.ignoreUnknown = ignoreUnknown;
            this.k = k;
            this.lambda = lambda;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new RankingCombiner<>(userIndex, itemIndex, ignoreUnknown, firstAlg.apply(userIndex, itemIndex), secondAlg.apply(userIndex,itemIndex), k, lambda);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new RankingCombiner<>(userIndex, itemIndex, ignoreUnknown, rngSeed, firstAlg.apply(userIndex, itemIndex), secondAlg.apply(userIndex,itemIndex), k, lambda);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.RANKINGCOMB + "-" + firstAlg.getName() + "-" + secondAlg.getName() + "-" + k + "-" + lambda + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
