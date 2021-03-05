package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.ensembles.DynamicEnsemble;
import es.uam.eps.ir.knnbandit.recommendation.ensembles.dynamic.optimizers.DynamicOptimizer;
import es.uam.eps.ir.knnbandit.recommendation.ensembles.dynamic.optimizers.NDCGOptimizer;
import es.uam.eps.ir.knnbandit.recommendation.ensembles.dynamic.optimizers.PrecisionOptimizer;
import es.uam.eps.ir.knnbandit.recommendation.ensembles.dynamic.optimizers.RecallOptimizer;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.algorithms.dynamic.optimizer.DynamicOptimizerIdentifiers;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

/**
 * Class for configuring a dynamic ensemble.
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class DynamicEnsembleConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    private static final String ALGORITHMS = "algorithms";
    private static final String NUMEPOCHS = "numEpochs";
    private static final String VALIDCUTOFF = "validCutoff";
    private static final String PERCVALID = "percValid";
    private static final String OPTIMIZER = "optimizer";
    private static final String NAME = "name";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        AlgorithmSelector<U,I> selector = new AlgorithmSelector<>();

        JSONArray algorithms = object.getJSONArray(ALGORITHMS);
        Map<String, InteractiveRecommenderSupplier<U,I>> recs = new HashMap<>();
        int numAlgs = algorithms.length();
        for(int i = 0; i < numAlgs; ++i)
        {
            InteractiveRecommenderSupplier<U,I> rec = selector.getAlgorithm(algorithms.getJSONObject(i));
            recs.put(rec.getName(), rec);
        }

        int numEpochs = object.getInt(NUMEPOCHS);
        int validationCutoff = object.getInt(VALIDCUTOFF);
        double percValid = object.getDouble(PERCVALID);
        JSONObject optim = object.getJSONObject(OPTIMIZER);
        DynamicOptimizer<U,I> optimizer = this.selectDynamicOptimizerConfigurator(optim.getString(NAME));
        // Select the metric to optimize

        return new DynamicEnsembleRecommenderSupplier<>(recs, numEpochs, validationCutoff, percValid, optimizer, ignoreUnknown);
    }

    protected DynamicOptimizer<U,I> selectDynamicOptimizerConfigurator(String name)
    {
        switch(name)
        {
            case DynamicOptimizerIdentifiers.P:
                return new PrecisionOptimizer<>();
            case DynamicOptimizerIdentifiers.R:
                return new RecallOptimizer<>();
            case DynamicOptimizerIdentifiers.NDCG:
                return new NDCGOptimizer<>();
            default:
                return null;
        }
    }

    /**
     * Supplier for RankingCombiner algorithms.
     * @param <U> type of the users.
     * @param <I> type of the items.
     */
    private static class DynamicEnsembleRecommenderSupplier<U,I> implements InteractiveRecommenderSupplier<U,I>
    {
        Map<String, InteractiveRecommenderSupplier<U,I>> algs;
        int numEpochs;
        int validCutoff;
        double percValid;
        DynamicOptimizer<U,I> optimizer;
        boolean ignoreUnknown;
        /**
         * Constructor.
         * @param algs          the list of interactive recommenders to combine.
         * @param numEpochs     the number of epochs before updating the algorithm.
         * @param ignoreUnknown true if we ignore the unknown ratings, false otherwise.
         * @param validCutoff   the cutoff for the validation recommendations.
         * @param optimizer     the metric we optimize during validation.
         */
        public DynamicEnsembleRecommenderSupplier(Map<String, InteractiveRecommenderSupplier<U,I>> algs, int numEpochs, int validCutoff, double percValid, DynamicOptimizer<U,I> optimizer, boolean ignoreUnknown)
        {
            this.algs = algs;
            this.numEpochs = numEpochs;
            this.ignoreUnknown = ignoreUnknown;
            this.validCutoff = validCutoff;
            this.percValid = percValid;
            this.optimizer = optimizer;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new DynamicEnsemble<>(userIndex, itemIndex, ignoreUnknown, algs, numEpochs, validCutoff, percValid, optimizer);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new DynamicEnsemble<>(userIndex, itemIndex, ignoreUnknown, rngSeed, algs, numEpochs, validCutoff, percValid, optimizer);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.RANKINGCOMB + "-" + numEpochs + "-" + validCutoff + "-" + percValid;
        }
    }
}
