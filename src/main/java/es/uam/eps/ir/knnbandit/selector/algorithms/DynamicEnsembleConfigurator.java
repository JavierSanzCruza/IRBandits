/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
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
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class DynamicEnsembleConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for selecting whether the algorithm is updated with items unknown by the system or not.
     */
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    /**
     * Identifier for the list of algorithms to retrieve.
     */
    private static final String ALGORITHMS = "algorithms";
    /**
     * Identifier for the number of epochs before a new algorithm is selected.
     */
    private static final String NUMEPOCHS = "numEpochs";
    /**
     * Identifier for the validation cut-off.
     */
    private static final String VALIDCUTOFF = "validCutoff";
    /**
     * Identifier for the proportion of the known data that is used as input in the algorithm selection.
     */
    private static final String PERCVALID = "percValid";
    /**
     * Identifier for the metric which has to be used to select the algorithm.
     */
    private static final String OPTIMIZER = "optimizer";
    /**
     * Identifier for the metric name.
     */
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

        return new DynamicEnsembleRecommenderSupplier(recs, numEpochs, validationCutoff, percValid, optimizer, ignoreUnknown);
    }

    /**
     * Given its name, it selects the metric to optimize.
     * @param name the metric name.
     * @return the metric to optimize.
     */
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
     */
    private class DynamicEnsembleRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        /**
         * A map containing algorithm suppliers.
         */
        Map<String, InteractiveRecommenderSupplier<U,I>> algs;
        /**
         * The number of epochs befor updating the algorithm.
         */
        int numEpochs;
        /**
         * The cutoff for the validation recommendations.
         */
        int validCutoff;
        /**
         * The percentage of the available data which is used as training for the algorithm selection.
         */
        double percValid;
        /**
         * The metric we optimize during validation.
         */
        DynamicOptimizer<U,I> optimizer;
        /**
         * True if we ignore the unknown ratings, false otherwise.
         */
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
