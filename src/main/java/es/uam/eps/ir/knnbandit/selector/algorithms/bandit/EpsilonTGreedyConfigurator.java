/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.AbstractMultiArmedBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.EpsilonGreedyUpdateFunction;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.EpsilonTGreedy;
import org.jooq.lambda.tuple.Tuple2;
import org.json.JSONObject;

/**
 * Class for configuring the epsilon-t greedy multi-armed bandit algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.EpsilonTGreedy
 */
public class EpsilonTGreedyConfigurator extends AbstractBanditConfigurator
{
    /**
     * Identifier of the update function.
     */
    private final static String UPDATEFUNC = "updateFunc";
    /**
     * Identifier of the decay factor for the exploration.
     */
    private final static String ALPHA = "alpha";

    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        JSONObject function = object.getJSONObject(UPDATEFUNC);
        Tuple2<String, EpsilonGreedyUpdateFunction> updFunction = EpsilonGreedyConfigurator.getUpdateFunction(function);
        if(updFunction == null) return null;

        String functionName = updFunction.v1;
        EpsilonGreedyUpdateFunction updateFunction = updFunction.v2;
        return new EpsilonTGreedyBanditSupplier(alpha, functionName, updateFunction);
    }

    /**
     * Configures a epsilon-t greedy bandit supplier.
     */
    private static class EpsilonTGreedyBanditSupplier implements BanditSupplier
    {
        /**
         * The slope parameter.
         */
        private final double alpha;
        /**
         * The update function for the value estimations.
         */
        private final EpsilonGreedyUpdateFunction updateFunction;
        /**
         * The name of the update function.
         */
        private final String functionName;

        /**
         * Constructor.
         *
         * @param alpha             slope parameter.
         * @param functionName      the name of the update function.
         * @param updateFunction    the update function for the arms.
         */
        public EpsilonTGreedyBanditSupplier(double alpha, String functionName, EpsilonGreedyUpdateFunction updateFunction)
        {
            this.alpha = alpha;
            this.functionName = functionName;
            this.updateFunction = updateFunction;
        }

        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new EpsilonTGreedy(numItems, alpha, updateFunction);
        }

        @Override
        public String getName()
        {
            return MultiArmedBanditIdentifiers.ETGREEDY + "-" + alpha + "-" + functionName;
        }
    }
}
