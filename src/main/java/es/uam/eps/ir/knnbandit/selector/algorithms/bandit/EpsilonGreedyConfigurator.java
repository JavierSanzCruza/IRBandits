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

import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.EpsilonGreedy;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.EpsilonGreedyUpdateFunction;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.EpsilonGreedyUpdateFunctions;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.AbstractMultiArmedBandit;
import es.uam.eps.ir.knnbandit.selector.EpsilonGreedyUpdateFunctionIdentifiers;
import org.jooq.lambda.tuple.Tuple2;
import org.json.JSONObject;

/**
 * Class for configuring the epsilon-greedy multi-armed bandit algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.EpsilonGreedy
 */
public class EpsilonGreedyConfigurator extends AbstractBanditConfigurator
{
    /**
     * Identifier for the probability of exploration.
     */
    private final static String EPSILON = "epsilon";
    /**
     * Identifier for the update function.
     */
    private final static String UPDATEFUNC = "updateFunc";
    /**
     * Identifier for the name of the function.
     */
    private final static String FUNCTION = "function";
    /**
     * Identifier, of the decay factor for the non-stationary update function.
     */
    private final static String ALPHA = "alpha";

    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        double epsilon = object.getDouble(EPSILON);
        JSONObject function = object.getJSONObject(UPDATEFUNC);
        Tuple2<String, EpsilonGreedyUpdateFunction> updFunction = getUpdateFunction(function);
        if(updFunction == null) return null;

        String functionName = updFunction.v1;
        EpsilonGreedyUpdateFunction updateFunction = updFunction.v2;
        return new EpsilonGreedyBanditSupplier(epsilon, functionName, updateFunction);
    }

    /**
     * Obtains a function to update an Epsilon-greedy algorithm.
     *
     * @return the update function if everything is OK, null otherwise.
     */
    static Tuple2<String, EpsilonGreedyUpdateFunction> getUpdateFunction(JSONObject object)
    {
        String name = object.getString(FUNCTION);
        switch (name)
        {
            case EpsilonGreedyUpdateFunctionIdentifiers.STATIONARY:
                return new Tuple2<>(name, EpsilonGreedyUpdateFunctions.stationary());
            case EpsilonGreedyUpdateFunctionIdentifiers.NONSTATIONARY:
                double alpha = object.getDouble(ALPHA);
                return new Tuple2<>(name + "-" + alpha, EpsilonGreedyUpdateFunctions.nonStationary(alpha));
            case EpsilonGreedyUpdateFunctionIdentifiers.USEALL:
                return new Tuple2<>(name, EpsilonGreedyUpdateFunctions.useall());
            case EpsilonGreedyUpdateFunctionIdentifiers.COUNT:
                return new Tuple2<>(name, EpsilonGreedyUpdateFunctions.count());
            default:
                return null;
        }
    }

    /**
     * Class for configuring an epsilon-greedy multi-armed bandit.
     */
    private static class EpsilonGreedyBanditSupplier implements BanditSupplier
    {
        /**
         * The probability of exploration.
         */
        private final double epsilon;
        /**
         * The update function.
         */
        private final EpsilonGreedyUpdateFunction updateFunction;
        /**
         * The name of the update function.
         */
        private final String functionName;

        /**
         * Constructor.
         * @param epsilon           the probability of exploration for the bandit.
         * @param functionName      the name of the update function.
         * @param updateFunction    the function for updating the bandit.
         */
        public EpsilonGreedyBanditSupplier(double epsilon, String functionName, EpsilonGreedyUpdateFunction updateFunction)
        {
            this.epsilon = epsilon;
            this.functionName = functionName;
            this.updateFunction = updateFunction;
        }

        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new EpsilonGreedy(numItems, epsilon, updateFunction);
        }

        @Override
        public String getName()
        {
            return MultiArmedBanditIdentifiers.EGREEDY + "-" + epsilon + "-" + functionName;
        }
    }
}
