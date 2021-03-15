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
import es.uam.eps.ir.knnbandit.recommendation.reranker.RankingCombiner;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

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
    /**
     * Identifier for the parameter that selects whether we use unknown information or not.
     */
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    /**
     * Identifier for the base algorithm.
     */
    private static final String FIRST = "first";
    /**
     * Identifier for the reranking algorithm.
     */
    private static final String SECOND = "second";
    /**
     * Identifier for the number of items to include in the initial ranking.
     */
    private static final String K = "k";
    /**
     * Identifier for the trade-off between the base and reranking algorithms.
     */
    private static final String LAMBDA = "lambda";

    /**
     * Checks whether a value is relevant or not for an algorithm.
     */
    private final DoublePredicate predicate;

    /**
     * Constructor.
     * @param predicate checks whether a value is relevant or not for an algorithm.
     */
    public RankingCombinerConfigurator(DoublePredicate predicate)
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

        AlgorithmSelector<U,I> selector = new AlgorithmSelector<>();
        selector.configure(predicate);
        // Select the base algorithm
        JSONObject firstAlgorithm = object.getJSONObject(FIRST);
        InteractiveRecommenderSupplier<U,I> first = selector.getAlgorithm(firstAlgorithm);

        // Select the reranking algorithm.
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
        /**
         * A supplier for the first algorithm (the base algorithm)
         */
        private final InteractiveRecommenderSupplier<U,I> firstAlg;
        /**
         * A supplier for the second algorithm (the reranking algorithm)
         */
        private final InteractiveRecommenderSupplier<U,I> secondAlg;
        /**
         * True if we ignore the unknown ratings, false otherwise.
         */
        private final boolean ignoreUnknown;
        /**
         * The cutoff we consider for the recommendation.
         */
        private final int k;
        /**
         * The trade-off between the base algorithm and the reranking one.
         */
        private final double lambda;

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
