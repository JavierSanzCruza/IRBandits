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
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.stochastic.BetaStochasticSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.AdditiveRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.BestRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.InteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.LastRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.KNNBanditIdentifiers;
import org.json.JSONObject;

/**
 * Class for configuring the user-based kNN bandit.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.knn.user.InteractiveUserBasedKNN
 */
public class UserBasedKNNBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for the initial value of the alpha parameter for the Beta distributions.
     */
    private static final String ALPHA = "alpha";
    /**
     * Identifier for the initial value of the beta parameter for the Beta distributions.
     */
    private static final String BETA = "beta";
    /**
     * Identifier for the number of neighbors to choose.
     */
    private static final String K = "k";
    /**
     * Identifier for the variant of the algorithm to select.
     */
    private static final String VARIANT = "variant";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        int k = object.getInt(K);
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);

        String variant = KNNBanditIdentifiers.BASIC;
        if(object.has(VARIANT))
        {
            variant = object.getString(VARIANT);
        }

        return new UserBasedKNNBanditInteractiveRecommenderSupplier(k, alpha, beta, variant);
    }

    /**
     * Configures the user-based kNN bandit algorithm.
     */
    private class UserBasedKNNBanditInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        /**
         * The initial value of the alpha parameter of the Beta distribution
         */
        private final double alpha;
        /**
         * The initial value of the beta parameter of the Beta distribution.
         */
        private final double beta;
        /**
         * The number of neighbors.
         */
        private final int k;
        /**
         * The name of the variant.
         */
        private final String variant;

        /**
         * Constructor.
         * @param k         the name of the variant.
         * @param alpha     the initial value of the alpha parameter of the Beta distribution
         * @param beta      the initial value of the beta parameter of the Beta distribution
         * @param variant   the variant of the algorithm to consider.
         */
        public UserBasedKNNBanditInteractiveRecommenderSupplier(int k,  double alpha, double beta, String variant)
        {
            this.alpha = alpha;
            this.beta = beta;
            this.k = k;
            this.variant = variant;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            UpdateableSimilarity sim = new BetaStochasticSimilarity(userIndex.numUsers(), alpha, beta);

            switch(this.variant)
            {
                case KNNBanditIdentifiers.BASIC:
                    return new InteractiveUserBasedKNN<>(userIndex, itemIndex, true, true, k, sim);
                case KNNBanditIdentifiers.BEST:
                    return new BestRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, true, k, sim);
                case KNNBanditIdentifiers.LAST:
                    return new LastRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, true, k, sim);
                case KNNBanditIdentifiers.ADDITIVE:
                    return new AdditiveRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, true, k, sim);
                default:
                    return null;
            }
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            UpdateableSimilarity sim = new BetaStochasticSimilarity(userIndex.numUsers(), alpha, beta);

            switch(this.variant)
            {
                case KNNBanditIdentifiers.BASIC:
                    return new InteractiveUserBasedKNN<>(userIndex, itemIndex, true, rngSeed, true, k, sim);
                case KNNBanditIdentifiers.BEST:
                    return new BestRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, rngSeed,true, k, sim);
                case KNNBanditIdentifiers.LAST:
                    return new LastRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, rngSeed,true, k, sim);
                case KNNBanditIdentifiers.ADDITIVE:
                    return new AdditiveRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, true, rngSeed, true, k, sim);
                default:
                    return null;
            }
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.UBBANDIT + "-" + variant + "-" + k + "-" + alpha + "-" + beta;
        }
    }
}
