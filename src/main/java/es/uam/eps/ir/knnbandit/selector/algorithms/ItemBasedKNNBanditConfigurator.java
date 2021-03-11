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
import es.uam.eps.ir.knnbandit.recommendation.knn.item.InteractiveItemBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.stochastic.BetaStochasticSimilarity;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

/**
 * Class for configuring a bandit based on the item-based kNN algorithm.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.knn.item.InteractiveItemBasedKNN
 */
public class ItemBasedKNNBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for the initial alpha value of the Beta distributions.
     */
    private static final String ALPHA = "alpha";
    /**
     * Identifier for the initial beta value for the Beta distributions.
     */
    private static final String BETA = "beta";
    /**
     * Identifier for the number of rated items to select for the target user.
     */
    private static final String USERK = "userK";
    /**
     * Identifier for the number of neighbors of the target item to pick.
     */
    private static final String ITEMK = "itemK";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        int userK = object.getInt(USERK);
        int itemK = object.getInt(ITEMK);
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);
        return new ItemBasedKNNBanditInteractiveRecommenderSupplier(userK, itemK, alpha, beta);
    }

    /**
     * Class that configures an item kNN bandit.
     */
    private class ItemBasedKNNBanditInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        /**
         * Initial alpha value for the Beta distribution.
         */
        private final double alpha;
        /**
         * Initial beta value for the Beta distribution.
         */
        private final double beta;

        /**
         * Number of items rated by the target user to select.
         */
        private final int userK;
        /**
         * Number of items to take as neighbors of the candidate item.
         */
        private final int itemK;

        /**
         * Constructor.
         * @param userK     number of items rated by the target user to select as possible neighbors.
         * @param itemK     number of items to take as neighbors of the candidate item.
         * @param alpha     the initial alpha value for the Beta distribution.
         * @param beta      the initial beta value for the Beta distribution.
         */
        public ItemBasedKNNBanditInteractiveRecommenderSupplier(int userK, int itemK,  double alpha, double beta)
        {
            this.alpha = alpha;
            this.beta = beta;
            this.userK = userK;
            this.itemK = itemK;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            UpdateableSimilarity sim = new BetaStochasticSimilarity(userIndex.numUsers(), alpha, beta);
            return new InteractiveItemBasedKNN<>(userIndex, itemIndex, true, true, userK, itemK, sim);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            UpdateableSimilarity sim = new BetaStochasticSimilarity(userIndex.numUsers(), alpha, beta);
            return new InteractiveItemBasedKNN<>(userIndex, itemIndex, true, rngSeed,true, userK, itemK, sim);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.IBBANDIT + "-" + userK + "-" + itemK + "-" + alpha + "-" + beta;
        }
    }
}
