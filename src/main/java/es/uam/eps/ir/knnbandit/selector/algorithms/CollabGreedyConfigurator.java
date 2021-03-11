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
import es.uam.eps.ir.knnbandit.recommendation.knn.user.CollaborativeGreedy;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

/**
 * Class for configuring the collaborative-greedy algorithm.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.knn.user.CollaborativeGreedy
 */
public class CollabGreedyConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for selecting whether the algorithm is updated with items unknown by the system or not.
     */
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    /**
     * Identifier for the similarity threshold.
     */
    private static final String THRESHOLD = "threshold";
    /**
     * Identifier for the exploration probability.
     */
    private static final String ALPHA = "alpha";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        double threshold = object.getDouble(THRESHOLD);
        double alpha = object.getDouble(ALPHA);
        return new CollabGreedyInteractiveRecommenderSupplier(alpha, threshold, ignoreUnknown);
    }

    /**
     * Configures a collaborative-greedy algorithm.
     */
    private class CollabGreedyInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        /**
         * Value for determining the exploration probability.
         */
        private final double alpha;
        /**
         * Similarity threshold. Two users are similar if their similarity is smaller than this value.
         */
        private final double threshold;
        /**
         * True if (user, item) pairs without training must be ignored.
         */
        private final boolean ignoreUnknown;

        /**
         * Constructor.
         * @param ignoreUnknown     true if (user, item) pairs without training must be ignored.
         * @param threshold         similarity threshold. Two users are similar if their similarity is smaller than this value.
         * @param alpha             value for determining the exploration probability.
         */
        public CollabGreedyInteractiveRecommenderSupplier(double alpha, double threshold, boolean ignoreUnknown)
        {
            this.alpha = alpha;
            this.ignoreUnknown = ignoreUnknown;
            this.threshold = threshold;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new CollaborativeGreedy<>(userIndex, itemIndex, ignoreUnknown, threshold, alpha);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new CollaborativeGreedy<>(userIndex, itemIndex, ignoreUnknown, rngSeed, threshold, alpha);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.COLLABGREEDY + "-" + threshold + "-" + alpha + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
