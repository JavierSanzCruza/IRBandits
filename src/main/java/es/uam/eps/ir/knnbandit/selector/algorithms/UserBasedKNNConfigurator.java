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
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.AdditiveRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.BestRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.InteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.LastRatingInteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.KNNBanditIdentifiers;
import org.json.JSONObject;

/**
 * Class for configuring the user-based kNN recommendation algorithm.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.knn.user.InteractiveUserBasedKNN
 */
public class UserBasedKNNConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for selecting whether the algorithm is updated with items unknown by the system or not.
     */
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    /**
     * Identifier for selecting whether we prefer zeroes to unknown values or not.
     */
    private static final String IGNOREZEROES = "ignoreZeroes";
    /**
     * Identifier for the number of neighbors of the target item to take.
     */
    private static final String K = "k";
    /**
     * Identifier for the variant of the algorithm to consider.
     */
    private static final String VARIANT = "variant";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        boolean ignoreZeroes = true;
        if(object.has(IGNOREZEROES))
        {
            ignoreZeroes = object.getBoolean(IGNOREZEROES);
        }

        String variant = KNNBanditIdentifiers.BASIC;
        if(object.has(VARIANT))
        {
            variant = object.getString(VARIANT);
        }

        int k = object.getInt(K);

        return new UserBasedInteractiveRecommenderSupplier(k, ignoreZeroes, ignoreUnknown, variant);
    }

    /**
     * Class for configuring the user-based kNN interactive recommender.
     */
    private class UserBasedInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        /**
         * True if we only use ratings we know about for the update procedure.
         */
        private final boolean ignoreUnknown;
        /**
         * True if we ignore zero ratings when updating.
         */
        private final boolean ignoreZeroes;
        /**
         * The number of neighbors of the candidate item.
         */
        private final int k;
        /**
         * The name of the variant.
         */
        private final String variant;

        /**
         * Constructor.
         *
         * @param k             the number of neighbors of the candidate item.
         * @param ignoreZeroes  true if ignore zero ratings when updating.
         * @param ignoreUnknown true if we only use ratings we know about for the update procedure, false otherwise.
         * @param variant       the name of the variant.
         */
        public UserBasedInteractiveRecommenderSupplier(int k, boolean ignoreZeroes, boolean ignoreUnknown, String variant)
        {
            this.ignoreUnknown = ignoreUnknown;
            this.ignoreZeroes = ignoreZeroes;
            this.k = k;
            this.variant = variant;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            UpdateableSimilarity sim = new VectorCosineSimilarity(userIndex.numUsers());

            switch(this.variant)
            {
                case KNNBanditIdentifiers.BASIC:
                    return new InteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.BEST:
                    return new BestRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.LAST:
                    return new LastRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.ADDITIVE:
                    return new AdditiveRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, ignoreZeroes, k, sim);
                default:
                    return null;
            }
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            UpdateableSimilarity sim = new VectorCosineSimilarity(userIndex.numUsers());

            switch(this.variant)
            {
                case KNNBanditIdentifiers.BASIC:
                    return new InteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, rngSeed, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.BEST:
                    return new BestRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, rngSeed, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.LAST:
                    return new LastRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, rngSeed, ignoreZeroes, k, sim);
                case KNNBanditIdentifiers.ADDITIVE:
                    return new AdditiveRatingInteractiveUserBasedKNN<>(userIndex, itemIndex, ignoreUnknown, rngSeed, ignoreZeroes, k, sim);
                default:
                    return null;
            }
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.USERBASEDKNN + "-" + variant + "-" + k + "-" + (ignoreZeroes ? "ignore" : "all") + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
