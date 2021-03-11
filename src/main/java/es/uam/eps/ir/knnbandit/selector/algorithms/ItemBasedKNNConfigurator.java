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
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

/**
 * Class for configuring the item-based kNN algorithm.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.knn.item.InteractiveItemBasedKNN
 */
public class ItemBasedKNNConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
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
     * The number of neighbors of the target item to take.
     */
    private static final String K = "k";

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

        int k = object.getInt(K);
        return new ItemBasedInteractiveRecommenderSupplier(k, ignoreZeroes, ignoreUnknown);
    }

    /**
     * Class that configures an item-based kNN recommender.
     */
    private class ItemBasedInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
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
         * Constructor.
         *
         * @param k             the number of neighbors of the candidate item.
         * @param ignoreZeroes  true if ignore zero ratings when updating.
         * @param ignoreUnknown true if we only use ratings we know about for the update procedure, false otherwise.
         */
        public ItemBasedInteractiveRecommenderSupplier(int k, boolean ignoreZeroes, boolean ignoreUnknown)
        {
            this.ignoreUnknown = ignoreUnknown;
            this.ignoreZeroes = ignoreZeroes;
            this.k = k;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            UpdateableSimilarity sim = new VectorCosineSimilarity(itemIndex.numItems());
            return new InteractiveItemBasedKNN<>(userIndex, itemIndex, ignoreUnknown, ignoreZeroes, 0, k, sim);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            UpdateableSimilarity sim = new VectorCosineSimilarity(itemIndex.numItems());
            return new InteractiveItemBasedKNN<>(userIndex, itemIndex, ignoreUnknown, rngSeed, ignoreZeroes, 0, k, sim);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.ITEMBASEDKNN + "-" + k + "-" + (ignoreZeroes ? "ignore" : "all") + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
