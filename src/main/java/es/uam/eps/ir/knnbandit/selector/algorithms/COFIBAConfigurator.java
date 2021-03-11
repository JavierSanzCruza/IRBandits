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
import es.uam.eps.ir.knnbandit.recommendation.clusters.cofiba.COFIBAErdos;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

/**
 * Class for configuring the COllaborative FIltering Bandits algorithm.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.clusters.cofiba.COFIBAErdos
 */
public class COFIBAConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for the parameter that manages the importance of the confidence bound for the item selection.
     */
    private static final String ALPHA = "alpha";
    /**
     * Identifier for the parameter that manages how difficult is for an edge in the user and item graphs to disappear.
     */
    private static final String ALPHA2 = "alpha2";
    /**
     * Identifier for selecting whether the algorithm is updated with items unknown by the system or not.
     */
    private static final String IGNOREUNKNOWN = "ignoreUnknown";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        double alpha = object.getDouble(ALPHA);
        double alpha2 = object.getDouble(ALPHA2);
        return new COFIBAInteractiveRecommenderSupplier<>(alpha, alpha2, ignoreUnknown);
    }

    /**
     * Class that configures COFIBA recommender.
     */
    private static class COFIBAInteractiveRecommenderSupplier<U,I> implements InteractiveRecommenderSupplier<U,I>
    {
        /**
         * Parameter that manages the importance of the confidence bound for the item selection.
         */
        private final double alpha;
        /**
         * Parameter that manages how difficult is for an edge in the graph to disappear.
         */
        private final double alpha2;
        /**
         * True if we only use ratings we know about for the update procedure.
         */
        private final boolean ignoreUnknown;

        /**
         * Constructor.
         *
         * @param alpha         parameter that manages the importance of the confidence bound for the item selection.
         * @param alpha2        parameter that manages how difficult is for an edge in the graph to disappear.
         * @param ignoreUnknown true if (user, item) pairs without training must be ignored.
         */
        public COFIBAInteractiveRecommenderSupplier(double alpha, double alpha2, boolean ignoreUnknown)
        {
            this.alpha = alpha;
            this.alpha2 = alpha2;
            this.ignoreUnknown = ignoreUnknown;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new COFIBAErdos<>(userIndex, itemIndex, ignoreUnknown, alpha, alpha2);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new COFIBAErdos<>(userIndex, itemIndex, ignoreUnknown, rngSeed, alpha, alpha2);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.COFIBA + "-" + alpha + "-" + alpha2 + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
