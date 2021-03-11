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
import es.uam.eps.ir.knnbandit.recommendation.wisdom.InformationTheoryUserDiversity;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

/**
 * Class for configuring the information theory user diversity algorithms.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.wisdom.InformationTheoryUserDiversity
 */
public class InformationTheoryUserDiversityConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for selecting whether the algorithm is updated with items unknown by the system or not.
     */
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    /**
     * Determines whether a rating is positive.
     */
    private final DoublePredicate predicate;

    /**
     * Constructor.
     * @param predicate determines whether a rating is positive or not.
     */
    public InformationTheoryUserDiversityConfigurator(DoublePredicate predicate)
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
        return new InformationTheoryUserDiversityInteractiveRecommenderSupplier(ignoreUnknown);
    }

    /**
     * Class that configures an information theory user diversity interactive recommender.
     */
    private class InformationTheoryUserDiversityInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        /**
         * True if we only use ratings we know about for the update procedure.
         */
        private final boolean ignoreUnknown;

        /**
         * Constructor.
         * @param ignoreUnknown true if we only use ratings we know about for the update procedure, false otherwise.
         */
        public InformationTheoryUserDiversityInteractiveRecommenderSupplier(boolean ignoreUnknown)
        {
            this.ignoreUnknown = ignoreUnknown;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new InformationTheoryUserDiversity<>(userIndex, itemIndex, ignoreUnknown, predicate);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new InformationTheoryUserDiversity<>(userIndex, itemIndex, ignoreUnknown, rngSeed, predicate);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.INFTHEOR + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
