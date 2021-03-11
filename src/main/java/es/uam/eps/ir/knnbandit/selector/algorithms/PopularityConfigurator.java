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
import es.uam.eps.ir.knnbandit.recommendation.basic.PopularityRecommender;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

/**
 * Class for configuring the average rating algorithm.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.basic.PopularityRecommender
 */
public class PopularityConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Checks the relevance of the rating values.
     */
    private final DoublePredicate predicate;

    /**
     * Constructor.
     * @param predicate checks the relevance of the rating values.
     */
    public PopularityConfigurator(DoublePredicate predicate)
    {
        this.predicate = predicate;
    }

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        return new PopularityInteractiveRecommenderSupplier();
    }

    /**
     * Class for configuring the popularity-based recommender.
     */
    private class PopularityInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new PopularityRecommender<>(userIndex, itemIndex, predicate);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new PopularityRecommender<>(userIndex, itemIndex, rngSeed, predicate);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.POP;
        }
    }
}
