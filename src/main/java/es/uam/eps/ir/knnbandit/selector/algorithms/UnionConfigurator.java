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
import es.uam.eps.ir.knnbandit.recommendation.wisdom.Union;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

/**
 * Class for configuring an algorithm that takes the number of items which have been rated
 * by the users who have rated by the candidate items as recommendation score.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.wisdom.Union
 */
public class UnionConfigurator<U,I> extends AbstractAlgorithmConfigurator<U, I>
{
    /**
     * Checks the relevance of the rating values.
     */
    private final DoublePredicate relevanceChecker;

    /**
     * Constructor.
     * @param relevanceChecker checks the relevance of the rating values.
     */
    public UnionConfigurator(DoublePredicate relevanceChecker)
    {
        this.relevanceChecker = relevanceChecker;
    }

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        return new UnionInteractiveRecommenderSupplier();
    }

    /**
     * Class that configures an item centroid distance interactive recommender.
     */
    private class UnionInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new Union<>(userIndex, itemIndex, relevanceChecker);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new Union<>(userIndex, itemIndex, rngSeed, relevanceChecker);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.UNION;
        }
    }
}