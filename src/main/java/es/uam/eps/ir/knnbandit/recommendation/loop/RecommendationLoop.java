/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.loop;

import es.uam.eps.ir.knnbandit.utils.Rating;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;

import java.util.List;
import java.util.Map;

/**
 * A general interface for defining interactive recommendation loops.
 * @param <U> type of the users.
 * @param <I> type of the items.
 */
public interface RecommendationLoop<U,I>
{
    /**
     * Initializes all the necessary structures and metrics for the interactive recommendation loop.
     */
    void init();

    /**
     * Initializes the necessary structures and metrics for the interactive recommendation loop,
     * having some initial data that the corresponding algorithm can use as training.
     * @param warmup the warmup data.
     */
    void init(Warmup warmup);

    /**
     * Executes the complete following iteration of the recommendation loop.
     * @return a triplet indicating: the selected user, the item id, and the payoff of the recommendation if the algorithm
     * is able to generate a recommendation, null otherwise.
     */
    Tuple2<U,I> nextIteration();

    /**
     * Obtains the result of a recommendation for the recommendation loop.
     * @return A triplet, containing three values: the userId, the itemId, and the payoff of the recommendation.
     */
    Tuple2<U,I> nextRecommendation();

    /**
     * Updates the algorithms and metrics after receiving a metric
     * @param u the identifier of the user.
     * @param i the identifier of the items.
     */
    void update(U u, I i);

    /**
     * Checks whether a recommendation loop has ended or not.
     * @return true if the loop has ended, false otherwise.
     */
    boolean hasEnded();

    /**
     * Obtains the number of the current iteration.
     * @return the current iteration number.
     */
    int getCurrentIter();

    /**
     * Obtains the current metric values.
     * @return the current metric values if everything is OK, null otherwise.
     */
    Map<String, Double> getMetricValues();

    /**
     * Obtains the names of the metrics used in the loop.
     * @return a list containing the names of the metrics used in the loop.
     */
    List<String> getMetrics();
}