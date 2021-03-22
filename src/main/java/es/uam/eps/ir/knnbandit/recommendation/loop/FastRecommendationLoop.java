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

import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.ranksys.fast.FastRecommendation;

import java.util.stream.Stream;

/**
 * Interface for fast recommendation loops, relying on indexes instead of identifiers to
 * perform the different operations.
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface FastRecommendationLoop<U,I> extends RecommendationLoop<U,I>
{
    /**
     * Executes the complete following iteration of the recommendation loop.
     * @return a tuple indicating: the selected user index and the item index if the algorithm
     * is able to generate a recommendation, null otherwise.
     */
    Pair<Integer> fastNextIteration();
    /**
     * Obtains the result of a recommendation for the recommendation loop.
     * @return a tuple indicating: the selected user index, the item index if the algorithm
     * is able to generate a recommendation, null otherwise.
     */
    Pair<Integer> fastNextRecommendation();
    /**
     * Executes the complete following iteration of the recommendation loop. It returns a ranking of items.
     * @return a recommendation ranking.
     */
    FastRecommendation fastNextIterationList();

    /**
     * Obtains the result of a recommendation for the recommendation loop.
     * @return a recommendation ranking.
     */
    FastRecommendation fastNextRecommendationList();

    /**
     * Updates the algorithms and metrics after receiving a metric
     * @param uidx the index identifier of the user.
     * @param iidx the index identifier of the item.
     */
    void fastUpdate (int uidx, int iidx);
    /**
     * Updates the algorithms and metrics after receiving a metric
     * @param rec the recommendation ranking.
     */
    void fastUpdate(FastRecommendation rec);


    void fastUpdateNotRec (int uidx, int iidx);
    void fastUpdateNotRec(FastRecommendation rec);
    void fastUpdateRec(Stream<Pair<Integer>> pair);
    void fastUpdateRecList(Stream<FastRecommendation> recs);
    void fastUpdateRec(int uidx, int iidx);
    void fastUpdateRec(FastRecommendation rec);

}
