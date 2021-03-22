/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.ensembles;

import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.utils.Pair;

import java.util.List;

/**
 * Abstract class for defining interactive recommendation ensembles.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface Ensemble<U, I> extends InteractiveRecommender<U,I>
{
    /**
     * Obtains the number of algorithms in the ensemble.
     * @return the number of algorithms in the ensemble.
     */
    int getAlgorithmCount();

    /**
     * Obtains the names of the algorithms in the ensemble.
     * @return a list containing the names of the algorithms in the ensemble.
     */
    List<String> getAlgorithms();

    /**
     * Obtains the identifier of the algorithm currently used in the ensemble.
     * If the ensemble does not use a single algorithm in a given iteration,
     * or it has not selected any, it returns -1.
     * @return the identifier of the currently used algorithm in the ensemble.
     */
    int getCurrentAlgorithm();

    /**
     * Obtains the name of one of the algorithms.
     * @param idx the index of the algorithm.
     * @return the algorithm name if it exists, null otherwise.
     */
    String getAlgorithmName(int idx);

    /**
     * Gets the statistics for the recommendation algorithm (number of hits/misses).
     * If the information is not available, it returns null.
     *
     * @param idx the index of the algorithm.
     * @return a pair containing, in its first position, the hits of the algorithm, and, in the second, its misses. Null
     * if the algorithm does not exist or success information is not available.
     */
    Pair<Integer> getAlgorithmStats(int idx);

    /**
     * Sets the current algorithm in the ensemble to apply.
     * @param idx the identifier of the algorithm.
     */
    void setCurrentAlgorithm(int idx);
}
