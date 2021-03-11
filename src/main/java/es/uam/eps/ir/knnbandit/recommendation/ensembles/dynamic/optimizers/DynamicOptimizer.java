/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.ensembles.dynamic.optimizers;

import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.ranksys.core.Recommendation;

/**
 * In a dynamic ensemble, this interface defines the methods for computing the metric we use for selecting
 * the next algorithm to execute.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface DynamicOptimizer<U,I>
{
    /**
     * Initializes the optimizer.
     * @param dataset the offline dataset to use.
     * @param cutoff the maximum number of items in the recommendation.
     */
    void init(OfflineDataset<U,I> dataset, int cutoff);

    /**
     * Evaluates the recommendation.
     * @param rec the recommendation.
     * @return the metric value for the recommenation.
     */
    double evaluate(Recommendation<U,I> rec);
}
