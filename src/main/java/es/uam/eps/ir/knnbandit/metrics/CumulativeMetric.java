/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.metrics;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import java.util.List;

/**
 * Interface for computing cumulative metrics.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface CumulativeMetric<U, I>
{
    /**
     * Obtains the current value of the metric.
     *
     * @return the value of the metric.
     */
    double compute();

    /**
     * Initializes the values without training data.
     *
     * @param dataset   the dataset containing the information.
     */
    void initialize(Dataset<U,I> dataset);


    /**
     * Initializes the values
     *
     * @param dataset       the dataset.
     * @param train         training data.
     */
    void initialize(Dataset<U,I> dataset, List<FastRating> train);

    /**
     * Updates the current value of the metric.
     *
     * @param uidx User identifier.
     * @param iidx Item identifier.
     */
    void update(int uidx, int iidx, double value);

    /**
     * Resets the metric.
     */
    void reset();
}
