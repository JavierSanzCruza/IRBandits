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
import es.uam.eps.ir.knnbandit.utils.statistics.FastGiniIndex;

import java.util.List;

/**
 * Cumulative version of the Gini index.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class CumulativeGini<U, I> implements CumulativeMetric<U, I>
{
    /**
     * The updateable Gini index to compute all the operations.
     */
    private FastGiniIndex gini;

    /**
     * Constructor.
     *
     */
    public CumulativeGini()
    {
    }

    @Override
    public void initialize(Dataset<U, I> dataset)
    {
        this.gini = new FastGiniIndex(dataset.numItems());
    }

    @Override
    public void initialize(Dataset<U, I> dataset, List<FastRating> train)
    {
        this.initialize(dataset);
    }

    @Override
    public double compute()
    {
        return 1.0 - this.gini.getValue();
    }

    @Override
    public void update(int uidx, int iidx, double value) {this.gini.increaseFrequency(1);}

    @Override
    public void reset()
    {
        this.gini.reset();
    }
}
