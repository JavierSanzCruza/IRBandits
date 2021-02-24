/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.metrics.atk;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.statistics.FastGiniIndex;

import java.util.List;

/**
 * Cumulative metric that computes the Gini index of the last k recommended items.
 *
 * @param <U> the type of the users.
 * @param <I> the type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class CumulativeGiniAtK<U, I> extends CumulativeMetricAtK<U, I>
{
    /**
     * The updateable Gini index.
     */
    private FastGiniIndex gini;

    /**
     * Constructor.
     *
     * @param k        the number of recommendations to consider.
     */
    public CumulativeGiniAtK(int k)
    {
        super(k);
    }

    @Override
    public void initialize(Dataset<U,I> dataset)
    {
        this.gini = new FastGiniIndex(dataset.numItems());
    }

    @Override
    public void initialize(Dataset<U,I> dataset, List<FastRating> warmup)
    {
        this.gini = new FastGiniIndex(dataset.numItems());
    }

    @Override
    public double compute()
    {
        return 1.0 - this.gini.getValue();
    }

    @Override
    protected void updateAdd(int uidx, int iidx)
    {
        this.gini.increaseFrequency(iidx);
    }

    @Override
    protected void updateDel(int uidx, int iidx)
    {
        this.gini.decreaseFrequency(iidx);
    }

    @Override
    protected void resetMetric()
    {
        this.gini.reset();
    }
}
