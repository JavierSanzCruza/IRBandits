/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.metrics;

import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.statistics.GiniIndex2;
import org.jooq.lambda.tuple.Tuple2;

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
    private final GiniIndex2 gini;

    /**
     * Constructor.
     *
     * @param numItems The number of items.
     */
    public CumulativeGini(int numItems)
    {
        gini = new GiniIndex2(numItems);
    }

    @Override
    public void initialize(List<FastRating> train, boolean notReciprocal)
    {
        this.reset();
    }

    @Override
    public double compute()
    {
        return this.gini.getValue();
    }

    @Override
    public void update(int uidx, int iidx, double value) {this.gini.updateFrequency(iidx);}

    @Override
    public void reset()
    {
        this.gini.reset();
    }
}
