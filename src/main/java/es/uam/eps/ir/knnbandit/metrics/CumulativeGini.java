/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.metrics;

import es.uam.eps.ir.knnbandit.utils.statistics.GiniIndex;
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
    private final GiniIndex gini;

    /**
     * Constructor.
     *
     * @param numItems The number of items.
     */
    public CumulativeGini(int numItems)
    {
        gini = new GiniIndex(numItems);
    }

    @Override
    public void initialize(List<Tuple2<Integer, Integer>> train, boolean notReciprocal)
    {

    }

    @Override
    public double compute()
    {
        return this.gini.getValue();
    }

    @Override
    public void update(int uidx, int iidx)
    {
        this.gini.updateFrequency(iidx, 1);
    }

    @Override
    public void reset()
    {
        this.gini.reset();
    }
}
