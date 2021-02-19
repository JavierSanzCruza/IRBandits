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
import java.util.function.DoublePredicate;

/**
 * Clickthrough rate of the system: computes the fraction of recommendations
 * which provide a positive value.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ClickthroughRate<U,I> implements CumulativeMetric<U,I>
{
    /**
     * The number of successes
     */
    private double hits;
    /**
     * The total number of trials
     */
    private double total;
    /**
     * Relevance threshold
     */
    private DoublePredicate relevance;

    /**
     * Constructor.
     */
    public ClickthroughRate()
    {
        this.hits = 0.0;
        this.total = 0.0;
    }

    @Override
    public double compute()
    {
        return total > 0.0 ? hits/total : 0.0;
    }

    @Override
    public void initialize(Dataset<U, I> dataset)
    {
        this.relevance = dataset.getRelevanceChecker();
        this.reset();
    }

    @Override
    public void initialize(Dataset<U,I> dataset, List<FastRating> train)
    {
        this.initialize(dataset);
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        ++total;
        hits += relevance.test(value) ? 1.0 : 0.0;
    }

    @Override
    public void reset()
    {
        hits = 0.0;
        total = 0.0;
    }
}
