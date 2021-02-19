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
import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;

import java.util.List;
import java.util.function.DoublePredicate;

/**
 * Cumulative implementation of global recall.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado Puig (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class CumulativeRecall<U, I> implements CumulativeMetric<U, I>
{
    /**
     * Number of relevant (user,item) pairs.
     */
    private int numRel;
    /**
     * Relevance checker.
     */
    private DoublePredicate relevance;
    /**
     * Number of training relevant (user,item) pairs.
     */
    private int toRemove;
    /**
     * Number of currently discovered (user, item) pairs.
     */
    private double current;

    /**
     * Constructor.
     */
    public CumulativeRecall()
    {
        this.current = 0.0;
        this.toRemove = 0;
    }

    @Override
    public double compute()
    {
        if (numRel == 0)
        {
            return 0.0;
        }
        return this.current / (this.numRel - this.toRemove + 0.0);
    }


    @Override
    public void initialize(Dataset<U,I> dataset)
    {
        this.numRel = dataset.getNumRel();
        // If no information is provided, we just count the number of positive ratings discovered.
        if(numRel == 0) numRel = 1;
        this.current = 0.0;
        this.relevance = dataset.getRelevanceChecker();
    }

    @Override
    public void initialize(Dataset<U,I> dataset, List<FastRating> train)
    {
        this.initialize(dataset);
        this.toRemove = train.stream().filter(rating -> relevance.test(rating.value())).mapToInt(x -> 1).sum();
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        if(relevance.test(value)) this.current++;
    }

    @Override
    public void reset()
    {
        this.current = 0.0;
    }

}
