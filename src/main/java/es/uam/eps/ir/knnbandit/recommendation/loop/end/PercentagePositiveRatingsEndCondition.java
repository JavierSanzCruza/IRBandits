/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.end;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;

import java.util.function.DoublePredicate;

/**
 * End condition specifying a fixed number of positive ratings to be found.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class PercentagePositiveRatingsEndCondition implements EndCondition
{
    private double percentage;
    /**
     * The total number of relevant items to retrieve.
     */
    private int numRel;
    /**
     * The current number of relevant items.
     */
    private int currentRel;
    /**
     * The relevance threshold.
     */
    private DoublePredicate threshold;

    /**
     * Constructor. Fixes the number of relevant items to retrieve.
     * @param numRel the number of relevant items.
     */
    public PercentagePositiveRatingsEndCondition(int numRel)
    {
        this.numRel = numRel;
        this.percentage = -1;
    }

    /**
     * Constructor. Fixes the number of relevant items to retrieve.
     * @param percentage the percentage of relevant items we want to retrieve.
     */
    public PercentagePositiveRatingsEndCondition(double percentage)
    {
        this.numRel = -1;
        this.percentage = percentage;
    }

    @Override
    public void init(Dataset<?,?> dataset)
    {
        if(percentage > 0.0)
        {
            this.numRel = ((Double) Math.ceil(dataset.getNumRel() * percentage)).intValue();
        }
        this.currentRel = 0;
        this.threshold = dataset.getRelevanceChecker();
    }

    @Override
    public boolean hasEnded()
    {
        return (this.currentRel >= this.numRel);
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        if (threshold.test(value))
        {
            this.currentRel++;
        }
    }
}
