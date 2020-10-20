/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.end;

/**
 * End condition specifying a fixed number of positive ratings to be found.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class PercentagePositiveRatingsEndCondition implements EndCondition
{
    /**
     * The total number of relevant items to retrieve.
     */
    private final int numRel;
    /**
     * The current number of relevant items.
     */
    private int currentRel;
    /**
     * The relevance threshold.
     */
    private final double threshold;

    /**
     * Constructor. Fixes the number of relevant items to retrieve.
     * @param numRel the number of relevant items.
     * @param threshold the relevance threshold.
     */
    public PercentagePositiveRatingsEndCondition(int numRel, double threshold)
    {
        this.numRel = numRel;
        this.threshold = threshold;
        this.init();
    }

    /**
     * Constructor. Fixes the number of relevant items to retrieve.
     * @param totalRel the total number of relevant items.
     * @param percentage the percentage of relevant items we want to retrieve.
     * @param threshold the relevance threshold.
     */
    public PercentagePositiveRatingsEndCondition(int totalRel, double percentage, double threshold)
    {
        this.numRel = ((Double) Math.ceil(totalRel * percentage)).intValue();
        this.threshold = threshold;
        this.init();
    }

    @Override
    public void init()
    {
        this.currentRel = 0;
    }

    @Override
    public boolean hasEnded()
    {
        return (this.currentRel >= this.numRel);
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        if (value >= threshold)
        {
            this.currentRel++;
        }
    }
}
