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
 * End condition that establishes the maximum number of iterations to execute.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class NumIterEndCondition implements EndCondition
{
    /**
     * The number of iterations.
     */
    private final int numIter;
    /**
     * The current iteration.
     */
    private int actualIter;

    /**
     * Constructor.
     *
     * @param numIter The number of iterations.
     */
    public NumIterEndCondition(int numIter)
    {
        this.numIter = numIter;
        this.init();
    }

    @Override
    public void init()
    {
        actualIter = 0;
    }

    @Override
    public boolean hasEnded()
    {
        return (actualIter >= numIter);
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        actualIter++;
    }
}
