/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.selection.user;

import java.util.Random;

/**
 * Fast implementation of a user selector.
 */
public abstract class FastUserSelector implements UserSelector
{
    /**
     * The seed for the random number generator.
     */
    protected final int rngSeed;
    /**
     * The random number generator.
     */
    protected Random rng;

    /**
     * Constructor.
     * @param rngSeed a seed for a random number generator.
     */
    public FastUserSelector(int rngSeed)
    {
        this.rngSeed = rngSeed;
        this.rng = new Random(rngSeed);
    }

    /**
     * Constructor.Takes a fixed random seed.
     */
    public FastUserSelector()
    {
        this.rngSeed = 0;
        this.rng = new Random(rngSeed);
    }

    @Override
    public void init()
    {
        this.rng = new Random(rngSeed);
    }
}
