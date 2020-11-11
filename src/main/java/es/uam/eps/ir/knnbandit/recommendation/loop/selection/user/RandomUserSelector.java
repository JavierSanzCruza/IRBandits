/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.selection.user;

/**
 * Each iteration, selects a user at random.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RandomUserSelector extends FastUserSelector
{

    /**
     * Constructor.
     * @param rngSeed a random seed
     */
    public RandomUserSelector(int rngSeed)
    {
        super(rngSeed);
    }


    @Override
    public int next(int numUsers, int lastRemovedIndex)
    {
        if(numUsers == 0)
        {
            return -1;
        }
        return rng.nextInt(numUsers);
    }

    @Override
    public boolean reshuffle()
    {
        return false;
    }
}
