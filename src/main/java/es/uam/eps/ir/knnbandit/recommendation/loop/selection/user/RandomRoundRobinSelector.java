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
 * Selects each user once. Then, it shuffles them all and changes the
 * order. The new order is taken until all users have been visited once more.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RandomRoundRobinSelector extends FastUserSelector
{
    /**
     * The current user to recommend items to
     */
    private int index;

    private int lastNumUsers;
    private int previousIndex;
    private boolean reshuffle;

    public RandomRoundRobinSelector(int rngSeed)
    {
        super(rngSeed);
        this.index = 0;
        this.previousIndex = 0;
        this.reshuffle = false;
        this.lastNumUsers = 0;
    }

    @Override
    public void init()
    {
        super.init();
        this.index = 0;
        this.previousIndex = 0;
        this.reshuffle = false;
        this.lastNumUsers = 0;
    }

    @Override
    public int next(int numUsers, int lastRemovedIndex)
    {
        int diff = lastNumUsers - numUsers;
        if(numUsers == 0) return -1;
        else if(lastNumUsers == 0)
        {
            index = 0;
        }
        else if(diff == 1 && lastRemovedIndex <= index)
        {
            index = index % numUsers;
        }
        else if(diff == 1)
        {
            index = (index+1) % numUsers;
        }
        else if(diff == 2 && lastRemovedIndex <= index)
        {
            index = (index-1) % numUsers;
        }
        else if(diff == 2)
        {
            index = index % numUsers;
        }
        else
        {
            index = (index+1) % numUsers;
        }

        if(previousIndex > 0 && index == 0)
        {
            reshuffle = true;
        }
        previousIndex = index;
        lastNumUsers = numUsers;
        return index;
    }

    @Override
    public boolean reshuffle()
    {
        if(reshuffle)
        {
            reshuffle = false;
            return true;
        }
        return false;
    }
}
