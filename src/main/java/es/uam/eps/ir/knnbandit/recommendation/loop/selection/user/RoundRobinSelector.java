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
 * Selects the next user in a round robin way: each user must be selected once
 * after they have been recommended an item, and the order is fixed from
 * the beginning (it is randomized once)
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RoundRobinSelector extends FastUserSelector
{
    /**
     * The current user to recommend items to
     */
    private int index;

    /**
     * The last number of users
     */
    private int lastNumUsers;

    @Override
    public void init()
    {
        super.init();
        this.index = 0;
        this.lastNumUsers = 0;
    }

    @Override
    public int next(int numUsers, int lastRemovedIndex)
    {
        // Here, the number has been reduced:
        if(lastNumUsers == numUsers + 1 && lastRemovedIndex <= index)
        {
            index = index % numUsers;
        }
        else if(lastNumUsers == numUsers + 2 && lastRemovedIndex <= index)
        {
            index = (index - 1) % numUsers;
        }
        else if(lastNumUsers == numUsers + 2 && lastRemovedIndex > index)
        {
            index = index % numUsers;
        }
        else // if(lastNumUsers == numUsers + 1 && lastRemovedIndex > index) or (lastNumUsers == numUsers);
        {
            index = (index+1)%numUsers;
        }

        this.lastNumUsers = numUsers;
        return index;
    }

    @Override
    public boolean reshuffle()
    {
        return false;
    }


}
