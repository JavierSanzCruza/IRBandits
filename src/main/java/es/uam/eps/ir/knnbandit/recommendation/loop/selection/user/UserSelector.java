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
 * Determines an strategy for selecting the next user in the recommendation.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface UserSelector
{
    /**
     * Selects the index of the user list for the next user.
     * @param numUsers the number of users.
     * @param lastRemovedIndex the index of the last removed user in the list.
     * @return the next index if possible, -1 otherwise.
     */
    int next(int numUsers, int lastRemovedIndex);

    /**
     * Indicates if the list of target users has to be randomly shuffled.
     * @return true if it has to be, false otherwise.
     */
    boolean reshuffle();

    /**
     * Initializes the values for the user selection strategy to work.
     */
    void init();
}
