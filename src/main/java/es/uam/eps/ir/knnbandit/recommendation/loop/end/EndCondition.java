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
 * Interface for the classes that check whether a recommendation loop has finished or not.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface EndCondition
{
    /**
     * Initializes the condition.
     */
    void init();

    /**
     * Checks whether the loop has ended or not.
     *
     * @return true if the end condition has been met, false otherwise.
     */
    boolean hasEnded();

    /**
     * Updates the condition
     *
     * @param uidx  last recommended user
     * @param iidx  last recommended item
     * @param value the value.
     */
    void update(int uidx, int iidx, double value);
}
