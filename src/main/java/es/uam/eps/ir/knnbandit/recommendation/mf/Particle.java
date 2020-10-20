/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf;

/**
 * Individual particle for reinforcement learning algorithms.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface Particle<U, I>
{
    /**
     * Initializes the particle.
     */
    void initialize();

    /**
     * Updates the particle.
     *
     * @param u     the user.
     * @param i     the item.
     * @param value the value of the interaction between user and item.
     */
    void update(U u, I i, double value);

    /**
     * Updates the particle.
     *
     * @param uidx  the index of the user.
     * @param iidx  the index of the item.
     * @param value the value of the interaction between user and item.
     */
    void update(int uidx, int iidx, double value);

    /**
     * Obtains the estimated value of the interaction between user and item.
     *
     * @param u the user.
     * @param i the item.
     * @return the estimated reward
     */
    double getEstimatedReward(U u, I i);


    double getEstimatedReward(int uidx, int iidx);

    /**
     * Obtains the weight of the particle.
     *
     * @param u     the user.
     * @param i     the item.
     * @param value the value of the interaction between user and item.
     * @return the weight of the particle.
     */
    double getWeight(U u, I i, double value);

    double getWeight(int uidx, int iidx, double value);

    Particle<U, I> clone();
}
