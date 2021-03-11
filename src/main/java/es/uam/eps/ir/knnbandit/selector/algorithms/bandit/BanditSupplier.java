/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.AbstractMultiArmedBandit;

/**
 * Interface for obtaining multiple copies of a previously configured multi-armed bandit algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface BanditSupplier
{
    /**
     * Obtains the configured multi-armed bandit.
     * @param numArms the number of arms in the bandit.
     * @return the multi-armed bandit.
     */
    AbstractMultiArmedBandit apply(int numArms);

    /**
     * Obtains the name of the bandit.
     * @return the name of the bandit.
     */
    String getName();
}
