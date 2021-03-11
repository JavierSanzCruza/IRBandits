/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

/**
 * Bandit identifiers.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class MultiArmedBanditIdentifiers
{
    /**
     * Identifier for epsilon greedy
     */
    public static final String EGREEDY = "epsilon";
    /**
     * Identifier for UCB1
     */
    public static final String UCB1 = "ucb1";
    /**
     * Identifier for UCB1-tuned
     */
    public static final String UCB1TUNED = "ucb1tuned";
    /**
     * Identifier for Thompson sampling
     */
    public static final String THOMPSON = "thompson";
    /**
     * Identifier for the delayed Thompson sampling.
     */
    public static final String DELAYTHOMPSON = "delaythompson";
    /**
     * Identifier for epsilon-t greedy
     */
    public static final String ETGREEDY = "epsilont";
    /**
     * Identifier for the algorithm that selects arms according to their popularity.
     */
    public static final String MLEPOP = "mlepop";
    /**
     * Identifier for the algorithm that selects arms according to their average estimated value.
     */
    public static final String MLEAVG = "mleavg";
}
