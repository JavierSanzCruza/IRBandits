/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector;

/**
 * Identifiers of the different Epsilon-Greedy update functions.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class EpsilonGreedyUpdateFunctionIdentifiers
{
    /**
     * Identifier for the stationary update.
     */
    public static final String STATIONARY = "stationary";
    /**
     * Identifier for the non-stationary version, which favors the last values.
     */
    public static final String NONSTATIONARY = "nonStationary";

    /**
     * Identifier for the version that computes the value of the arm considering the rewards for the rest of the arms.
     */
    public static final String USEALL = "useAll";

    /**
     * Instead of using average value, it uses a simple count.
     */
    public static final String COUNT = "count";
}
