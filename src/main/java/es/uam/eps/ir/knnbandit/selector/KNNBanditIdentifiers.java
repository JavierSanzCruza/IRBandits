/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector;

/**
 * Identifier for the variants of the knn bandit.
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.knn.user.AbstractInteractiveUserBasedKNN
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class KNNBanditIdentifiers
{
    /**
     * Identifier for the basic version (takes the first value given to the user-item pair).
     */
    public final static String BASIC = "basic";
    /**
     * Identifier for the version that takes the best value given to the user-item pair.
     */
    public final static String BEST = "best";
    /**
     * Identifier for the version that takes the last value given to the user-item pair.
     */
    public final static String LAST = "last";
    /**
     * Identifier for the version that takes the sum of all values given to the user-item pair.
     */
    public final static String ADDITIVE = "additive";
}
