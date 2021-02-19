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
 * Identifiers of the algorithms that can be used.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class AlgorithmIdentifiers
{
    // Simple algorithms.
    public static final String RANDOM = "random";
    public static final String AVG = "average";
    public static final String POP = "popularity";
    // Non-personalized item-oriented bandits.
    public static final String ITEMBANDIT = "itembandit";
    // User based.
    public static final String USERBASEDKNN = "ub";
    public static final String UBBANDIT = "ub-bandit";
    // Item based
    public static final String ITEMBASEDKNN = "ib";
    public static final String IBBANDIT = "ib-bandit";

    // Matrix factorization.
    public static final String MF = "mf";
    // Bandits using probabilistic matrix factorization.
    public static final String PMFBANDIT = "inter-pmf";
    public static final String PTS = "pts";
    public static final String BAYESIANPTS = "bayesian-pts";
    // Bandits using knn
    public static final String COLLABGREEDY = "collab-greedy";
    // Cluster-based approaches
    public static final String CLUB = "club";
    public static final String CLUBERDOS = "club-erdos";
    public static final String COFIBA = "cofiba";

    // Wisdom of the diverse crowds:
    public static final String INFTHEOR = "inf-theory";
}