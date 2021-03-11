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
 * Identifier for the different variants of the interactive collaborative filtering algorithm
 * based on probabilistic matrix factorization.
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.mf.icf.InteractivePMFRecommender
 */
public class PMFBanditIdentifiers
{
    /**
     * Identifier for the epsilon-greedy variant.
     */
    public static final String EGREEDY = "epsilon";
    /**
     * Identifier for the UCB variant.
     */
    public static final String UCB = "ucb";
    /**
     * Identifier for the generalized linear model - UCB variant.
     */
    public static final String GENERALIZEDUCB = "glmucb";
    /**
     * Identifier for the Thompson sampling variant.
     */
    public static final String THOMPSON = "thompson";
}
