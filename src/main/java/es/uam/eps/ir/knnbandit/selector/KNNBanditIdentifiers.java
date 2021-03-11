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
 * Identifiers for the different variants of the MF and kNN algorithms.
 */
public class KNNBanditIdentifiers
{
    /**
     * Identifier for the basic variant (the first obtained rating for a user-item pair is the one that is left).
     */
    public final static String BASIC = "basic";
    /**
     * Identifier for the best variant (the best obtained rating for a user-item pair is the one that is left).
     */
    public final static String BEST = "best";
    /**
     * Identifier for the last variant (the last obtained rating for a user-item pair is the one that is left).
     */
    public final static String LAST = "last";
    /**
     * Identifier for the additive variant (the rating for a user-item pairs is the sum of all the obtained ones).
     */
    public final static String ADDITIVE = "additive";
}
