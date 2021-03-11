/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms.factorizer;

/**
 * Identifier for matrix factorization factorizers.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class FactorizerIdentifiers
{
    /**
     * Implicit matrix factorization -- original algorithm.
     */
    public final static String IMF = "imf";
    /**
     * Implicit matrix factorization -- fast implementation.
     */
    public final static String FASTIMF = "fastimf";
    /**
     * Probabilistic Latent Semantic Analysis.
     */
    public final static String PLSA = "plsa";
}
