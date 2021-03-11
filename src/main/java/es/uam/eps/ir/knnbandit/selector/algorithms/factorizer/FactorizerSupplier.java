/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms.factorizer;

import es.uam.eps.ir.ranksys.mf.Factorizer;

/**
 * Interface for obtaining multiple (already configured) factorizers.
 *
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface FactorizerSupplier<U,I>
{
    /**
     * Gets the factorizer.
     * @return the configured factorizer.
     */
    Factorizer<U,I> apply();

    /**
     * Gets the name of the factorizer.
     * @return the name of the factorizer.
     */
    String getName();
}
