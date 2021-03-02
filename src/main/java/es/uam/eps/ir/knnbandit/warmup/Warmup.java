/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.warmup;

import es.uam.eps.ir.knnbandit.utils.FastRating;
import java.util.List;

/**
 * Interface for storing the warm-up data.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface Warmup
{
    /**
     * Obtains the number of relevant ratings in the warm-up data.
     * @return the number of relevant ratings.
     */
    int getNumRel();
    /**
     * Gets the full list of training tuples.
     *
     * @return the full list of training tuples, null if the initializer has not been configured.
     */
    List<FastRating> getFullTraining();

    /**
     * Gets the list of training tuples without unknown ratings.
     *
     * @return the full list of training tuples, null if the initializer has not been configured.
     */
    List<FastRating> getCleanTraining();
}
