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

import it.unimi.dsi.fastutil.ints.IntList;

import java.util.List;

/**
 * Interface for defining the warm-up data for off-line datasets.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface OfflineWarmup extends Warmup
{
    /**
     * Gets the availability lists.
     *
     * @return the availability lists, null if the Initializer has not been configured.
     */
    List<IntList> getAvailability();
}
