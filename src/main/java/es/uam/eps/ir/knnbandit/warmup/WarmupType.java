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

/**
 * Enumeration for the type of warmup.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public enum WarmupType
{
    FULL, ONLYRATINGS;

    /**
     * Obtains the warm-up type.
     *
     * @param type the warm-up type selection.
     * @return the type if everything is OK, null otherwise.
     */
    public static WarmupType fromString(String type)
    {
        switch (type.toLowerCase())
        {
            case "onlyratings":
                return ONLYRATINGS;
            case "full":
                return FULL;
            default:
                return null;
        }
    }
}
