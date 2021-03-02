/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation;

/**
 * Enumeration for defining the type of uses we can make of the data with knowledge.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public enum KnowledgeDataUse
{
    ONLYKNOWN, ONLYUNKNOWN, ALL;

    /**
     * Obtains the KnowledgeDataUse value from a string.
     *
     * @param string known for ONLYKNOWN, unknown for ONLYUNKNOWN, all else for ALL.
     * @return the KnowledgeDataUse value (ALL by default).
     */
    public static KnowledgeDataUse fromString(String string)
    {
        String aux = string.toLowerCase();
        switch (aux)
        {
            case "known":
                return ONLYKNOWN;
            case "unknown":
                return ONLYUNKNOWN;
            case "all":
            default:
                return ALL;
        }
    }
}
