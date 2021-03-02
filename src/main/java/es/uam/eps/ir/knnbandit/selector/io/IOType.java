/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.selector.io;

/**
 * Auxiliar class for selecting the type of input/output.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public enum IOType
{
    BINARY, TEXT, ERROR;

    /**
     * Obtains the type of input from string.
     * @param str the string.
     * @return the type. If an error occurs, the ERROR value is returned.
     */
    public static IOType fromString(String str)
    {
        switch (str.toLowerCase())
        {
            case "binary": // for binary files.
                return BINARY;
            case "text": // for text files.
                return TEXT;
            default:
                return ERROR;
        }
    }
}