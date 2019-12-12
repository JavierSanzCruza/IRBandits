/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.utils;

/**
 * Class that represents a pair of objects of the same type.
 *
 * @param <U> Type of the objects.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class Pair<U> extends Tuple2oo<U, U>
{
    /**
     * Constructor
     *
     * @param first  First element.
     * @param second Second element.
     */
    public Pair(U first, U second)
    {
        super(first, second);
    }
}
