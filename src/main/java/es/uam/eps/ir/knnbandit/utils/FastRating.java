/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.utils;

/**
 * Class that represents a fast rating.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class FastRating
{
    /**
     * The user.
     */
    private final int user;
    /**
     * The item.
     */
    private final int item;
    /**
     * The rating value.
     */
    private final double value;

    /**
     * Constructor.
     * @param user  the user identifier.
     * @param item  the item identifier.
     * @param value the rating value.
     */
    public FastRating(int user, int item, double value)
    {
        this.user = user;
        this.item = item;
        this.value = value;
    }

    /**
     * Obtains the user identifier.
     * @return the user identifier.
     */
    public int uidx()
    {
        return user;
    }

    /**
     * Obtains the item identifier.
     * @return the item identifier.
     */
    public int iidx()
    {
        return item;
    }

    /**
     * Obtains the rating value for the user-item pair.
     * @return the rating value for the user-item pair.
     */
    public double value()
    {
        return value;
    }

    @Override
    public boolean equals(Object obj)
    {
        if (obj == this) {
            return true;
        }
        if (obj == null || obj.getClass() != this.getClass()) {
            return false;
        }

        FastRating rating = (FastRating) obj;
        return this.user == rating.user && this.item == rating.item;
    }
}
