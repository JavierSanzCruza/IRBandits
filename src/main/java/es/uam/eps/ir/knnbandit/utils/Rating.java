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
 * Class for representing a rating.
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class Rating<U,I>
{
    /**
     * The user.
     */
    private final U user;
    /**
     * The item.
     */
    private final I item;
    /**
     * The value for the user-item pair.
     */
    private final double value;

    /**
     * Constructor.
     * @param user  the user.
     * @param item  the item.
     * @param value the value of the rating for the user-item pair.
     */
    public Rating(U user, I item, double value)
    {
        this.user = user;
        this.item = item;
        this.value = value;
    }

    /**
     * Obtains the user.
     * @return the user.
     */
    public U getUser()
    {
        return user;
    }

    /**
     * Obtains the item.
     * @return the item.
     */
    public I getItem()
    {
        return item;
    }

    /**
     * Obtains the rating value.
     * @return the rating value.
     */
    public double getValue()
    {
        return value;
    }
}