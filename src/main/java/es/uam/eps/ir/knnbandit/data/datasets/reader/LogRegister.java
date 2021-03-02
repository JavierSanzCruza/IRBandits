/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.data.datasets.reader;

import java.util.Collection;

/**
 * Class that represents a single register in a log-based dataset.
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class LogRegister<U,I>
{
    /**
     * The user.
     */
    private final U user;
    /**
     * The featured item.
     */
    private final I featuredItem;
    /**
     * The rating the user gives to the featured item.
     */
    private final double rating;
    /**
     * The whole set of candidate items.
     */
    private final Collection<I> candidateItems;

    /**
     * Constructor.
     * @param user              the user.
     * @param featuredItem      the featured item.
     * @param rating            the rating for the user - featured item pair.
     * @param candidateItems    the whole set of candidate items.
     */
    public LogRegister(U user, I featuredItem, double rating, Collection<I> candidateItems)
    {
        this.user = user;
        this.featuredItem = featuredItem;
        this.rating = rating;
        this.candidateItems = candidateItems;
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
     * Obtains the featured item.
     * @return the featured item.
     */
    public I getFeaturedItem()
    {
        return featuredItem;
    }

    /**
     * Obtains the rating for the user - featured item pair.
     * @return the rating for the user - featured item pair.
     */
    public double getRating()
    {
        return rating;
    }

    /**
     * Obtains the set of candidate items.
     * @return the set of candidate items.
     */
    public Collection<I> getCandidateItems()
    {
        return candidateItems;
    }
}
