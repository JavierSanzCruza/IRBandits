/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.data.preference.updateable;

import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;

import java.util.stream.Stream;

/**
 * Preference data that allows updating over time.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface Updateable<U, I>
{
    /**
     * Updates the preference data given a set of preferences.
     * It does not add new users/items. Tuples with non-existing
     * users/items will be ignored.
     *
     * @param tuples The tuples.
     */
    void update(Stream<Tuple3<U, I, Double>> tuples);

    /**
     * Updates an individual preference.
     *
     * @param u   User.
     * @param i   Item.
     * @param val Preference value.
     *
     * @return true if the value for the (u,i) pair has changed, false otherwise.
     */
    boolean update(U u, I i, double val);

    /**
     * Updates the preference data given a set of preferences to delete.
     *
     * @param tuples The tuples.
     */
    void updateDelete(Stream<Tuple2<U, I>> tuples);

    /**
     * Deletes an individual preference.
     *
     * @param u User.
     * @param i Item.
     */
    void updateDelete(U u, I i);

    /**
     * Adds a user.
     *
     * @param u User.
     */
    void updateAddUser(U u);

    /**
     * Adds an item.
     *
     * @param i Item.
     */
    void updateAddItem(I i);
}
