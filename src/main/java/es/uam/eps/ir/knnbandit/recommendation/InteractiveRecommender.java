/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation;

import es.uam.eps.ir.knnbandit.utils.Rating;
import org.jooq.lambda.tuple.Tuple3;

import java.util.List;
import java.util.stream.Stream;

public interface InteractiveRecommender<U,I>
{
    /**
     * Initializes the specific variables of a method.
     */
    void init();

    /**
     * Initializes the specific variables of a method, using some information as training data.
     * @param values a stream of (user, item, value) triplets.
     */
    void initialize(Stream<Rating<U,I>> values);

    /**
     * Obtains the users.
     *
     * @return the users.
     */
    Stream<U> getUsers();


    /**
     * Obtains the items.
     *
     * @return the items.
     */
    Stream<I> getItems();

    /**
     * Obtains the number of users.
     *
     * @return the number of users.
     */
    int numUsers();

    /**
     * Obtains the number of items.
     *
     * @return the number of items.
     */
    int numItems();

    /**
     * Given a user, and a list of items, returns the next value.
     * @param u user.
     * @param available the list of identifiers of the candidate items.
     * @return the identifier of the recommended item if everything went OK, -1 otherwise i.e. when a user cannot be recommended an item)
     */
    I next(U u, List<I> available);

    /**
     * Given a user, and the list of available items, returns a top-k recommendation (when possible).
     * If the algorithm can only recommend l < k items, but there are more available, those are
     * recommended randomly.
     *
     * @param u user identifier.
     * @param available the list of identifiers of the candidate items.
     * @param k the number of items to recommend.
     * @return a list of recommended items.
     */
    List<I> next(U u, List<I> available, int k);

    /**
     * Updates the method.
     *
     * @param u  User.
     * @param i  Item.
     * @param value The rating u provides to i.
     */
    void update(U u, I i, double value);

    /**
     * Updates the method.
     *
     * @param train Training data.
     */
    void update(List<Tuple3<U, I, Double>> train);

    /**
     * Checks if the recommender uses all the received information, or only known data.
     *
     * @return true if the recommender uses all the received information, false otherwise.
     */
    boolean usesAll();
}
