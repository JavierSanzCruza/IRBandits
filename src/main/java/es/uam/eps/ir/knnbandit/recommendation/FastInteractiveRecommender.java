/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation;

import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;

import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public interface FastInteractiveRecommender<U,I> extends InteractiveRecommender<U,I>
{
    /**
     * Initializes the specific variables of a method.
     */
    void init();

    /**
     * Initializes the specific variables of a method, using some information as training data.
     * @param values a stream of (user, item, value) triplets.
     */
    void init(Stream<FastRating> values);

    /**
     * Obtains the users.
     *
     * @return the users.
     */
    IntStream getUidx();


    /**
     * Obtains the items.
     *
     * @return the items.
     */
    IntStream getIidx();

    /**
     * Given a user, and a list of items, returns the next value.
     * @param uidx user.
     * @param available the list of identifiers of the candidate items.
     * @return the identifier of the recommended item if everything went OK, -1 otherwise i.e. when a user cannot be recommended an item)
     */
    int next(int uidx, IntList available);

    /**
     * Given a user, and the list of available items, returns a top-k recommendation (when possible).
     * If the algorithm can only recommend l < k items, but there are more available, those are
     * recommended randomly.
     *
     * @param uidx user identifier.
     * @param available the list of identifiers of the candidate items.
     * @param k the number of items to recommend.
     * @return a list of recommended items.
     */
    IntList next(int uidx, IntList available, int k);

    /**
     * Updates the method.
     *
     * @param uidx  User identifier.
     * @param iidx  Item identifier.
     * @param value The rating u provides to i.
     */
    void fastUpdate(int uidx, int iidx, double value);

    /**
     * Updates the method.
     *
     * @param train Training data.
     */
    void fastUpdate(List<Tuple3<Integer, Integer, Double>> train);
}
