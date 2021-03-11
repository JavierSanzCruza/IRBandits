/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.data.datasets;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.utils.Pair;

import java.util.List;
import java.util.Optional;
import java.util.function.DoublePredicate;

/**
 * Interface for defining datasets.
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface Dataset<U,I> extends FastUpdateableUserIndex<U>, FastUpdateableItemIndex<I>
{
    /**
     * Obtains the threshold for determining the relevance of the ratings
     * @return the relevance threshold.
     */
    DoublePredicate getRelevanceChecker();

    /**
     * Obtains the rating value that a user has given to an item.
     * @param u the user.
     * @param i the item.
     * @return the rating.
     */
    Optional<Double> getPreference(U u, I i);

    /**
     * Obtains the rating value that a user has given to an item.
     * @param uidx the user identifier.
     * @param iidx the item identifier.
     * @return the rating.
     */
    Optional<Double> getPreference(int uidx, int iidx);

    /**
     * Obtains the number of relevant ratings in the dataset.
     * If there is no information, 0 is returned.
     * @return the number of relevant ratings if information is available, 0 otherwise.
     */
    int getNumRel();

    /**
     * Obtains the total number of ratings in the dataset.
     * If there is no information, 0 is returned.
     * @return the number of total ratings if information is available, 0 otherwise.
     */
    int getNumRatings();

    /**
     * Subsamples a dataset.
     * @param pairs the list of pairs to keep.
     * @return the reduced dataset.
     */
    Dataset<U,I> load(List<Pair<Integer>> pairs);
}
