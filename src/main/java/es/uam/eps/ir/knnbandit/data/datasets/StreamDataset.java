/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.data.datasets;

import it.unimi.dsi.fastutil.ints.IntList;

import java.io.IOException;
import java.util.List;

/**
 * Dataset represented as a stream of logged data, advancing over time.
 * It is considered that, at a given time, a single item is rated, selected
 * from a set of them.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface StreamDataset<U,I> extends Dataset<U, I>
{
    /**
     * Restarts the stream, reading information from the beginning.
     */
    void restart() throws IOException;

    /**
     * Advances the stream.
     */
    void advance() throws IOException;

    /**
     * Checks whether the stream has ended or not.
     * @return true if it has ended, false otherwise.
     */
    boolean hasEnded();

    /**
     * Get the current target user.
     * @return the target user
     */
    U getCurrentUser();

    /**
     * Get the current list of candidate items
     * @return the current list of candidate items.
     */
    List<I> getCandidateItems();

    /**
     * Get the featured item (the item containing the read rating)
     * @return the featured item.
     */
    I getFeaturedItem();

    /**
     * Get the rating given by the current target user to the featured item.
     * @return the rating the user has given to the featured item.
     */
    double getFeaturedItemRating();

    /**
     * Get the identifier of the target user.
     * @return the identifier of the current target user.
     */
    int getCurrentUidx();

    /**
     * Get the identifiers of the candidate items.
     * @return the identifier of the current candidate items.
     */
    IntList getCandidateIidx();

    /**
     * Get the identifier of the featured item.
     * @return the identifier of the featured item.
     */
    int getFeaturedIidx();
}
