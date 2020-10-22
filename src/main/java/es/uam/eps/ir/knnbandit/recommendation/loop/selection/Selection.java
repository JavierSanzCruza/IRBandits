/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.selection;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import it.unimi.dsi.fastutil.ints.IntList;

/**
 * Interface for classes that select the target and candidate items for the interactive recommendation
 * dataset.
 *
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface Selection<U,I>
{
    /**
     * Selects the next target user of the recommendation.
     * @return the target user of the recommendation, -1 if no one can be selected.
     */
    int selectTarget();

    /**
     * Selects a list of candidate items for the recommendation.
     * @param uidx the index of the target user
     * @return the list of candidate items for the recommendation, null if there is not any.
     */
    IntList selectCandidates(int uidx);

    /**
     * Updates the selection strategy.
     * @param uidx user index.
     * @param iidx item index.
     * @param value payoff of the user/index term.
     */
    void update(int uidx, int iidx, double value);

    /**
     * Initializes the selector.
     * @param dataset the dataset containing information.
     */
    void init(Dataset<U,I> dataset);

    /**
     * Initializes the selector after some warmup data has been processed.
     * @param dataset the dataset.
     * @param warmup the warm-up
     */
    void init(Dataset<U,I> dataset, Warmup warmup);

    /**
     * Indicates whether an item can be recommended to a user in future times.
     * @param uidx the identifier of the user.
     * @param iidx the identifier of the item.
     * @return true if iidx can be recommended to uidx, false otherwise
     */
    boolean isAvailable(int uidx, int iidx);


}
