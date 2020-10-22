/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.update;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.Selection;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.Warmup;

import java.util.List;

/**
 * Strategy for selecting the (user,item,rating) triplets which shall be used
 * for updating the recommendation loop.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface UpdateStrategy<U,I>
{
    /**
     * Initializes the update selection mechanism.
     * @param dataset the dataset.
     */
    void init(Dataset<U,I> dataset);

    /**
     * Selects the (user,item,rating) triplets which shall be applied for updating
     * the recommendation loop.
     * @param uidx the identifier of the user.
     * @param iidx the identifier of the item.
     * @param selection a selection mechanism for checking the availability of the (uidx,iidx) pairs.
     * @return a pair containing two lists: the first list contains the ratings for updating
     * the interactive recommender and the selection mechanism, whereas the second list is used
     * for updating the metric values.
     */
    Pair<List<FastRating>> selectUpdate(int uidx, int iidx, Selection<U,I> selection);

    /**
     * Given a warmup, gets the whole rating list.
     * @param warmup the warmup.
     * @return a list containing the ratings.
     */
    List<FastRating> getList(Warmup warmup);
}
