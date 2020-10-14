/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.data.preference.updateable.fast;

import es.uam.eps.ir.knnbandit.data.preference.updateable.UpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;

/**
 * Interface for updateable preference data.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface FastUpdateablePreferenceData<U, I> extends UpdateablePreferenceData<U, I>, FastPreferenceData<U, I>, FastUpdateableUserIndex<U>, FastUpdateableItemIndex<I>
{
    /**
     * Updates a rating value.
     *
     * @param uidx   Identifier of the user.
     * @param iidx   Identifier of the item.
     * @param rating The rating.
     */
    void updateRating(int uidx, int iidx, double rating);

    /**
     * Deletes a rating.
     *
     * @param uidx Identifier of the user.
     * @param iidx Identifier of the item.
     */
    void updateDelete(int uidx, int iidx);
}
