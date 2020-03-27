/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.data.preference.userknowledge.fast;


import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import org.ranksys.fast.preference.FastPointWisePreferenceData;

import java.util.Optional;

/**
 * Fast updateable version of a pointwise preference data.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface FastUserKnowledgePointWisePreferenceData<U, I> extends FastPointWisePreferenceData<U, I>, FastPreferenceData<U, I>
{
    /**
     * Gets a preference if exists and the user knew about the item before submitting the rating.
     *
     * @param uidx the identifier of the user.
     * @param iidx the identifier of the item.
     * @return the rating if exists.
     */
    Optional<IdxPref> getKnownPreference(int uidx, int iidx);

    /**
     * Gets a preference if exists and the user did not know about the item before submitting the rating.
     *
     * @param uidx the identifier of the user.
     * @param iidx the identifier of the item.
     * @return the rating if exists.
     */
    Optional<IdxPref> getUnknownPreference(int uidx, int iidx);

    /**
     * Gets a preference if exists and the user knew about the item before submitting the rating.
     *
     * @param u the user.
     * @param i the item.
     * @return the rating if exists.
     */
    Optional<? extends IdPref<I>> getKnownPreference(U u, I i);

    /**
     * Gets a preference if exists and the user did not know about the item before submitting the rating.
     *
     * @param u the user.
     * @param i the item.
     * @return the rating if exists.
     */
    Optional<? extends IdPref<I>> getUnknownPreference(U u, I i);


}
