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

import es.uam.eps.ir.knnbandit.data.preference.userknowledge.UserKnowledgePreferenceData;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import it.unimi.dsi.fastutil.doubles.DoubleIterator;
import it.unimi.dsi.fastutil.ints.IntIterator;

import java.util.stream.Stream;

/**
 * Interface for fast preference data with user knowledge.
 *
 * @param <U> User type.
 * @param <I> Item type.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface FastUserKnowledgePreferenceData<U, I> extends UserKnowledgePreferenceData<U, I>, FastPreferenceData<U, I>, FastUserIndex<U>, FastItemIndex<I>
{
    /**
     * Obtains the number of ratings for an item which were previously known by users.
     * @param iidx the item.
     * @return the number of ratings for item i which were previously known by users.
     */
    int numKnownUsers(int iidx);

    /**
     * Obtains the number of ratings for an item which were not previously known by users.
     * @param iidx the item.
     * @return the number of ratings for item i which were not previously known by users.
     */
    default int numUnknownUsers(int iidx)
    {
        return this.numUsers(iidx) - this.numKnownUsers(iidx);
    }

    /**
     * Obtains the number rated items previously known by a user.
     * @param uidx the user.
     * @return the number rated items previously known by u.
     */
    int numKnownItems(int uidx);

    /**
     * Obtains the number rated items not previously known by a user.
     * @param uidx the user.
     * @return the number rated items not previously known by u.
     */
    default int numUnknownItems(int uidx)
    {
        return this.numItems(uidx) - this.numKnownItems(uidx);
    }

    Stream<? extends IdxPref> getUidxKnownPreferences(int uidx);
    Stream<? extends IdxPref> getIidxKnownPreferences(int iidx);
    Stream<? extends IdxPref> getUidxUnknownPreferences(int uidx);
    Stream<? extends IdxPref> getIidxUnknownPreferences(int iidx);

    public DoubleIterator getUidxKnownVs(int uidx);
    public IntIterator getUidxKnownIidxs(int uidx);
    public DoubleIterator getUidxUnknownVs(int uidx);
    public IntIterator getUidxUnknownIidxs(int uidx);
    public DoubleIterator getIidxKnownVs(int iidx);
    public IntIterator getIidxKnownUidxs(int iidx);
    public DoubleIterator getIidxUnknownVs(int iidx);
    public IntIterator getIidxUnknownUidxs(int iidx);

}
