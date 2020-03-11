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
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import it.unimi.dsi.fastutil.doubles.DoubleIterator;
import it.unimi.dsi.fastutil.ints.IntIterator;

import java.util.function.Function;
import java.util.stream.Stream;

import static java.util.stream.IntStream.range;

/**
 * Extends AbstractFastUpdateablePreferenceData and implements the data access stream-based methods using the iterator-based ones. Avoids duplicating code where iterator-based methods are preferred.
 *
 * @param <U> User type.
 * @param <I> Item type.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public abstract class IteratorsAbstractFastUserKnowledgePreferenceData<U, I> extends AbstractFastUserKnowledgePreferenceData<U, I>
{
    /**
     * Constructor with default IdxPref to IdPref converter.
     *
     * @param userIndex User index.
     * @param itemIndex Item index.
     */
    public IteratorsAbstractFastUserKnowledgePreferenceData(FastUserIndex<U> userIndex, FastItemIndex<I> itemIndex)
    {
        super(userIndex, itemIndex);
    }

    /**
     * Constructor with custom IdxPref to IdPref converter.
     *
     * @param userIndex User index.
     * @param itemIndex Item index.
     * @param uPrefFun  User IdxPref to IdPref converter.
     * @param iPrefFun  Item IdxPref to IdPref converter.
     */
    public IteratorsAbstractFastUserKnowledgePreferenceData(FastUserIndex<U> userIndex, FastItemIndex<I> itemIndex, Function<IdxPref, IdPref<I>> uPrefFun, Function<IdxPref, IdPref<U>> iPrefFun)
    {
        super(userIndex, itemIndex, uPrefFun, iPrefFun);
    }

    @Override
    public Stream<? extends IdxPref> getUidxKnownPreferences(int uidx)
    {
        return getPreferences(numKnownItems(uidx), getUidxKnownIidxs(uidx), getUidxKnownVs(uidx));
    }

    @Override
    public Stream<? extends IdxPref> getUidxUnknownPreferences(int uidx)
    {
        return getPreferences(numUnknownItems(uidx), getUidxUnknownIidxs(uidx), getUidxUnknownVs(uidx));
    }

    @Override
    public Stream<? extends IdxPref> getIidxKnownPreferences(int iidx)
    {
        return getPreferences(numKnownUsers(iidx), getIidxKnownUidxs(iidx), getIidxKnownVs(iidx));
    }

    @Override
    public Stream<? extends IdxPref> getIidxUnknownPreferences(int iidx)
    {
        return getPreferences(numUnknownUsers(iidx), getIidxUnknownUidxs(iidx), getIidxUnknownVs(iidx));
    }

    @Override
    public Stream<? extends IdxPref> getUidxPreferences(int uidx)
    {
        return getPreferences(numItems(uidx), getUidxIidxs(uidx), getUidxVs(uidx));
    }

    @Override
    public Stream<? extends IdxPref> getIidxPreferences(int iidx)
    {
        return getPreferences(numUsers(iidx), getIidxUidxs(iidx), getIidxVs(iidx));
    }

    /**
     * Converts the int and double iterators to a stream of IdxPref.
     *
     * @param n    Length of iterators.
     * @param idxs Iterator of user/item indices.
     * @param vs   Iterator of user/item values.
     *
     * @return Stream of IdxPref.
     */
    protected Stream<IdxPref> getPreferences(int n, IntIterator idxs, DoubleIterator vs)
    {
        return range(0, n).mapToObj(i -> new IdxPref(idxs.nextInt(), vs.nextDouble()));
    }

    @Override
    public boolean useIteratorsPreferentially()
    {
        return true;
    }

}
