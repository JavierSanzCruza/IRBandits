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

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import it.unimi.dsi.fastutil.doubles.DoubleIterator;
import it.unimi.dsi.fastutil.ints.IntIterator;
import org.ranksys.core.util.iterators.StreamDoubleIterator;
import org.ranksys.core.util.iterators.StreamIntIterator;

import java.util.function.Function;

/**
 * Extends AbstractFastUpdateablePreferenceData and implements the data access iterator-based methods
 * using the stream-based ones. Avoids duplicating code where stream-based methods
 * are preferred.
 *
 * @param <U> User type.
 * @param <I> Item type.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public abstract class StreamsAbstractFastUpdateablePreferenceData<U, I> extends AbstractFastUpdateablePreferenceData<U, I>
{

    /**
     * Constructor with default IdxPref to IdPref converter.
     *
     * @param userIndex User index.
     * @param itemIndex Item index.
     */
    public StreamsAbstractFastUpdateablePreferenceData(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
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
    public StreamsAbstractFastUpdateablePreferenceData(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, Function<IdxPref, IdPref<I>> uPrefFun, Function<IdxPref, IdPref<U>> iPrefFun)
    {
        super(userIndex, itemIndex, uPrefFun, iPrefFun);
    }

    @Override
    public IntIterator getUidxIidxs(int uidx)
    {
        return new StreamIntIterator(getUidxPreferences(uidx).mapToInt(IdxPref::v1));
    }

    @Override
    public DoubleIterator getUidxVs(int uidx)
    {
        return new StreamDoubleIterator(getUidxPreferences(uidx).mapToDouble(IdxPref::v2));
    }

    @Override
    public IntIterator getIidxUidxs(int iidx)
    {
        return new StreamIntIterator(getIidxPreferences(iidx).mapToInt(IdxPref::v1));
    }

    @Override
    public DoubleIterator getIidxVs(int iidx)
    {
        return new StreamDoubleIterator(getIidxPreferences(iidx).mapToDouble(IdxPref::v2));
    }

    @Override
    public boolean useIteratorsPreferentially()
    {
        return false;
    }
}
