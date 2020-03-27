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
import es.uam.eps.ir.ranksys.fast.preference.AbstractFastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;

import java.io.Serializable;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * Abstract updateable fast preference data, implementing the FastUpdateablePreferenceData interface.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 * @author Saúl Vargas (saul.vargas@uam.es)
 */
public abstract class AbstractFastUserKnowledgePreferenceData<U, I> extends AbstractFastPreferenceData<U, I> implements FastUserKnowledgePreferenceData<U, I>
{
    /**
     * Constructor.
     *
     * @param users User index.
     * @param items Item index.
     */
    public AbstractFastUserKnowledgePreferenceData(FastUserIndex<U> users, FastItemIndex<I> items)
    {
        this(users, items,
             (Function<IdxPref, IdPref<I>> & Serializable) p -> new IdPref<>(items.iidx2item(p)),
             (Function<IdxPref, IdPref<U>> & Serializable) p -> new IdPref<>(users.uidx2user(p)));
    }

    /**
     * Constructor.
     *
     * @param userIndex User index.
     * @param itemIndex Item index.
     * @param uPrefFun  Converter from IdxPref to IdPref (preference for item).
     * @param iPrefFun  Converter from IdxPref to IdPref (preference from user).
     */
    public AbstractFastUserKnowledgePreferenceData(FastUserIndex<U> userIndex, FastItemIndex<I> itemIndex, Function<IdxPref, IdPref<I>> uPrefFun, Function<IdxPref, IdPref<U>> iPrefFun)
    {
        super(userIndex, itemIndex, uPrefFun, iPrefFun);
    }

    @Override
    public Stream<? extends IdPref<I>> getUserKnownPreferences(U u)
    {
        return this.getUidxKnownPreferences(this.user2uidx(u)).map(uPrefFun);
    }

    @Override
    public Stream<? extends IdPref<I>> getUserUnknownPreferences(U u)
    {
        return this.getUidxUnknownPreferences(this.user2uidx(u)).map(uPrefFun);
    }

    @Override
    public Stream<? extends IdPref<U>> getItemKnownPreferences(I i)
    {
        return this.getIidxKnownPreferences(this.item2iidx(i)).map(iPrefFun);
    }

    @Override
    public Stream<? extends IdPref<U>> getItemUnknownPreferences(I i)
    {
        return this.getIidxUnknownPreferences(this.item2iidx(i)).map(iPrefFun);
    }

    @Override
    public int numKnownUsers(I i)
    {
        return this.numKnownUsers(this.item2iidx(i));
    }

    @Override
    public int numUnknownUsers(I i)
    {
        int iidx = this.item2iidx(i);
        return this.numUsers(iidx) - this.numKnownUsers(iidx);
    }

    @Override
    public int numKnownItems(U u)
    {
        return this.numKnownUsers(this.user2uidx(u));
    }

    @Override
    public int numUnknownItems(U u)
    {
        int uidx = this.user2uidx(u);
        return this.numUsers(uidx) - this.numKnownUsers(uidx);
    }
}
