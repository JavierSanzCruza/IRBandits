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
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.TransposedPreferenceData;
import it.unimi.dsi.fastutil.doubles.DoubleIterator;
import it.unimi.dsi.fastutil.ints.IntIterator;
import org.jooq.lambda.function.Function2;
import org.ranksys.fast.preference.FastPointWisePreferenceData;

import java.util.stream.Stream;

/**
 * Updateable transposed preferences, where users and items change roles. This class is useful to simplify the implementation of many algorithms that work user or item-wise, such as similarities or matrix factorization.
 *
 * @param <U> User type.
 * @param <I> Item type.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 * @author Saúl Vargas (saul.vargas@uam.es)
 */
public class TransposedUserKnowledgePreferenceData<I, U> extends TransposedPreferenceData<I,U> implements FastUserKnowledgePreferenceData<I,U>,FastPointWisePreferenceData<I,U>
{
    protected FastUserKnowledgePreferenceData<U,I> recommenderData;

    /**
     * Constructor with default converters between IdxPref and IdPref.
     *
     * @param recommenderData Preference data to be transposed.
     */
    public TransposedUserKnowledgePreferenceData(FastUserKnowledgePreferenceData<U, I> recommenderData)
    {
        this(recommenderData, (u, p) -> new IdPref<>(u, p.v2), (uidx, p) -> new IdxPref(uidx, p.v2));
    }

    /**
     * Constructor with custom converters between IdxPref and IdPref.
     *
     * @param recommenderData Preference data to be transposed.
     * @param idPrefFun       Converter from item IdPref to user IdPref.
     * @param idxPrefFun      Cnverter from item IdxPref to item IdxPref.
     */
    public TransposedUserKnowledgePreferenceData(FastUserKnowledgePreferenceData<U, I> recommenderData,
                                                 Function2<U, IdPref<I>, IdPref<U>> idPrefFun,
                                                 Function2<Integer, IdxPref, IdxPref> idxPrefFun)
    {
        super(recommenderData, idPrefFun, idxPrefFun);
        this.recommenderData = recommenderData;
    }


    @Override
    public int numKnownUsers(int iidx)
    {
        return this.recommenderData.numKnownItems(iidx);
    }

    @Override
    public int numKnownItems(int uidx)
    {
        return this.recommenderData.numKnownUsers(uidx);
    }

    @Override
    public Stream<? extends IdxPref> getUidxKnownPreferences(int uidx)
    {
        return this.recommenderData.getIidxKnownPreferences(uidx);
    }

    @Override
    public Stream<? extends IdxPref> getIidxKnownPreferences(int iidx)
    {
        return this.recommenderData.getUidxKnownPreferences(iidx);
    }

    @Override
    public Stream<? extends IdxPref> getUidxUnknownPreferences(int uidx)
    {
        return this.recommenderData.getIidxUnknownPreferences(uidx);
    }

    @Override
    public Stream<? extends IdxPref> getIidxUnknownPreferences(int iidx)
    {
        return this.recommenderData.getUidxUnknownPreferences(iidx);
    }

    @Override
    public DoubleIterator getUidxKnownVs(int uidx)
    {
        return this.recommenderData.getIidxKnownVs(uidx);
    }

    @Override
    public IntIterator getUidxKnownIidxs(int uidx)
    {
        return this.recommenderData.getIidxKnownUidxs(uidx);
    }

    @Override
    public DoubleIterator getUidxUnknownVs(int uidx)
    {
        return this.recommenderData.getIidxUnknownVs(uidx);
    }

    @Override
    public IntIterator getUidxUnknownIidxs(int uidx)
    {
        return this.recommenderData.getIidxUnknownUidxs(uidx);
    }

    @Override
    public DoubleIterator getIidxKnownVs(int iidx)
    {
        return this.recommenderData.getUidxKnownVs(iidx);
    }

    @Override
    public IntIterator getIidxKnownUidxs(int iidx)
    {
        return this.recommenderData.getUidxKnownIidxs(iidx);
    }

    @Override
    public DoubleIterator getIidxUnknownVs(int iidx)
    {
        return this.recommenderData.getUidxUnknownVs(iidx);
    }

    @Override
    public IntIterator getIidxUnknownUidxs(int iidx)
    {
        return this.recommenderData.getUidxUnknownIidxs(iidx);
    }

    @Override
    public int numKnown()
    {
        return this.recommenderData.numKnown();
    }

    @Override
    public int numKnownUsers(U u)
    {
        return this.recommenderData.numKnownItems(u);
    }

    @Override
    public int numKnownItems(I i)
    {
        return this.recommenderData.numKnownUsers(i);
    }

    @Override
    public Stream<? extends IdPref<U>> getUserKnownPreferences(I i)
    {
        return this.recommenderData.getItemKnownPreferences(i);
    }

    @Override
    public Stream<? extends IdPref<U>> getUserUnknownPreferences(I i)
    {
        return this.recommenderData.getItemUnknownPreferences(i);
    }

    @Override
    public Stream<? extends IdPref<I>> getItemKnownPreferences(U u)
    {
        return this.recommenderData.getUserKnownPreferences(u);
    }

    @Override
    public Stream<? extends IdPref<I>> getItemUnknownPreferences(U u)
    {
        return this.recommenderData.getUserUnknownPreferences(u);
    }

    @Override
    public PreferenceData<I, U> getKnownPreferenceData()
    {
        return new TransposedPreferenceData<>((FastPreferenceData<U,I>) this.recommenderData.getKnownPreferenceData());
    }

    @Override
    public PreferenceData<I, U> getUnknownPreferenceData()
    {
        return new TransposedPreferenceData<>((FastPreferenceData<U,I>) this.recommenderData.getUnknownPreferenceData());
    }

    @Override
    public PreferenceData<I, U> getPreferenceData()
    {
        return new TransposedPreferenceData<>((FastPreferenceData<U,I>) this.recommenderData.getPreferenceData());
    }


}
