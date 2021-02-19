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
import it.unimi.dsi.fastutil.ints.Int2IntMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import org.jooq.lambda.function.Function4;
import org.jooq.lambda.tuple.Tuple3;
import org.jooq.lambda.tuple.Tuple4;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * Simple implementation of FastPreferenceData backed by nested lists.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Saúl Vargas (saul.vargas@uam.es)
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class BestRatingFastUpdateablePreferenceData<U, I> extends AbstractSimpleFastUpdateablePreferenceData<U,I>
{
    /**
     * Constructor with default IdxPref to IdPref converter.
     *
     * @param numPreferences Initial number of total preferences.
     * @param uidxList       List of lists of preferences by user index.
     * @param iidxList       List of lists of preferences by item index.
     * @param uIndex         User index.
     * @param iIndex         Item index.
     */
    protected BestRatingFastUpdateablePreferenceData(int numPreferences, List<List<IdxPref>> uidxList, List<List<IdxPref>> iidxList,
                                                     FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex)
    {
        super(numPreferences, uidxList, iidxList, uIndex, iIndex, (x,y) -> x > y,
             (Function<IdxPref, IdPref<I>> & Serializable) p -> new IdPref<>(iIndex.iidx2item(p)),
             (Function<IdxPref, IdPref<U>> & Serializable) p -> new IdPref<>(uIndex.uidx2user(p)));
    }

    /**
     * Constructor with custom IdxPref to IdPref converter.
     *
     * @param numPreferences Initial number of total preferences.
     * @param uidxList       List of lists of preferences by user index.
     * @param iidxList       List of lists of preferences by item index.
     * @param uIndex         User index.
     * @param iIndex         Item index.
     * @param uPrefFun       User IdxPref to IdPref converter.
     * @param iPrefFun       Item IdxPref to IdPref converter.
     */
    protected BestRatingFastUpdateablePreferenceData(int numPreferences, List<List<IdxPref>> uidxList, List<List<IdxPref>> iidxList,
                                                     FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex,
                                                     Function<IdxPref, IdPref<I>> uPrefFun, Function<IdxPref, IdPref<U>> iPrefFun)
    {
        super(numPreferences, uidxList, iidxList, uIndex, iIndex, (x,y) -> x > y, uPrefFun, iPrefFun);
    }

    /**
     * Loads a SimpleFastPreferenceData from a stream of user-item-value triples.
     *
     * @param <U>    User type.
     * @param <I>    Item type.
     * @param tuples Stream of user-item-value triples.
     * @param uIndex User index.
     * @param iIndex Item index.
     * @return an instance of SimpleFastPreferenceData containing the data from the input stream.
     */
    public static <U, I> BestRatingFastUpdateablePreferenceData<U, I> load(Stream<Tuple3<U, I, Double>> tuples, FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex)
    {
        return load(tuples.map(t -> t.concat((Void) null)),
                    (uidx, iidx, v, o) -> new IdxPref(iidx, v),
                    (uidx, iidx, v, o) -> new IdxPref(uidx, v),
                    uIndex, iIndex,
                    (Function<IdxPref, IdPref<I>> & Serializable) p -> new IdPref<>(iIndex.iidx2item(p)),
                    (Function<IdxPref, IdPref<U>> & Serializable) p -> new IdPref<>(uIndex.uidx2user(p)));
    }

    /**
     * Loads a SimpleFastPreferenceData from a stream of user-item-value-other tuples. It can accomodate other information, thus you need to provide sub-classes of IdxPref IdPref accomodating for this new information.
     *
     * @param <U>         User type.
     * @param <I>         Item type.
     * @param <O>         Additional information type.
     * @param tuples      Stream of user-item-value-other tuples.
     * @param uIdxPrefFun Converts a tuple to a user IdxPref.
     * @param iIdxPrefFun Converts a tuple to a item IdxPref.
     * @param uIndex      User index.
     * @param iIndex      Item index.
     * @param uIdPrefFun  User IdxPref to IdPref converter.
     * @param iIdPrefFun  Item IdxPref to IdPref converter.
     * @return an instance of SimpleFastPreferenceData containing the data from the input stream.
     */
    public static <U, I, O> BestRatingFastUpdateablePreferenceData<U, I> load(Stream<Tuple4<U, I, Double, O>> tuples,
                                                                              Function4<Integer, Integer, Double, O, ? extends IdxPref> uIdxPrefFun,
                                                                              Function4<Integer, Integer, Double, O, ? extends IdxPref> iIdxPrefFun,
                                                                              FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex,
                                                                              Function<IdxPref, IdPref<I>> uIdPrefFun,
                                                                              Function<IdxPref, IdPref<U>> iIdPrefFun)
    {
        AtomicInteger numPreferences = new AtomicInteger();

        List<Int2IntMap> uidxVisited = new ArrayList<>();
        List<List<IdxPref>> uidxList = new ArrayList<>();
        for (int uidx = 0; uidx < uIndex.numUsers(); uidx++)
        {
            uidxList.add(null);
            uidxVisited.add(new Int2IntOpenHashMap());
        }

        List<Int2IntMap> iidxVisited = new ArrayList<>();
        List<List<IdxPref>> iidxList = new ArrayList<>();
        for (int iidx = 0; iidx < iIndex.numItems(); iidx++)
        {
            iidxList.add(null);
            iidxVisited.add(new Int2IntOpenHashMap());
        }

        tuples.forEach(t ->
        {
            int uidx = uIndex.user2uidx(t.v1);
            int iidx = iIndex.item2iidx(t.v2);

            if(uidxVisited.get(uidx).containsKey(iidx))
            {
                int uidxIndex = uidxVisited.get(uidx).get(iidx);
                List<IdxPref> uList = uidxList.get(uidx);
                double oldValue = uList.get(uidxIndex).v2;
                double newValue = Math.max(t.v3, oldValue);
                uList.set(uidxIndex, uIdxPrefFun.apply(uidx, iidx, newValue, t.v4));

                int iidxIndex = iidxVisited.get(iidx).get(uidx);
                List<IdxPref> iList = iidxList.get(iidx);
                iList.set(iidxIndex, iIdxPrefFun.apply(uidx, iidx, newValue, t.v4));
            }
            else
            {
                numPreferences.incrementAndGet();

                List<IdxPref> uList = uidxList.get(uidx);
                if (uList == null)
                {
                    uList = new ArrayList<>();
                    uidxList.set(uidx, uList);
                }
                uidxVisited.get(uidx).put(iidx, uList.size());
                uList.add(uIdxPrefFun.apply(uidx, iidx, t.v3, t.v4));

                List<IdxPref> iList = iidxList.get(iidx);
                if (iList == null)
                {
                    iList = new ArrayList<>();
                    iidxList.set(iidx, iList);
                }
                iidxVisited.get(iidx).put(uidx, iList.size());
                iList.add(iIdxPrefFun.apply(uidx, iidx, t.v3, t.v4));
            }
        });

        return new BestRatingFastUpdateablePreferenceData<>(numPreferences.intValue(), uidxList, iidxList, uIndex, iIndex, uIdPrefFun, iIdPrefFun);
    }

    @Override
    public double updatedValue(double newValue, double oldValue)
    {
        return Math.max(newValue, oldValue);
    }
}
