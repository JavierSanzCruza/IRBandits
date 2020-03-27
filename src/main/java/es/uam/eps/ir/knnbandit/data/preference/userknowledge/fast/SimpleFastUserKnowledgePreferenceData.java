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

import es.uam.eps.ir.knnbandit.utils.OrderedListCombiner;
import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.function.Function5;
import org.jooq.lambda.tuple.Tuple3;
import org.jooq.lambda.tuple.Tuple4;
import org.jooq.lambda.tuple.Tuple5;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.util.Comparator.comparingInt;

/**
 * Simple implementation of FastPreferenceData backed by nested lists.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Saúl Vargas (saul.vargas@uam.es)
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class SimpleFastUserKnowledgePreferenceData<U, I> extends StreamsAbstractFastUserKnowledgePreferenceData<U, I> implements FastUserKnowledgePointWisePreferenceData<U, I>, Serializable
{
    /**
     * User preferences.
     */
    private final List<List<IdxPref>> uidxKnownList;
    /**
     * Item preferences.
     */
    private final List<List<IdxPref>> iidxKnownList;
    /**
     * Information about the knowledge of the users.
     */
    private final List<List<IdxPref>> uidxUnknownList;
    /**
     * Information about the knowledge of the items.
     */
    private final List<List<IdxPref>> iidxUnknownList;
    /**
     * Current number of preferences.
     */
    private int numPreferences;
    /**
     * Number of preferences known.
     */
    private int numKnown;

    /**
     * Constructor with default IdxPref to IdPref converter.
     *
     * @param numPreferences  Initial number of total preferences.
     * @param uidxKnownList   List of lists of preferences by user index.
     * @param iidxUnknownList List of lists of preferences by item index.
     * @param uIndex          User index.
     * @param iIndex          Item index.
     */
    protected SimpleFastUserKnowledgePreferenceData(int numPreferences, int numKnown, List<List<IdxPref>> uidxKnownList, List<List<IdxPref>> iidxKnownList,
                                                    List<List<IdxPref>> uidxUnknownList, List<List<IdxPref>> iidxUnknownList,
                                                    FastUserIndex<U> uIndex, FastItemIndex<I> iIndex)
    {
        this(numPreferences, numKnown, uidxKnownList, iidxKnownList, uidxUnknownList, iidxUnknownList, uIndex, iIndex,
             (Function<IdxPref, IdPref<I>> & Serializable) p -> new IdPref<>(iIndex.iidx2item(p)),
             (Function<IdxPref, IdPref<U>> & Serializable) p -> new IdPref<>(uIndex.uidx2user(p)));
    }

    /**
     * Constructor with custom IdxPref to IdPref converter.
     *
     * @param numPreferences  Initial number of total preferences.
     * @param uidxKnownList   List of lists of preferences by user index.
     * @param iidxUnknownList List of lists of preferences by item index.
     * @param uIndex          User index.
     * @param iIndex          Item index.
     * @param uPrefFun        User IdxPref to IdPref converter.
     * @param iPrefFun        Item IdxPref to IdPref converter.
     */
    protected SimpleFastUserKnowledgePreferenceData(int numPreferences, int numKnown, List<List<IdxPref>> uidxKnownList, List<List<IdxPref>> iidxKnownList,
                                                    List<List<IdxPref>> uidxUnknownList, List<List<IdxPref>> iidxUnknownList,
                                                    FastUserIndex<U> uIndex, FastItemIndex<I> iIndex,
                                                    Function<IdxPref, IdPref<I>> uPrefFun, Function<IdxPref, IdPref<U>> iPrefFun)
    {
        super(uIndex, iIndex, uPrefFun, iPrefFun);
        this.uidxKnownList = uidxKnownList;
        this.iidxKnownList = iidxKnownList;
        this.uidxUnknownList = uidxUnknownList;
        this.iidxUnknownList = iidxUnknownList;

        this.numPreferences = numPreferences;
        this.numKnown = numKnown;

        uidxKnownList.parallelStream()
                .filter(l -> l != null)
                .forEach(l -> l.sort(comparingInt(IdxPref::v1)));
        iidxKnownList.parallelStream()
                .filter(l -> l != null)
                .forEach(l -> l.sort(comparingInt(IdxPref::v1)));
        uidxUnknownList.parallelStream()
                .filter(l -> l != null)
                .forEach(l -> l.sort(comparingInt(IdxPref::v1)));
        iidxUnknownList.parallelStream()
                .filter(l -> l != null)
                .forEach(l -> l.sort(comparingInt(IdxPref::v1)));
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
    public static <U, I> SimpleFastUserKnowledgePreferenceData<U, I> load(Stream<Tuple4<U, I, Double, Boolean>> tuples, FastUserIndex<U> uIndex, FastItemIndex<I> iIndex)
    {
        return load(tuples.map(t -> t.concat((Void) null)),
                    (uidx, iidx, v, known, o) -> new IdxPref(iidx, v),
                    (uidx, iidx, v, known, o) -> new IdxPref(uidx, v),
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
    public static <U, I, O> SimpleFastUserKnowledgePreferenceData<U, I> load(Stream<Tuple5<U, I, Double, Boolean, O>> tuples,
                                                                             Function5<Integer, Integer, Double, Boolean, O, ? extends IdxPref> uIdxPrefFun,
                                                                             Function5<Integer, Integer, Double, Boolean, O, ? extends IdxPref> iIdxPrefFun,
                                                                             FastUserIndex<U> uIndex, FastItemIndex<I> iIndex,
                                                                             Function<IdxPref, IdPref<I>> uIdPrefFun,
                                                                             Function<IdxPref, IdPref<U>> iIdPrefFun)
    {
        AtomicInteger numPreferences = new AtomicInteger();
        AtomicInteger numKnown = new AtomicInteger();
        List<List<IdxPref>> uidxKnownList = new ArrayList<>();
        List<List<IdxPref>> uidxUnknownList = new ArrayList<>();

        for (int uidx = 0; uidx < uIndex.numUsers(); uidx++)
        {
            uidxKnownList.add(null);
            uidxUnknownList.add(null);
        }


        List<List<IdxPref>> iidxKnownList = new ArrayList<>();
        List<List<IdxPref>> iidxUnknownList = new ArrayList<>();
        for (int iidx = 0; iidx < iIndex.numItems(); iidx++)
        {
            iidxKnownList.add(null);
            iidxUnknownList.add(null);
        }

        tuples.forEach(t ->
                       {
                           int uidx = uIndex.user2uidx(t.v1);
                           int iidx = iIndex.item2iidx(t.v2);

                           numPreferences.incrementAndGet();

                           boolean known = t.v4;
                           if (known)
                           {
                               numKnown.incrementAndGet();
                               List<IdxPref> uList = uidxKnownList.get(uidx);
                               if (uList == null)
                               {
                                   uList = new ArrayList<>();
                                   uidxKnownList.set(uidx, uList);
                               }
                               uList.add(uIdxPrefFun.apply(uidx, iidx, t.v3, known, t.v5));

                               List<IdxPref> iList = iidxKnownList.get(iidx);
                               if (iList == null)
                               {
                                   iList = new ArrayList<>();
                                   iidxKnownList.set(iidx, iList);
                               }
                               iList.add(iIdxPrefFun.apply(uidx, iidx, t.v3, known, t.v5));
                           }
                           else
                           {
                               List<IdxPref> uList = uidxUnknownList.get(uidx);
                               if (uList == null)
                               {
                                   uList = new ArrayList<>();
                                   uidxUnknownList.set(uidx, uList);
                               }
                               uList.add(uIdxPrefFun.apply(uidx, iidx, t.v3, known, t.v5));

                               List<IdxPref> iList = iidxUnknownList.get(iidx);
                               if (iList == null)
                               {
                                   iList = new ArrayList<>();
                                   iidxUnknownList.set(iidx, iList);
                               }
                               iList.add(iIdxPrefFun.apply(uidx, iidx, t.v3, known, t.v5));
                           }
                       });

        return new SimpleFastUserKnowledgePreferenceData<>(numPreferences.intValue(), numKnown.intValue(), uidxKnownList, iidxKnownList, uidxUnknownList, iidxUnknownList, uIndex, iIndex, uIdPrefFun, iIdPrefFun);
    }

    @Override
    public int numUsers(int iidx)
    {
        return this.numKnownUsers(iidx) + this.numUnknownUsers(iidx);
    }

    @Override
    public int numItems(int uidx)
    {
        return this.numKnownItems(uidx) + this.numUnknownItems(uidx);
    }

    @Override
    public Stream<IdxPref> getUidxPreferences(int uidx)
    {
        Stream<IdxPref> known = this.getUidxKnownPreferences(uidx);
        Stream<IdxPref> unknown = this.getUidxUnknownPreferences(uidx);
        return OrderedListCombiner.mergeLists(known, unknown, (o1, o2) -> (o1.v1 - o2.v1), (x, y) -> y).stream();
    }

    @Override
    public Stream<IdxPref> getIidxPreferences(int iidx)
    {
        Stream<IdxPref> known = this.getIidxKnownPreferences(iidx);
        Stream<IdxPref> unknown = this.getIidxUnknownPreferences(iidx);
        return OrderedListCombiner.mergeLists(known, unknown, (o1, o2) -> (o1.v1 - o2.v1), (x, y) -> y).stream();
    }

    @Override
    public int numPreferences()
    {
        return numPreferences;
    }

    @Override
    public IntStream getUidxWithPreferences()
    {
        return IntStream.range(0, numUsers()).filter(uidx -> (uidxKnownList.get(uidx) != null && uidxUnknownList.get(uidx) != null));
    }

    @Override
    public IntStream getIidxWithPreferences()
    {
        return IntStream.range(0, numItems()).filter(iidx -> (iidxKnownList.get(iidx) != null && iidxUnknownList.get(iidx) != null));
    }

    @Override
    public int numUsersWithPreferences()
    {
        return (int) IntStream.range(0, numUsers())
                .filter(uidx -> (uidxKnownList.get(uidx) != null && uidxUnknownList.get(uidx) != null))
                .count();
    }

    @Override
    public int numItemsWithPreferences()
    {
        return (int) IntStream.range(0, numItems())
                .filter(iidx -> (iidxKnownList.get(iidx) != null && iidxUnknownList.get(iidx) != null))
                .count();
    }

    @Override
    public Optional<IdxPref> getPreference(int uidx, int iidx)
    {
        Optional<IdxPref> pref = this.getUnknownPreference(uidx, iidx);
        if (pref.isPresent())
        {
            return pref;
        }
        else
        {
            return this.getKnownPreference(uidx, iidx);
        }
    }

    @Override
    public Optional<? extends IdPref<I>> getPreference(U u, I i)
    {
        if (this.containsUser(u) && this.containsItem(i))
        {
            Optional<? extends IdxPref> pref = getPreference(user2uidx(u), item2iidx(i));

            return pref.map(uPrefFun::apply);
        }
        else
        {
            return Optional.empty();
        }
    }

    @Override
    public Optional<IdxPref> getKnownPreference(int uidx, int iidx)
    {
        List<IdxPref> uList = uidxKnownList.get(uidx);
        if (uList == null)
        {
            return Optional.empty();
        }
        Comparator<IdxPref> comp = (x, y) -> x.v1 - y.v1;
        int position = Collections.binarySearch(uList, new IdxPref(iidx, 1.0), comp);

        if (position >= 0)
        {
            return Optional.of(uList.get(position));
        }

        return Optional.empty();
    }

    @Override
    public Optional<IdxPref> getUnknownPreference(int uidx, int iidx)
    {
        List<IdxPref> uList = uidxUnknownList.get(uidx);
        if (uList == null)
        {
            return Optional.empty();
        }
        Comparator<IdxPref> comp = (x, y) -> x.v1 - y.v1;
        int position = Collections.binarySearch(uList, new IdxPref(iidx, 1.0), comp);

        if (position >= 0)
        {
            return Optional.of(uList.get(position));
        }

        return Optional.empty();
    }

    @Override
    public Optional<? extends IdPref<I>> getKnownPreference(U u, I i)
    {
        if (this.containsUser(u) && this.containsItem(i))
        {
            Optional<? extends IdxPref> pref = getKnownPreference(user2uidx(u), item2iidx(i));

            return pref.map(uPrefFun::apply);
        }
        else
        {
            return Optional.empty();
        }
    }

    @Override
    public Optional<? extends IdPref<I>> getUnknownPreference(U u, I i)
    {
        if (this.containsUser(u) && this.containsItem(i))
        {
            Optional<? extends IdxPref> pref = getUnknownPreference(user2uidx(u), item2iidx(i));

            return pref.map(uPrefFun::apply);
        }
        else
        {
            return Optional.empty();
        }
    }

    @Override
    public int numKnownUsers(int iidx)
    {
        if (iidx < 0 || iidx >= this.numItems())
        {
            return 0;
        }
        List<IdxPref> iList = iidxKnownList.get(iidx);
        if (iList == null || iList.isEmpty())
        {
            return 0;
        }
        return iList.size();
    }

    @Override
    public int numUnknownUsers(int iidx)
    {
        if (iidx < 0 || iidx >= this.numItems())
        {
            return 0;
        }
        List<IdxPref> iList = iidxUnknownList.get(iidx);
        if (iList == null || iList.isEmpty())
        {
            return 0;
        }
        return iList.size();
    }

    @Override
    public int numKnownItems(int uidx)
    {
        if (uidx < 0 || uidx >= this.numUsers())
        {
            return 0;
        }
        List<IdxPref> uList = uidxKnownList.get(uidx);
        if (uList == null || uList.isEmpty())
        {
            return 0;
        }
        return uList.size();
    }

    @Override
    public int numUnknownItems(int uidx)
    {
        if (uidx < 0 || uidx >= this.numUsers())
        {
            return 0;
        }
        List<IdxPref> uList = uidxUnknownList.get(uidx);
        if (uList == null || uList.isEmpty())
        {
            return 0;
        }
        return uList.size();
    }

    @Override
    public Stream<IdxPref> getUidxKnownPreferences(int uidx)
    {
        if (uidx < 0 || uidx >= this.numUsers())
        {
            return Stream.empty();
        }
        List<IdxPref> uList = uidxKnownList.get(uidx);
        if (uList == null || uList.isEmpty())
        {
            return Stream.empty();
        }
        return uList.stream();
    }

    @Override
    public Stream<IdxPref> getIidxKnownPreferences(int iidx)
    {
        if (iidx < 0 || iidx >= this.numUsers())
        {
            return Stream.empty();
        }
        List<IdxPref> iList = iidxKnownList.get(iidx);
        if (iList == null || iList.isEmpty())
        {
            return Stream.empty();
        }
        return iList.stream();
    }

    @Override
    public Stream<IdxPref> getUidxUnknownPreferences(int uidx)
    {
        if (uidx < 0 || uidx >= this.numUsers())
        {
            return Stream.empty();
        }
        List<IdxPref> uList = uidxUnknownList.get(uidx);
        if (uList == null || uList.isEmpty())
        {
            return Stream.empty();
        }
        return uList.stream();
    }

    @Override
    public Stream<IdxPref> getIidxUnknownPreferences(int iidx)
    {
        if (iidx < 0 || iidx >= this.numUsers())
        {
            return Stream.empty();
        }
        List<IdxPref> iList = iidxUnknownList.get(iidx);
        if (iList == null || iList.isEmpty())
        {
            return Stream.empty();
        }
        return iList.stream();
    }

    @Override
    public int numKnown()
    {
        return this.numKnown;
    }

    @Override
    public PreferenceData<U, I> getKnownPreferenceData()
    {
        List<Tuple3<U, I, Double>> triplets = new ArrayList<>();
        this.getAllUidx().forEach(uidx ->
                                  {
                                      List<IdxPref> uidxList = this.uidxKnownList.get(uidx);
                                      if (uidxList != null && !uidxList.isEmpty())
                                      {
                                          U u = uidx2user(uidx);
                                          for (IdxPref iidxPref : uidxList)
                                          {
                                              triplets.add(new Tuple3<>(u, iidx2item(iidxPref.v1), iidxPref.v2));
                                          }
                                      }
                                  });

        return SimpleFastPreferenceData.load(triplets.stream(), this, this);
    }

    @Override
    public PreferenceData<U, I> getUnknownPreferenceData()
    {
        List<Tuple3<U, I, Double>> triplets = new ArrayList<>();
        this.getAllUidx().forEach(uidx ->
                                  {
                                      List<IdxPref> uidxList = this.uidxUnknownList.get(uidx);
                                      if (uidxList != null && !uidxList.isEmpty())
                                      {
                                          U u = uidx2user(uidx);
                                          for (IdxPref iidxPref : uidxList)
                                          {
                                              triplets.add(new Tuple3<>(u, iidx2item(iidxPref.v1), iidxPref.v2));
                                          }
                                      }
                                  });

        return SimpleFastPreferenceData.load(triplets.stream(), this, this);
    }

    @Override
    public PreferenceData<U, I> getPreferenceData()
    {
        List<Tuple3<U, I, Double>> triplets = new ArrayList<>();
        this.getAllUidx().forEach(uidx ->
                                  {
                                      List<IdxPref> uidxList = this.uidxUnknownList.get(uidx);
                                      if (uidxList != null && !uidxList.isEmpty())
                                      {
                                          U u = uidx2user(uidx);
                                          for (IdxPref iidxPref : uidxList)
                                          {
                                              triplets.add(new Tuple3<>(u, iidx2item(iidxPref.v1), iidxPref.v2));
                                          }
                                      }

                                      uidxList = this.uidxKnownList.get(uidx);
                                      if (uidxList != null && !uidxList.isEmpty())
                                      {
                                          U u = uidx2user(uidx);
                                          for (IdxPref iidxPref : uidxList)
                                          {
                                              triplets.add(new Tuple3<>(u, iidx2item(iidxPref.v1), iidxPref.v2));
                                          }
                                      }

                                  });

        return SimpleFastPreferenceData.load(triplets.stream(), this, this);
    }


}
