/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
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

import java.io.Serializable;
import java.util.*;
import java.util.function.BiPredicate;
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
public abstract class AbstractSimpleFastUpdateablePreferenceData<U, I> extends StreamsAbstractFastUpdateablePreferenceData<U, I> implements FastUpdateablePointWisePreferenceData<U, I>, Serializable
{
    /**
     * User preferences.
     */
    private final List<List<IdxPref>> uidxList;
    /**
     * Item preferences.
     */
    private final List<List<IdxPref>> iidxList;
    /**
     * Current number of preferences.
     */
    private int numPreferences;

    private final short UPDATED = 0;
    private final short NEW = 1;
    private final short NOTUPDATED = -1;

    private final BiPredicate<Double, Double> predicate;

    /**
     * Constructor with default IdxPref to IdPref converter.
     *
     * @param numPreferences Initial number of total preferences.
     * @param uidxList       List of lists of preferences by user index.
     * @param iidxList       List of lists of preferences by item index.
     * @param uIndex         User index.
     * @param iIndex         Item index.
     */
    protected AbstractSimpleFastUpdateablePreferenceData(int numPreferences, List<List<IdxPref>> uidxList, List<List<IdxPref>> iidxList,
                                                         FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex,
                                                         BiPredicate<Double, Double> predicate)
    {
        this(numPreferences, uidxList, iidxList, uIndex, iIndex, predicate,
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
    protected AbstractSimpleFastUpdateablePreferenceData(int numPreferences, List<List<IdxPref>> uidxList, List<List<IdxPref>> iidxList,
                                                         FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, BiPredicate<Double, Double> predicate,
                                                         Function<IdxPref, IdPref<I>> uPrefFun, Function<IdxPref, IdPref<U>> iPrefFun)
    {
        super(uIndex, iIndex, uPrefFun, iPrefFun);
        this.uidxList = uidxList;
        this.iidxList = iidxList;
        this.numPreferences = numPreferences;
        uidxList.parallelStream()
                .filter(Objects::nonNull)
                .forEach(l -> l.sort(comparingInt(IdxPref::v1)));
        iidxList.parallelStream()
                .filter(Objects::nonNull)
                .forEach(l -> l.sort(comparingInt(IdxPref::v1)));
        this.predicate = predicate;
    }

    @Override
    public int numUsers(int iidx)
    {
        if (iidxList.get(iidx) == null)
        {
            return 0;
        }
        return iidxList.get(iidx).size();
    }

    @Override
    public int numItems(int uidx)
    {
        if (uidxList.get(uidx) == null)
        {
            return 0;
        }
        return uidxList.get(uidx).size();
    }

    @Override
    public Stream<IdxPref> getUidxPreferences(int uidx)
    {
        if (uidxList.get(uidx) == null)
        {
            return Stream.empty();
        }
        else
        {
            return uidxList.get(uidx).stream();
        }
    }

    @Override
    public Stream<IdxPref> getIidxPreferences(int iidx)
    {
        if (iidxList.get(iidx) == null)
        {
            return Stream.empty();
        }
        else
        {
            return iidxList.get(iidx).stream();
        }
    }

    @Override
    public int numPreferences()
    {
        return numPreferences;
    }

    @Override
    public IntStream getUidxWithPreferences()
    {
        return IntStream.range(0, numUsers())
                .filter(uidx -> uidxList.get(uidx) != null);
    }

    @Override
    public IntStream getIidxWithPreferences()
    {
        return IntStream.range(0, this.numItems())
                .filter(iidx -> iidxList.get(iidx) != null);
    }

    @Override
    public int numUsersWithPreferences()
    {
        return (int) uidxList.stream()
                .filter(Objects::nonNull)
                .count();
    }

    @Override
    public int numItemsWithPreferences()
    {
        return (int) iidxList.stream()
                .filter(Objects::nonNull)
                .count();
    }

    @Override
    public Optional<IdxPref> getPreference(int uidx, int iidx)
    {
        List<IdxPref> uList = uidxList.get(uidx);
        if (uList == null)
        {
            return Optional.empty();
        }
        Comparator<IdxPref> comp = comparingInt(x -> x.v1);
        int position = Collections.binarySearch(uList, new IdxPref(iidx, 1.0), comp);

        if (position >= 0)
        {
            return Optional.of(uList.get(position));
        }

        return Optional.empty();
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
    public int addUser(U u)
    {
        int uidx = ((FastUpdateableUserIndex<U>) this.ui).addUser(u);
        if (this.uidxList.size() == uidx) // If the user is really new
        {
            this.uidxList.add(null);
        }
        return uidx;
    }

    @Override
    public int addItem(I i)
    {
        int iidx = ((FastUpdateableItemIndex<I>) this.ii).addItem(i);
        if (this.iidxList.size() == iidx) // If the item is really new
        {
            this.iidxList.add(null);
        }
        return iidx;
    }

    @Override
    public boolean updateRating(int uidx, int iidx, double rating)
    {
        // If the user or the item are not in the preference data, do nothing.
        if (uidx < 0 || this.uidxList.size() <= uidx || iidx < 0 || this.iidxList.size() <= iidx)
        {
            return false;
        }

        boolean hasBeenUpdated;

        // Update the value for the user
        if(this.uidxList.get(uidx) == null)
        {
            List<IdxPref> idxPrefList = new ArrayList<>();
            idxPrefList.add(new IdxPref(iidx, rating));
            this.uidxList.set(uidx, idxPrefList);
            this.numPreferences++;
            hasBeenUpdated = true;
        }
        else // If the user has at least one preference:
        {
            // Update the preference for the user.
            short value = this.updatePreference(iidx, rating, this.uidxList.get(uidx));
            hasBeenUpdated = value != NOTUPDATED;
            if(value == NEW) this.numPreferences++;
        }

        if(hasBeenUpdated) // If the value for the item has been updated, we do it:
        {
            // Update the value for the item.
            if (this.iidxList.get(iidx) == null) // If the item does not have ratings.
            {
                List<IdxPref> idxPrefList = new ArrayList<>();
                idxPrefList.add(new IdxPref(uidx, rating));
                this.iidxList.set(iidx, idxPrefList);
            }
            else // If the item has been rated by at least one user.
            {
                this.updatePreference(uidx, rating, this.iidxList.get(iidx));
            }
        }

        return hasBeenUpdated;
    }

    /**
     * Updates a preference.
     *
     * @param idx   The identifier of the preference to add.
     * @param value The rating value.
     * @param list  The list in which we want to update the preference.
     * @return -1 if the rating did not change, 0 if it did, and 1 if the rating is a new added value.
     */
    private short updatePreference(int idx, double value, List<IdxPref> list)
    {
        IdxPref newIdx = new IdxPref(idx, value);

        // Use binary search to find the rating.
        Comparator<IdxPref> comp = comparingInt(x -> x.v1);
        int position = Collections.binarySearch(list, newIdx, comp);

        if (position < 0) // The rating does not exist.
        {
            position = Math.abs(position + 1);
            list.add(position, newIdx);
            return NEW;
        }
        else // The rating did already exist.
        {
            IdxPref oldValue = list.get(position);
            double newValue = this.updatedValue(value, oldValue.v2);

            if(predicate.test(newValue, oldValue.v2))
            {
                list.set(position, new IdxPref(idx, newValue));
                return UPDATED;
            }
            else
            {
                return NOTUPDATED;
            }
        }
    }

    @Override
    public void updateDelete(int uidx, int iidx)
    {
        // If the user or the item are not in the preference data, do nothing.
        if (uidx < 0 || this.uidxList.size() <= uidx || iidx < 0 || this.iidxList.size() <= iidx)
        {
            return;
        }

        // First, delete from the uidxList.
        if (this.updateDelete(iidx, this.uidxList.get(uidx)))
        {
            // Then, delete from the iidxList.
            this.updateDelete(uidx, this.iidxList.get(iidx));
            this.numPreferences--;
        }
    }

    /**
     * Deletes a rating from the data.
     *
     * @param idx  Identifier of the element to delete.
     * @param list List from where the element has to be removed.
     * @return true if the element was removed, false otherwise.
     */
    private boolean updateDelete(int idx, List<IdxPref> list)
    {
        // If the list is empty, do nothing.
        if (list == null)
        {
            return false;
        }

        // Search for the position of the element to remove.
        IdxPref newIdx = new IdxPref(idx, 1.0);
        Comparator<IdxPref> comp = (x, y) -> x.v1 - y.v1;
        int position = Collections.binarySearch(list, newIdx, comp);

        // If it exists.
        if (position >= 0)
        {
            list.remove(position);
            return true;
        }

        return false;
    }

    @Override
    public void clear()
    {
        this.numPreferences = 0;
        this.uidxList.parallelStream()
                .filter(Objects::nonNull)
                .forEach(List::clear);
        this.iidxList.parallelStream()
                .filter(Objects::nonNull)
                .forEach(List::clear);
    }

}
