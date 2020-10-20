/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.data.datasets;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A dataset. Contains all the info dataset.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class GeneralDataset<U, I>
{
    /**
     * User index. Relates integer identifiers to real user values.
     */
    protected final FastUpdateableUserIndex<U> uIndex;
    /**
     * Item index. Relates integer identifiers to real item values.
     */
    protected final FastUpdateableItemIndex<I> iIndex;
    /**
     * Rating data.
     */
    protected final SimpleFastPreferenceData<U, I> prefData;
    /**
     * Number of relevant ratings.
     */
    protected final int numRel;
    /**
     * Predicate for measuring the relevance.
     */
    DoublePredicate relevance;

    /**
     * Constructor.
     *
     * @param uIndex   User index.
     * @param iIndex   Item index.
     * @param prefData Preference data.
     * @param numRel   Number of relevant (user, item) pairs.
     */
    protected GeneralDataset(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, int numRel, DoublePredicate relevance)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.prefData = prefData;
        this.numRel = numRel;
        this.relevance = relevance;
    }

    /**
     * Obtains the user index of the dataset.
     *
     * @return the user index of the dataset.
     */
    public FastUpdateableUserIndex<U> getUserIndex()
    {
        return uIndex;
    }

    /**
     * Obtains the item index of the dataset.
     *
     * @return the item index of the dataset.
     */
    public FastUpdateableItemIndex<I> getItemIndex()
    {
        return iIndex;
    }

    /**
     * Obtains the preference data.
     *
     * @return the preference data of the dataset.
     */
    public SimpleFastPreferenceData<U, I> getPrefData()
    {
        return prefData;
    }

    /**
     * Get the number of relevant (user, item) pairs.
     *
     * @return the number of relevant (user, item) pairs.
     */
    public int getNumRel()
    {
        return numRel;
    }

    /**
     * Given a list of (uidx, iidx) pairs, finds how many relevant pairs there are.
     *
     * @param list the list of (uidx, iidx)
     * @return the count of how many relevant (uidx, iidx) pairs appear in the list.
     */
    public int getNumRel(List<Tuple2<Integer, Integer>> list)
    {
        return list.stream().mapToInt(t ->
        {
            if (prefData.numItems(t.v1) > 0 && prefData.numUsers(t.v2) > 0)
            {
                Optional<IdxPref> opt = prefData.getPreference(t.v1, t.v2);
                if (opt.isPresent() && relevance.test(opt.get().v2))
                {
                    return 1;
                }
            }
            return 0;
        }).sum();
    }

    /**
     * Obtains the set of users in the dataset.
     * @return an stream containing the users in the dataset.
     */
    public Stream<U> getUsers()
    {
        return this.uIndex.getAllUsers();
    }

    /**
     * Obtains the set of identifiers of users in the dataset.
     * @return an stream containing the identifier of the users in the dataset.
     */
    public IntStream getUidx()
    {
        return this.uIndex.getAllUidx();
    }
    /**
     * Obtains the set of items in the dataset.
     * @return an stream containing the items in the dataset.
     */
    public Stream<I> getItems()
    {
        return this.iIndex.getAllItems();
    }
    /**
     * Obtains the set of identifiers of items in the dataset.
     * @return an stream containing the identifier of the items in the dataset.
     */
    public IntStream getIidx()
    {
        return this.iIndex.getAllIidx();
    }
    /**
     * Obtains the set of users with ratings in the dataset.
     * @return an stream containing the users with ratings in the dataset.
     */
    public Stream<U> getUsersWithPreferences()
    {
        return this.prefData.getUsersWithPreferences();
    }
    /**
     * Obtains the set of identifiers of users with ratings in the dataset.
     * @return an stream containing the identifiers users with ratings in the dataset.
     */
    public IntStream getUidxWithPreferences()
    {
        return this.prefData.getUidxWithPreferences();
    }

    /**
     * Obtains the number of users in the dataset.
     * @return the number of users.
     */
    public int numUsers()
    {
        return this.uIndex.numUsers();
    }

    /**
     * Obtains the number of items in the dataset.
     * @return the number of items.
     */
    public int numItems()
    {
        return this.iIndex.numItems();
    }

    /**
     * Finds a textual representation of the dataset.
     * @return the textual representation of the dataset.
     */
    public String toString()
    {
        return "Users: " +
                this.numUsers() +
                "\nItems: " +
                this.numItems() +
                "\nNum. ratings: " +
                this.prefData.numPreferences() +
                "\nNum. relevant: " +
                this.getNumRel();
    }



    /**
     * Loads a dataset.
     *
     * @param filename       the name of the file where the dataset is.
     * @param uParser        parser for the users.
     * @param iParser        parser for the items.
     * @param separator      separator for the different fields in the file.
     * @param weightFunction function for determining the rating value.
     * @param relevance      function that determines if a (user, item) pair is relevant or not.
     * @param <U>            type of the users.
     * @param <I>            type of the items.
     * @return the dataset.
     * @throws IOException if something fails while reading the dataset file.
     */
    public static <U, I> GeneralDataset<U, I> load(String filename, Parser<U> uParser, Parser<I> iParser, String separator, DoubleUnaryOperator weightFunction, DoublePredicate relevance) throws IOException
    {
        List<Tuple3<U, I, Double>> triplets = new ArrayList<>();
        Set<U> users = new HashSet<>();
        Set<I> items = new HashSet<>();

        int numrel = 0;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filename))))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] split = line.split(separator);
                U user = uParser.parse(split[0]);
                I item = iParser.parse(split[1]);
                double val = Parsers.dp.parse(split[2]);

                users.add(user);
                items.add(item);

                double rating = weightFunction.applyAsDouble(val);
                if (relevance.test(rating))
                {
                    numrel++;
                }

                triplets.add(new Tuple3<>(user, item, rating));
            }
        }

        // Create the data.
        FastUpdateableUserIndex<U> uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        FastUpdateableItemIndex<I> iIndex = SimpleFastUpdateableItemIndex.load(items.stream());
        SimpleFastPreferenceData<U, I> prefData = SimpleFastPreferenceData.load(triplets.stream(), uIndex, iIndex);

        return new GeneralDataset<>(uIndex, iIndex, prefData, numrel, relevance);
    }

    /**
     * Loads a dataset from another dataset.
     * @param dataset the original dataset.
     * @param list a list of (user,item) interactions.
     * @param <U> type of the users.
     * @param <I> type of the items.
     * @return the new dataset.
     */
    public static <U, I> GeneralDataset<U, I> load(GeneralDataset<U, I> dataset, List<Tuple2<Integer, Integer>> list)
    {
        List<Tuple3<U, I, Double>> triplets = new ArrayList<>();
        FastUpdateableUserIndex<U> userIndex = dataset.getUserIndex();
        FastUpdateableItemIndex<I> itemIndex = dataset.getItemIndex();

        SimpleFastPreferenceData<U, I> datasetPrefData = dataset.getPrefData();

        int numrel = 0;

        for (Tuple2<Integer, Integer> tuple : list)
        {
            int uidx = tuple.v1;
            int iidx = tuple.v2;
            U u = userIndex.uidx2user(uidx);
            I i = itemIndex.iidx2item(iidx);

            if (datasetPrefData.numItems(uidx) > 0 && datasetPrefData.numUsers(iidx) > 0)
            {
                Optional<IdxPref> pref = datasetPrefData.getPreference(uidx, iidx);
                if (pref.isPresent())
                {
                    double value = pref.get().v2;
                    Tuple3<U, I, Double> t = new Tuple3<>(u, i, value);
                    triplets.add(t);

                    if (dataset.relevance.test(value))
                    {
                        numrel++;
                    }
                }
            }
        }

        SimpleFastPreferenceData<U, I> prefData = SimpleFastPreferenceData.load(triplets.stream(), userIndex, itemIndex);
        return new GeneralDataset<>(userIndex, itemIndex, prefData, numrel, dataset.relevance);
    }
}
