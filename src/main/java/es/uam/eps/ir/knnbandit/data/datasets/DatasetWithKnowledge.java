/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.data.datasets;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.userknowledge.fast.SimpleFastUserKnowledgePreferenceData;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple4;
import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;


/**
 * A dataset containing additional information on whether the users
 * knew the items before they rated them.
 * Example: cm100k
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class DatasetWithKnowledge<U, I> extends GeneralDataset<U, I>
{
    /**
     * Preference data with knowledge information about whether the users knew about the items before the rating.
     */
    protected final SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData;
    /**
     * Number of relevant (and known) ratings.
     */
    protected final int numRelKnown;

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param knowledgeData Preference data.
     * @param numRel        Number of relevant (user, item) pairs.
     * @param numRelKnown   Number of relevant known (user, item) pairs.
     */
    protected DatasetWithKnowledge(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData, int numRel, int numRelKnown, DoublePredicate relevance)
    {
        super(uIndex, iIndex, (SimpleFastPreferenceData<U, I>) knowledgeData.getPreferenceData(), numRel, relevance);
        this.knowledgeData = knowledgeData;
        this.numRelKnown = numRelKnown;
    }

    /**
     * Obtains a reduced dataset containing only the set of rating given by users to items they did know before
     * the recommendation.
     * @return the reduced dataset.
     */
    public OfflineDataset<U,I> getKnownDataset()
    {
        return new GeneralDataset<>(this.userIndex, this.itemIndex, this.getKnownPrefData(), this.getNumRelKnown(), this.relevance);
    }

    /**
     * Obtains a reduced dataset containing only the set of rating given by users to items they did not know before
     * the recommendation.
     * @return the reduced dataset.
     */
    public OfflineDataset<U,I> getUnknownDataset()
    {
        return new GeneralDataset<>(this.userIndex, this.itemIndex, this.getUnknownPrefData(), this.getNumRelUnknown(), this.relevance);
    }

    /**
     * Obtains a reduced dataset depending on the use we are going to make of the data.
     * @param dataUse the selection of the dataset.
     * @return the reduced dataset
     */
    public OfflineDataset<U,I> getDataset(KnowledgeDataUse dataUse)
    {
        switch (dataUse)
        {
            case ONLYKNOWN: return this.getKnownDataset();
            case ONLYUNKNOWN: return this.getUnknownDataset();
            default: return this;
        }
    }

    /**
     * Obtains the data previously known by the users.
     * @return the previously known ratings.
     */
    private SimpleFastPreferenceData<U, I> getKnownPrefData()
    {
        return (SimpleFastPreferenceData<U, I>) this.knowledgeData.getKnownPreferenceData();
    }
    /**
     * Obtains the data unknown by the users before the rating.
     * @return the previously unknown ratings.
     */
    private SimpleFastPreferenceData<U, I> getUnknownPrefData()
    {
        return (SimpleFastPreferenceData<U, I>) this.knowledgeData.getUnknownPreferenceData();
    }

    /**
     * Obtains the number of relevant and known ratings.
     * @return the number of relevant and known ratings.
     */
    public int getNumRelKnown()
    {
        return this.numRelKnown;
    }
    /**
     * Obtains the number of relevant and unknown ratings.
     * @return the number of relevant and unknown ratings.
     */
    public int getNumRelUnknown()
    {
        return this.numRel - this.numRelKnown;
    }

    @Override
    public String toString()
    {
        return "Users: " +
                this.numUsers() +
                "\nItems: " +
                this.numItems() +
                "\nNum. ratings: " +
                this.prefData.numPreferences() +
                "\nNum. relevant: " +
                this.getNumRel() +
                "\nNum. known ratings: " +
                this.knowledgeData.getKnownPreferenceData().numPreferences() +
                "\nNum. relevant known ratings: " +
                this.getNumRelKnown() +
                "\nNum. unknown ratings: " +
                this.knowledgeData.getUnknownPreferenceData().numPreferences() +
                "\nNum. relevant unknown ratings: " +
                this.getNumRelUnknown();
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
    public static <U, I> DatasetWithKnowledge<U, I> load(String filename, Parser<U> uParser, Parser<I> iParser, String separator, DoubleUnaryOperator weightFunction, DoublePredicate relevance) throws IOException
    {
        // Then, we read the ratings.
        Set<U> users = new HashSet<>();
        Set<I> items = new HashSet<>();
        List<Tuple4<U, I, Double, Boolean>> quartets = new ArrayList<>();
        int numrel = 0;
        int numrelknown = 0;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filename))))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] split = line.split(separator);
                U user = uParser.parse(split[0]);
                I item = iParser.parse(split[1]);
                double val = Parsers.dp.parse(split[2]);
                boolean known = split[3].equals("1");

                users.add(user);
                items.add(item);

                double rating = weightFunction.applyAsDouble(val);
                if (relevance.test(rating))
                {
                    numrel++;
                    if (known)
                    {
                        numrelknown++;
                    }
                }

                quartets.add(new Tuple4<>(user, item, rating, known));
            }
        }

        // Create the data.
        FastUpdateableUserIndex<U> uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        FastUpdateableItemIndex<I> iIndex = SimpleFastUpdateableItemIndex.load(items.stream());

        SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData = SimpleFastUserKnowledgePreferenceData.load(quartets.stream(), uIndex, iIndex);

        return new DatasetWithKnowledge<>(uIndex, iIndex, knowledgeData, numrel, numrelknown, relevance);
    }

    /**
     * Loads a dataset from another dataset.
     * @param dataset the original dataset.
     * @param list a list of (user,item) interactions.
     * @param <U> type of the users.
     * @param <I> type of the items.
     * @return the new dataset.
     */
    public static <U, I> DatasetWithKnowledge<U, I> load(DatasetWithKnowledge<U, I> dataset, List<Tuple2<Integer, Integer>> list)
    {
        List<Tuple4<U, I, Double, Boolean>> quartets = new ArrayList<>();
        FastUpdateableUserIndex<U> userIndex = dataset.userIndex;
        FastUpdateableItemIndex<I> itemIndex = dataset.itemIndex;

        SimpleFastUserKnowledgePreferenceData<U, I> datasetKnowledgeData = dataset.knowledgeData;

        int numrel = 0;
        int numrelknown = 0;

        for (Tuple2<Integer, Integer> tuple : list)
        {
            int uidx = tuple.v1;
            int iidx = tuple.v2;
            U u = userIndex.uidx2user(uidx);
            I i = itemIndex.iidx2item(iidx);

            if (datasetKnowledgeData.numItems(uidx) > 0 && datasetKnowledgeData.numUsers(iidx) > 0)
            {
                Optional<IdxPref> pref = datasetKnowledgeData.getPreference(uidx, iidx);
                if (pref.isPresent())
                {
                    double value = pref.get().v2;
                    boolean known = datasetKnowledgeData.getKnownPreference(uidx, iidx).isPresent();
                    Tuple4<U, I, Double, Boolean> t = new Tuple4<>(u, i, value, known);
                    quartets.add(t);

                    if (dataset.relevance.test(value))
                    {
                        numrel++;
                        if (known)
                        {
                            numrelknown++;
                        }
                    }
                }
            }
        }

        SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData = SimpleFastUserKnowledgePreferenceData.load(quartets.stream(), userIndex, itemIndex);
        return new DatasetWithKnowledge<>(userIndex, itemIndex, knowledgeData, numrel, numrelknown, dataset.relevance);
    }

    @Override
    public int getNumRatings()
    {
        return this.prefData.numPreferences();
    }

    @Override
    public Dataset<U,I> load(List<Pair<Integer>> pairs)
    {
        return DatasetWithKnowledge.load(this, pairs);
    }

}
