/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.io.Reader;
import es.uam.eps.ir.knnbandit.partition.Partition;
import es.uam.eps.ir.knnbandit.selector.io.IOSelector;
import es.uam.eps.ir.knnbandit.utils.Pair;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;

/**
 * Class for computing the statistics for training data.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class TrainingStatistics<U,I>
{
    /**
     * Selects the format for reading the training data.
     */
    private final IOSelector warmupIOSelector;

    /**
     * Constructor.
     * @param warmupIOSelector selects the format for reading the training data.
     */
    public TrainingStatistics(IOSelector warmupIOSelector)
    {
        this.warmupIOSelector = warmupIOSelector;
    }
    /**
     * Finds the statistics of the training set.
     * @param training the file containing the training data.
     * @param partition the partition strategy.
     * @param numSplits the number of splits.
     * @param percTrain the percentage of the warm-up data to use as training.
     */
    public void statistics(String training, Partition partition, int numSplits, double percTrain) throws IOException
    {
        // Read the training data.
        Reader reader = warmupIOSelector.getReader();
        InputStream input = warmupIOSelector.getInputStream(training);
        List<Pair<Integer>> train = reader.readFile(input);

        Dataset<U,I> dataset = this.getDataset();

        // Print the general data:
        System.out.println("General information: ");
        System.out.println("Users\tItems\tRatings\tRel.Ratings");
        System.out.println(dataset.numUsers() + "\t" + dataset.numItems() + "\t" + dataset.getNumRatings() + "\t" + dataset.getNumRel());

        // Then, for each split:
        System.out.println("Training");
        System.out.println("Num.Split\tNum.Recs\tRatings\tRel.Ratings");

        List<Integer> splitPoints;
        if(Double.isNaN(percTrain) || percTrain <= 0.0 || percTrain >= 1.0)
        {
            splitPoints = partition.split(dataset, train, numSplits);
        }
        else
        {
            splitPoints = new ArrayList<>();
            for(int i = 0; i < numSplits; ++i)
            {
                splitPoints.add(partition.split(dataset, train, percTrain*(i+1.0)));
            }
        }

        for (int part = 0; part < numSplits; ++part)
        {
            int val = splitPoints.get(part);
            List<Pair<Integer>> partTrain = train.subList(0, val);

            long trainCount = partTrain.stream().filter(t ->
            {
                Optional<Double> pref = dataset.getPreference(t.v1(), t.v2());
                return pref.isPresent();
            }).count();

            long trainRelCount = partTrain.stream().filter(t ->
            {
                Optional<Double> pref = dataset.getPreference(t.v1(), t.v2());
                return pref.isPresent() && dataset.getRelevanceChecker().test(t.v2());
            }).count();

            System.out.println((part + 1) + "\t" + val + "\t" + trainCount + "\t" + trainRelCount);
        }
    }

    /**
     * Obtains the dataset for the analysis.
     * @return the dataset.
     */
    protected abstract Dataset<U,I> getDataset();
}
