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

import es.uam.eps.ir.knnbandit.UntieRandomNumber;
import es.uam.eps.ir.knnbandit.UntieRandomNumberReader;
import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.io.*;
import java.util.*;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * Class for performing interactive recommendations.
 *
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class Recommendation<U,I>
{
    /**
     * Applies an interactive recommendation loop for different algorithms.
     * @param algorithms a file containing the algorithm configuration.
     * @param output folder in which to store the outcomes.
     * @param endCond end condition for the recommendation loop.
     * @param resume true if we want to retrieve previous values.
     * @param k the number of times we want to execute each approach.
     * @param interval time points to summarize.
     *
     * @throws IOException if something fails while reading / writing.
     * @throws UnconfiguredException if the algorithm configurator is not properly configured.
     */
    public void recommend(String algorithms, String output, Supplier<EndCondition> endCond, boolean resume, int k, int interval) throws IOException, UnconfiguredException
    {
        // Obtains the dataset and the metrics.
        Dataset<U,I> dataset = this.getDataset();

        // Select the algorithms
        long a = System.currentTimeMillis();
        AlgorithmSelector<U, I> algorithmSelector = new AlgorithmSelector<>();
        algorithmSelector.configure(dataset.getRelevanceChecker());
        algorithmSelector.addFile(algorithms, false);
        Map<String, InteractiveRecommenderSupplier<U, I>> recs = algorithmSelector.getRecs();
        long b = System.currentTimeMillis();

        System.out.println("Recommenders prepared (" + (b - a) + " ms.)");

        UntieRandomNumber.configure(resume, output, k);

        // If it does not exist, create the directory in which to store the recommendation.
        String outputFolder = output + File.separator;
        File folder = new File(outputFolder);
        if (!folder.exists())
        {
            if (!folder.mkdirs())
            {
                System.err.println("ERROR: Invalid output folder");
                return;
            }
        }
        UntieRandomNumber.configure(resume, output, k);

        // Run each algorithm
        recs.entrySet().parallelStream().forEach((entry) ->
        {
            // Get the recommender:
            String name = entry.getKey();
            InteractiveRecommenderSupplier<U, I> rec = entry.getValue();
            // Obtain the corresponding random numbers:
            UntieRandomNumberReader rngSeedGen = new UntieRandomNumberReader();
            // Configure and initialize the recommendation loop:
            System.out.println("Starting algorithm " + name);
            long aaa = System.nanoTime();
            // Create a map storing the average values for each metric:
            Map<String, List<Double>> averagedValues = new HashMap<>();
            IntList counter = new IntArrayList();

            this.getMetrics().keySet().forEach(metricName -> averagedValues.put(metricName, new ArrayList<>()));

            // Execute each recommender k times.
            for (int i = 0; i < k; ++i)
            {
                // Obtain the random seed:
                int rngSeed = rngSeedGen.nextSeed();
                long bbb = System.nanoTime();
                System.out.println("Algorithm " + name + " (" + i + ") " + " starting (" + (bbb - aaa) / 1000000.0 + " ms.)");

                // Create the recommendation loop: in this case, a general offline dataset loop
                FastRecommendationLoop<U,I> loop = this.getRecommendationLoop(rec, endCond.get(), rngSeed);
                // Execute the loop:
                Executor<U, I> executor = new Executor<>();
                String fileName = outputFolder + name + "_" + i + ".txt";
                Map<String, List<Double>> metricValues = executor.executeWithoutWarmup(loop, fileName, resume, interval);
                int currentIter = loop.getCurrentIter();
                if(currentIter > 0) // if at least one iteration has been recorded:
                {
                    int currentSize = counter.size();
                    String someMetric = new TreeSet<>(getMetrics().keySet()).first();
                    int auxSize = metricValues.get(someMetric).size();

                    if(auxSize > currentSize)
                    {
                        IntStream.range(currentSize, auxSize).forEach(j -> counter.add(1));
                    }
                    IntStream.range(0, Math.min(currentSize, auxSize)).forEach(j -> counter.set(j, counter.get(j)+1));

                    //Map<String, Double> metricValues = loop.getMetricValues();
                    for(String metric : this.getMetrics().keySet())
                    {
                        List<Double> newVals = metricValues.get(metric);

                        if(i == 0)
                        {
                            averagedValues.get(metric).addAll(newVals);
                        }
                        else
                        {
                            List<Double> oldVals = averagedValues.get(metric);
                            for(int j = 0; j < auxSize; ++j)
                            {
                                if(j >= currentSize)
                                {
                                    averagedValues.get(metric).add(newVals.get(j));
                                }
                                else
                                {
                                    double oldM = oldVals.get(j);
                                    double averaged = oldM + (newVals.get(j) - oldM) / (counter.get(j));
                                    oldVals.set(j, averaged);
                                }
                            }
                        }
                    }
                }

                bbb = System.nanoTime();
                System.out.println("Algorithm " + name + " (" + i + ") " + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
            }

            int size = counter.size();
            try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output + name + "-summary.txt"))))
            {
                bw.write("Iteration");
                for(String metric : averagedValues.keySet())
                {
                    bw.write("\t" + metric);
                }
                System.out.println("Size: " + size);

                for(int i = 0; i < size; ++i)
                {
                    bw.write("\n" + (interval*(i+1)));
                    for(String metric : averagedValues.keySet())
                    {
                        bw.write("\t" + averagedValues.get(metric).get(i));
                    }
                }
            }
            catch(IOException ioe)
            {
                System.err.println("ERROR: Something ocurred while writing the summary for algorithm " + name);
            }

            long bbb = System.nanoTime();
            System.out.println("Algorithm " + name + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
        });
    }

    /**
     * Obtains the dataset.
     * @return the dataset used during the validation.
     */
    protected abstract Dataset<U,I> getDataset();

    /**
     * Obtains the recommendation loop.
     * @param rec the recommender supplier.
     * @param endCond the ending condition for the loop.
     * @param rngSeed the random number generator seed.
     * @return a configured recommendation loop.
     */
    protected abstract FastRecommendationLoop<U, I> getRecommendationLoop(InteractiveRecommenderSupplier<U,I> rec, EndCondition endCond, int rngSeed);

    /**
     * Obtains the metrics.
     * @return a map with supplier for the metrics.
     */
    protected abstract Map<String, Supplier<CumulativeMetric<U,I>>> getMetrics();
}
