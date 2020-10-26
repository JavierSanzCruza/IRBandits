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

import es.uam.eps.ir.knnbandit.UntieRandomNumberReader;
import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import org.jooq.lambda.tuple.Tuple2;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Class for performing the validation of several approaches
 *
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class Validation<U,I>
{
    /**
     * Applies validation over a set of algorithms:
     * @param algorithms a file containing the algorithm configuration.
     * @param output folder in which to store the outcomes.
     * @param endCond end condition for the recommendation loop.
     * @param resume true if we want to retrieve previous values.
     * @param k the number of times we want to execute each approach.
     *
     * @throws IOException if something fails while reading / writing.
     * @throws UnconfiguredException if the algorithm configurator is not properly configured.
     */
    public void validate(String algorithms, String output, Supplier<EndCondition> endCond, boolean resume, int k) throws IOException, UnconfiguredException
    {
        // Obtains the dataset and the metrics.
        Dataset<U,I> dataset = this.getDataset();
        Map<String, Supplier<CumulativeMetric<U,I>>> metrics = this.getMetrics();
        int interval = Integer.MAX_VALUE;

        // Select the algorithms
        long a = System.currentTimeMillis();
        AlgorithmSelector<U, I> algorithmSelector = new AlgorithmSelector<>();
        algorithmSelector.configure(dataset.getRelevanceChecker());
        algorithmSelector.addFile(algorithms, false);
        Map<String, InteractiveRecommenderSupplier<U, I>> recs = algorithmSelector.getRecs();
        long b = System.currentTimeMillis();

        System.out.println("Recommenders prepared (" + (b - a) + " ms.)");

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

        // Store the values for each algorithm.
        Map<String, Map<String, Double>> auxiliarValues = new ConcurrentHashMap<>();
        metrics.keySet().forEach(metric -> auxiliarValues.put(metric, new HashMap<>()));

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
            Map<String, Double> averagedLastIteration = new HashMap<>();
            this.getMetrics().keySet().forEach(metricName -> averagedLastIteration.put(metricName, 0.0));

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
                executor.executeWithoutWarmup(loop, fileName, resume, interval);
                int currentIter = loop.getCurrentIter();
                if(currentIter > 0) // if at least one iteration has been recorded:
                {
                    Map<String, Double> metricValues = loop.getMetricValues();
                    for(String metric : this.getMetrics().keySet())
                    {
                        double value = metricValues.get(metric);
                        if(i == 0)
                        {
                            averagedLastIteration.put(metric, value);
                        }
                        else
                        {
                            double lastValue = averagedLastIteration.get(metric);
                            double newValue = lastValue + (value - lastValue)/(i + 1.0);
                            averagedLastIteration.put(metric, newValue);
                        }
                    }

                    if(i == 0) // Store the information about the number of iterations.
                    {
                        averagedLastIteration.put("numIter", currentIter+0.0);
                    }
                    else
                    {
                        double lastIter = averagedLastIteration.get("numIter");
                        double newIter = lastIter + (currentIter - lastIter)/(i + 1.0);
                        averagedLastIteration.put("numIter", newIter);
                    }
                }

                bbb = System.nanoTime();
                System.out.println("Algorithm " + name + " (" + i + ") " + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
            }

            averagedLastIteration.forEach((metric, value) -> auxiliarValues.get(metric).put(name,value));
            long bbb = System.nanoTime();
            System.out.println("Algorithm " + name + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
        });

        // Now, for each metric, we generate a ranking, and we write it into a file:
        for(String metric : auxiliarValues.keySet())
        {
            PriorityQueue<Tuple2<String, Double>> ranking = new PriorityQueue<>(recs.size(), (Tuple2<String, Double> x, Tuple2<String, Double> y) -> Double.compare(y.v2, x.v2));
            auxiliarValues.get(metric).forEach((algorithm, value) -> ranking.add(new Tuple2<>(algorithm, value)));
            File aux = new File(algorithms);
            String rankingname = aux.getName();

            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output + rankingname + "-ranking.txt"))))
            {
                bw.write("Algorithm\t" + metric);
                while (!ranking.isEmpty())
                {
                    Tuple2<String, Double> alg = ranking.poll();
                    bw.write("\n" + alg.v1 + "\t" + alg.v2);
                }
            }
        }
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
