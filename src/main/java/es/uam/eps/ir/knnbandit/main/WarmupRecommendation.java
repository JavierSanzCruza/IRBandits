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
import es.uam.eps.ir.knnbandit.io.Reader;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.partition.Partition;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.json.JSONArray;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * Class for executing the interactive recommendation loop with warmup.
 *
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class WarmupRecommendation<U,I>
{
    /**
     * Applies recommendation with warmup over a set of algorithms:
     * @param algorithms a file containing the algorithm configuration. It should contain as many lines
     * @param output folder in which to store the outcomes.
     * @param endCond end condition for the recommendation loop.
     * @param resume true if we want to retrieve previous values.
     * @param warmupData file containing a previous warmup execution.
     * @param partition the partition strategy.
     * @param numParts the number of parts.
     * @param k the number of times we want to execute each approach.
     * @param interval indicates the number of times the output is written in a resume file.
     *
     * @throws IOException if something fails while reading / writing.
     */
    public void recommend(String algorithms, String output, Supplier<EndCondition> endCond, boolean resume, String warmupData, Partition partition, int numParts, double percTrain, int k, int interval) throws IOException
    {
        // Obtains the dataset and the metrics.
        Dataset<U,I> dataset = this.getDataset();
        UntieRandomNumber.configure(resume, output, k);

        // Select the algorithms
        long a = System.currentTimeMillis();

        // First, we read the file:
        StringBuilder jSon = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(algorithms))))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                jSon.append(line);
                jSon.append("\n");
            }
        }
        JSONArray array = new JSONArray(jSon.toString());

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

        // Read the warmup data
        Reader reader = new Reader();
        List<Pair<Integer>> train = reader.read(warmupData, "\t", true);
        System.out.println("The warmup data has been read");

        List<Integer> splitPoints;
        if(Double.isNaN(percTrain) || percTrain <= 0.0 || percTrain >= 1.0)
        {
            splitPoints = partition.split(dataset, train, numParts);
        }
        else
        {
            splitPoints = new ArrayList<>();
            for(int i = 0; i < numParts; ++i)
            {
                splitPoints.add(partition.split(dataset, train, percTrain*(i+1.0)));
            }
        }

        IntStream.range(0, numParts).parallel().forEach(part ->
        {
            try
            {
                String currentOutputFolder = outputFolder + part + File.separator;
                File f = new File(currentOutputFolder);
                if (!f.exists())
                {
                    if (!f.mkdirs())
                    {
                        System.err.println("ERROR: Invalid output folder");
                        return;
                    }
                }

                System.out.println("Started part " + (part + 1) + "/" + numParts);

                AlgorithmSelector<U, I> algorithmSelector = new AlgorithmSelector<>();
                algorithmSelector.configure(dataset.getRelevanceChecker());
                algorithmSelector.addList(array.getJSONArray(part), false);
                Map<String, InteractiveRecommenderSupplier<U, I>> recs = algorithmSelector.getRecs();
                long b = System.currentTimeMillis();

                // Obtain the lists: only the first "numParts" algorithms shall be considered
                System.out.println("Recommenders for part " + (part + 1) + " prepared (" + (b - a) + " ms.)");

                List<Pair<Integer>> partTrain = train.subList(0, splitPoints.get(part));
                Warmup warmup = this.getWarmup(partTrain);
                int notRel = warmup.getNumRel();

                System.out.println("Training: " + splitPoints.get(part) + " recommendations (" + (part + 1) + "/" + numParts + ")");
                System.out.println(dataset.toString());
                System.out.println("Training recommendations: " + splitPoints.get(part) + " (" + (part + 1) + "/" + numParts + ")");
                System.out.println("Relevant recommendations (with training): " + (dataset.getNumRel() - notRel));

                // Store the values for each algorithm.
                Map<String, List<Double>> averagedValues = new HashMap<>();
                IntList counter = new IntArrayList();

                this.getMetrics().keySet().forEach(metricName -> averagedValues.put(metricName, new ArrayList<>()));

                // Run each algorithm
                recs.forEach((name, rec) ->
                {
                    // Get the recommender:
                    // Obtain the corresponding random numbers:
                    UntieRandomNumberReader rngSeedGen = new UntieRandomNumberReader();
                    // Configure and initialize the recommendation loop:
                    System.out.println("Starting algorithm " + name + " for the " + (part + 1) + "/" + numParts + " part.");
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
                        System.out.println("Algorithm " + name + " (" + i + ") " + " for the " + (part + 1) + "/" + numParts + " split starting (" + (bbb - aaa) / 1000000.0 + " ms.)");

                        // Create the recommendation loop: in this case, a general offline dataset loop
                        FastRecommendationLoop<U, I> loop = this.getRecommendationLoop(rec, endCond.get(), rngSeed);
                        // Execute the loop:
                        Executor<U, I> executor = new Executor<>();
                        String fileName = currentOutputFolder + name + "_" + i + ".txt";
                        Map<String, List<Double>> metricValues = executor.executeWithWarmup(loop, fileName, resume, interval, warmup);
                        int currentIter = loop.getCurrentIter();
                        if (currentIter > 0) // if at least one iteration has been recorded:
                        {
                            for (String metric : this.getMetrics().keySet())
                            {
                                List<Double> values = metricValues.get(metric);
                                if (i == 0)
                                {
                                    int auxSize = values.size();
                                    averagedValues.get(metric).addAll(values);
                                    IntStream.range(0, auxSize).forEach(x -> counter.add(1));
                                }
                                else
                                {
                                    List<Double> oldVals = averagedValues.get(metric);
                                    int currentSize = oldVals.size();
                                    int auxSize = values.size();
                                    for (int j = 0; j < auxSize; ++j)
                                    {
                                        if (j >= currentSize)
                                        {
                                            oldVals.add(values.get(j));
                                            counter.add(1);
                                        }
                                        else
                                        {
                                            double oldM = oldVals.get(j);
                                            double averaged = oldM + (values.get(j) - oldM) / (counter.get(j) + 1);
                                            oldVals.set(j, averaged);
                                            counter.set(j, counter.get(j) + 1);
                                        }
                                    }
                                }
                            }

                            if (i == 0) // Store the information about the number of iterations.
                            {
                                averagedLastIteration.put("numIter", currentIter + 0.0);
                            }
                            else
                            {
                                double lastIter = averagedLastIteration.get("numIter");
                                double newIter = lastIter + (currentIter - lastIter) / (i + 1.0);
                                averagedLastIteration.put("numIter", newIter);
                            }
                        }

                        bbb = System.nanoTime();
                        System.out.println("Algorithm " + name + " (" + i + ") " + " for the " + (part+1) + "/" + numParts + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
                    }

                    int size = counter.size();
                    try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output + name + "-summary.txt"))))
                    {
                        bw.write("Iteration");
                        for (String metric : averagedValues.keySet())
                        {
                            bw.write("\t" + metric);
                        }

                        for (int i = 0; i < size; ++i)
                        {
                            bw.write("\n" + (interval * (i + 1)));
                            for (String metric : averagedValues.keySet())
                            {
                                bw.write("\t" + averagedValues.get(metric).get(i));
                            }
                        }
                    }
                    catch (IOException ioe)
                    {
                        System.err.println("ERROR: Something ocurred while writing the summary for algorithm " + name);
                    }

                    long bbb = System.nanoTime();
                    System.out.println("Algorithm " + name + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
                });
            }
            catch(UnconfiguredException ioe)
            {
                System.err.println("ERROR: Something occurred while reading the algorithm list");
            }
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

    /**
     * Obtains the warmup
     * @param trainData the training data.
     * @return the warmup.
     */
    protected abstract Warmup getWarmup(List<Pair<Integer>> trainData);
}
