/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.general.movielens;

import es.uam.eps.ir.knnbandit.UntieRandomNumber;
import es.uam.eps.ir.knnbandit.UntieRandomNumberReader;
import es.uam.eps.ir.knnbandit.data.datasets.GeneralOfflineDataset;
import es.uam.eps.ir.knnbandit.main.Executor;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.GeneralOfflineDatasetRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.NoLimitsEndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.NumIterEndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.PercentagePositiveRatingsEndCondition;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import org.jooq.lambda.tuple.Tuple2;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;

/**
 * Class for executing contact recommender systems in simulated interactive loops (with training)
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class Validation
{
    /**
     * @param args Execution arguments
     *             <ol>
     *                  <li><b>Algorithms:</b> the recommender systems to apply validation for</li>
     *                  <li><b>Input:</b> Full preference data</li>
     *                  <li><b>Output:</b> Folder in which to store the output</li>
     *                  <li><b>Num. Iter:</b> Number of iterations for the validation. 0 if we want to run out of recommendable items</li>
     *                  <li><b>Resume:</b> True if we want to resume previous executions, false to overwrite them</li>
     *                  <li><b>Threshold:</b> Relevance threshold</li>
     *                  <li><b>Use ratings:</b>True if we want to take the true rating value, false if we want to binarize them</li>
     *             </ol>
     *             Also, as optional arguments, we have one:
     *             <ol>
     *                  <li><b>-k value :</b> The number of times each individual approach has to be executed (by default: 1)</li>
     *             </ol>
     */
    public static void main(String[] args) throws IOException, UnconfiguredException
    {
        if (args.length < 7)
        {
            System.err.println("ERROR: Invalid arguments");
            System.err.println("Usage:");
            System.err.println("\tAlgorithms: recommender systems list");
            System.err.println("\tInput: preference data input");
            System.err.println("\tOutput: folder in which to store the output");
            System.err.println("\tNum. Iter.: number of iterations. 0 if we want to run until we run out of recommendable items");
            System.err.println("\tresume: true if we want to resume previous executions, false if we want to overwrite");
            System.err.println("\tthreshold: true if the graph is directed, false otherwise");
            System.err.println("\tuse ratings: true if we want to recommend reciprocal edges, false otherwise");
            System.err.println("Optional arguments:");
            System.err.println("\t-k value : The number of times each individual approach has to be executed (by default: 1)");
            return;
        }

        // First, read the program arguments.
        String algorithms = args[0];
        String input = args[1];
        String output = args[2];

        // Defining the stop condition
        Double auxIter = Parsers.dp.parse(args[3]);
        boolean iterationsStop = auxIter == 0.0 || auxIter >= 1.0;
        int numIter = (iterationsStop && auxIter > 1.0) ? auxIter.intValue() : Integer.MAX_VALUE;

        // Checking whether we have to resume the execution or not.
        boolean resume = args[4].equalsIgnoreCase("true");

        // General recommendation specific arguments
        double threshold = Parsers.dp.parse(args[5]);
        boolean useRatings = args[6].equalsIgnoreCase("true");

        int auxK = 1;

        for (int i = 7; i < args.length; ++i)
        {
            if ("-k".equals(args[i]))
            {
                ++i;
                auxK = Parsers.ip.parse(args[i]);
            }
        }

        int k = auxK;

        System.out.println("Read parameters");

        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x :
                (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
        double realThreshold = useRatings ? threshold : 0.5;

        // Configure the random number generator for ties.
        UntieRandomNumber.configure(resume, output, k);

        // Read the whole ratings:
        GeneralOfflineDataset<Long, Long> dataset = GeneralOfflineDataset.load(input, Parsers.lp, Parsers.lp, "::", weightFunction, relevance);
        System.out.println("Read the whole data");
        System.out.println(dataset.toString());

        // Then, initialize the metrics to compute:
        Map<String, Supplier<CumulativeMetric<Long, Long>>> metrics = new HashMap<>();
        metrics.put("recall", () -> new CumulativeRecall<>(dataset.getNumRel(), 0.5));
        List<String> metricNames = new ArrayList<>(metrics.keySet());

        // Select the algorithms:
        long a = System.currentTimeMillis();
        AlgorithmSelector<Long, Long> algorithmSelector = new AlgorithmSelector<>();
        algorithmSelector.configure(realThreshold);
        algorithmSelector.addFile(algorithms);
        Map<String, InteractiveRecommenderSupplier<Long, Long>> recs = algorithmSelector.getRecs();
        long b = System.currentTimeMillis();

        System.out.println("Recommenders prepared (" + (b - a) + " ms.)");

        int interval = Integer.MAX_VALUE;

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
        Map<String, Double> auxiliarValues = new ConcurrentHashMap<>();

        // For each recommender:
        recs.entrySet().parallelStream().forEach((entry) ->
        {
            // Obtain its name and supplier:
            String name = entry.getKey();
            InteractiveRecommenderSupplier<Long, Long> rec = entry.getValue();

            // Read the corresponding random number seed
            UntieRandomNumberReader rngSeedGen = new UntieRandomNumberReader();
            // and the particular metric values.
            Map<String, CumulativeMetric<Long, Long>> localMetrics = new HashMap<>();
            metricNames.forEach(metricName -> localMetrics.put(metricName, metrics.get(metricName).get()));

            // Configure the recommendation loop:

            System.out.println("Starting algorithm " + name);
            long aaa = System.nanoTime();
            // end condition
            EndCondition endcond = iterationsStop ? (auxIter == 0.0 ? new NoLimitsEndCondition() : new NumIterEndCondition(numIter)) : new PercentagePositiveRatingsEndCondition(dataset.getNumRel(), auxIter, 0.5);
            Map<String, Double> averagedLastIteration = new HashMap<>();
            metricNames.forEach(metricName -> averagedLastIteration.put(metricName, 0.0));

            double maxIter = 0;
            // Execute each recommender k times.
            for (int i = 0; i < k; ++i)
            {
                int rngSeed = rngSeedGen.nextSeed();
                long bbb = System.nanoTime();
                System.out.println("Algorithm " + name + " (" + i + ") " + " starting (" + (bbb - aaa) / 1000000.0 + " ms.)");
                // Create the recommendation loop: in this case, a general offline dataset loop
                FastRecommendationLoop<Long, Long> loop = new GeneralOfflineDatasetRecommendationLoop<>(dataset, rec, localMetrics, endcond, rngSeed);
                Executor<Long, Long> executor = new Executor<>();
                String fileName = outputFolder + name + "_" + i + ".txt";
                int currentIter = executor.executeWithoutWarmup(loop, fileName, resume, interval);
                if(currentIter > 0)
                {
                    Map<String, Double> metricValues = loop.getMetricValues();
                    for (String metric : metricNames)
                    {
                        double value = metricValues.get(metric);
                        if (i == 0)
                        {
                            averagedLastIteration.put(metric, value);
                        }
                        else
                        {
                            double lastIter = averagedLastIteration.get(metric);
                            double newValue = lastIter + (value - lastIter) / (i + 1.0);
                            averagedLastIteration.put(metric, newValue);
                        }
                    }
                }

                bbb = System.nanoTime();
                System.out.println("Algorithm " + name + " (" + i + ") " + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
            }

            if (iterationsStop)
            {
                auxiliarValues.put(name, averagedLastIteration.get("recall"));
            }
            else
            {
                auxiliarValues.put(name, maxIter);
            }

            long bbb = System.nanoTime();
            System.out.println("Algorithm " + name + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
        });

        PriorityQueue<Tuple2<String, Double>> ranking = new PriorityQueue<>(recs.size(), (Tuple2<String, Double> x, Tuple2<String, Double> y) -> Double.compare(y.v2, x.v2));
        auxiliarValues.forEach((algorithm, value) -> ranking.add(new Tuple2<>(algorithm, value)));

        File aux = new File(algorithms);
        String rankingname = aux.getName();

        // Write the algorithm ranking
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output + rankingname + "-ranking.txt"))))
        {
            bw.write("Algorithm\t" + (iterationsStop ? "recall@" + numIter : "numIters@" + auxIter));
            while (!ranking.isEmpty())
            {
                Tuple2<String, Double> alg = ranking.poll();
                bw.write("\n" + alg.v1 + "\t" + alg.v2);
            }
        }
    }
}
