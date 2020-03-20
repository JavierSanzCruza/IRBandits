/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.general.cm100k;

import es.uam.eps.ir.knnbandit.UntieRandomNumber;
import es.uam.eps.ir.knnbandit.UntieRandomNumberReader;
import es.uam.eps.ir.knnbandit.data.datasets.DatasetWithKnowledge;
import es.uam.eps.ir.knnbandit.io.Writer;
import es.uam.eps.ir.knnbandit.main.AuxiliarMethods;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.knnbandit.recommendation.RecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.NoLimitsEndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.NumIterEndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.PercentagePositiveRatingsEndCondition;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
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
     *                  <li><b>Knowledge data use:</b>ALL if we want to use all the ratings, KNOWN if we want to use only the known ones, UNKNOWN otherwise</li>
     *             </ol>
     *             Also, as optional arguments, we have one:
     *             <ol>
     *                  <li><b>-k value :</b> The number of times each individual approach has to be executed (by default: 1)</li>
     *             </ol>
     */
    public static void main(String[] args) throws IOException, UnconfiguredException
    {
        if (args.length < 8)
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
            System.err.println("\tknowledge: ALL if we want to use all the ratings, KNOWN if we want to use only the known ones, UNKNOWN otherwise");
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

        KnowledgeDataUse dataUse = KnowledgeDataUse.valueOf(args[7]);
        int auxK = 1;

        for(int i = 8; i < args.length; ++i)
        {
            if(args[i].equals("-k"))
            {
                auxK = Parsers.ip.parse(args[++i]);
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
        DatasetWithKnowledge<Long, Long> dataset = DatasetWithKnowledge.load(input, Parsers.lp, Parsers.lp, "::", weightFunction, relevance);
        System.out.println("Read the whole data");
        System.out.println(dataset.toString());


        int interval = Integer.MAX_VALUE;

        // Initialize the metrics to compute.
        Map<String, Supplier<CumulativeMetric<Long, Long>>> metrics = new HashMap<>();
        metrics.put("recall", () -> new CumulativeRecall<>(dataset.getPrefData(), dataset.getNumRel(), 0.5));
        List<String> metricNames = new ArrayList<>(metrics.keySet());

        // Select the algorithms
        // Select the algorithms
        long a = System.currentTimeMillis();
        AlgorithmSelector<Long, Long> algorithmSelector = new AlgorithmSelector<>();
        algorithmSelector.configure(dataset.getUserIndex(), dataset.getItemIndex(), dataset.getPrefData(), realThreshold, dataset.getKnowledgeData(), dataUse);
        algorithmSelector.addFile(algorithms);
        Map<String, InteractiveRecommender<Long, Long>> recs = algorithmSelector.getRecs();
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
        Map<String, Double> auxiliarValues = new ConcurrentHashMap<>();

        // Run each algorithm
        recs.entrySet().parallelStream().forEach((entry) ->
        {
            String name = entry.getKey();
            InteractiveRecommender<Long, Long> rec = entry.getValue();

            UntieRandomNumberReader rngSeedGen = new UntieRandomNumberReader();
            // And the metrics
            Map<String, CumulativeMetric<Long, Long>> localMetrics = new HashMap<>();
            metricNames.forEach(metricName -> localMetrics.put(metricName, metrics.get(metricName).get()));

            // Configure and initialize the recommendation loop:
            System.out.println("Starting algorithm " + name);
            long aaa = System.nanoTime();
            EndCondition endcond = iterationsStop ? (auxIter == 0.0 ? new NoLimitsEndCondition() : new NumIterEndCondition(numIter)) : new PercentagePositiveRatingsEndCondition(dataset.getNumRel(), auxIter, 0.5);

            Map<String, Double> averagedLastIteration = new HashMap<>();
            metricNames.forEach(metricName -> averagedLastIteration.put(metricName, 0.0));

            double maxIter = 0;
            // Execute each recommender k times.
            for(int i = 0; i < k; ++i)
            {
                int rngSeed = rngSeedGen.nextSeed();

                // Create the recommendation loop:
                RecommendationLoop<Long, Long> loop = new RecommendationLoop<>(dataset.getUserIndex(), dataset.getItemIndex(), dataset.getPrefData(), rec, localMetrics, endcond, rngSeed, false);
                loop.init(false);
                long bbb = System.nanoTime();
                System.out.println("Algorithm " + name + " has been initialized (" + (bbb - aaa) / 1000000.0 + " ms.)");

                Map<String, List<Double>> metricValues = new HashMap<>();
                metricNames.forEach(metricName -> metricValues.put(metricName, new ArrayList<>()));

                try
                {
                    List<Tuple3<Integer, Integer, Long>> list = new ArrayList<>();
                    String fileName = outputFolder + name + "_" + i + ".txt";
                    if (resume)
                    {
                        list = AuxiliarMethods.retrievePreviousIterations(fileName);
                    }

                    Writer writer = new Writer(outputFolder + name + "_" + i + ".txt", metricNames);
                    if (resume && !list.isEmpty())
                    {
                        metricValues.putAll(AuxiliarMethods.updateWithPrevious(loop, list, writer, interval));
                    }

                    int currentIter = AuxiliarMethods.executeRemaining(loop, writer, interval, metricValues);
                    maxIter = maxIter + (currentIter - maxIter)/(i+1.0);
                    writer.close();
                }
                catch (IOException ioe)
                {
                    System.err.println("ERROR: Some error occurred when executing algorithm " + name + " (" + i + ") ");
                }

                for(String metric : metricNames)
                {
                    List<Double> values = metricValues.get(metric);
                    int auxSize = values.size();
                    if(i == 0)
                    {
                        averagedLastIteration.put(metric, values.get(auxSize-1));
                    }
                    else
                    {
                        double lastIter = averagedLastIteration.get(metric);
                        double newValue = lastIter + (values.get(auxSize-1) - lastIter)/(i + 1.0);
                        averagedLastIteration.put(metric, newValue);
                    }
                }

                System.out.println("Algorithm " + name + " (" + i + ") " + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
            }

            if(iterationsStop) auxiliarValues.put(name, averagedLastIteration.get("recall"));
            else auxiliarValues.put(name, maxIter);

            long bbb = System.nanoTime();
            System.out.println("Algorithm " + name + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
        });

        PriorityQueue<Tuple2<String, Double>> ranking = new PriorityQueue<>(recs.size(), (Tuple2<String, Double> x, Tuple2<String, Double> y) -> Double.compare(y.v2,x.v2));
        auxiliarValues.forEach((algorithm, value) -> ranking.add(new Tuple2<>(algorithm, value)));

        // Write the algorithm ranking
        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output + "ranking.txt"))))
        {
            bw.write("Algorithm\t" + (iterationsStop ? "recall@" + numIter : "numIters@" + auxIter));
            while(!ranking.isEmpty())
            {
                Tuple2<String, Double> alg = ranking.poll();
                bw.write("\n" + alg.v1 + "\t" + alg.v2 + "\n");
            }
        }
    }
}
