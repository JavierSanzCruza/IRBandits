/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.general.foursquare;

import es.uam.eps.ir.knnbandit.UntieRandomNumber;
import es.uam.eps.ir.knnbandit.UntieRandomNumberReader;
import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.io.Reader;
import es.uam.eps.ir.knnbandit.io.Writer;
import es.uam.eps.ir.knnbandit.main.AuxiliarMethods;
import es.uam.eps.ir.knnbandit.main.Initializer;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.partition.Partition;
import es.uam.eps.ir.knnbandit.partition.RelevantPartition;
import es.uam.eps.ir.knnbandit.partition.UniformPartition;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.RecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.NoLimitsEndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.NumIterEndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.PercentagePositiveRatingsEndCondition;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2od;
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
public class WarmupValidation
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
     *                  <li><b>Training data:</b>File containing the training data (a previous execution of a recommender over the cold start problem)</li>
     *                  <li><b>Num. partitions:</b> Number of training partitions we are going to use. Ex: if this argument is equal to 5, we will
     *                         execute the loop 5 times: one with 20% of the training, one with 40%, etc. </li>
     *             </ol>
     *             Also, as optional arguments, we have one:
     *             <ol>
     *                  <li><b>-k value :</b> The number of times each individual approach has to be executed (by default: 1)</li>
     *             </ol>
     */
    public static void main(String[] args) throws IOException, UnconfiguredException
    {
        if (args.length < 9)
        {
            System.err.println("ERROR:iInvalid arguments");
            System.err.println("Usage:");
            System.err.println("\tAlgorithms: recommender systems list");
            System.err.println("\tInput: preference data input");
            System.err.println("\tOutput: folder in which to store the output");
            System.err.println("\tNum. Iter.: number of iterations. 0 if we want to run until we run out of recommendable items");
            System.err.println("\tresume: true if we want to resume previous executions, false if we want to overwrite");
            System.err.println("\tthreshold: true if the graph is directed, false otherwise");
            System.err.println("\tuse ratings: true if we want to recommend reciprocal edges, false otherwise");
            System.err.println("\tTraining data: file containing the training data");
            System.err.println("\tNum. partitions: number of partitions to make with the training data");
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


        String trainingData = args[7];
        int auxNumParts = Parsers.ip.parse(args[8]);
        boolean relevantPartition = auxNumParts < 0;
        int numParts = Math.abs(auxNumParts);

        double percTrain = Parsers.dp.parse(args[9]);
        int auxK = 1;

        // If indicated as so, retrieve the number of repetitions.
        for(int i = 10; i < args.length; ++i)
        {
            if(args[i].equals("-k"))
            {
                auxK = Parsers.ip.parse(args[++i]);
            }
        }

        int k = auxK;
        int interval = Integer.MAX_VALUE;

        System.out.println("Read parameters");

        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x :
                (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
        double realThreshold = useRatings ? threshold : 0.5;

        // Configure the random number generator for ties.
        UntieRandomNumber.configure(resume, output, k);

        // Read the whole ratings:
        Dataset<Long, String> dataset = Dataset.load(input, Parsers.lp, Parsers.sp, "::", weightFunction, relevance);
        System.out.println("Read the whole data");
        System.out.println(dataset.toString());

        // Read the training data
        Reader reader = new Reader();
        List<Tuple2<Integer, Integer>> train = reader.read(trainingData, "\t", true);
        System.out.println("Read the training data");

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

        Partition partition = relevantPartition ? new RelevantPartition(dataset.getPrefData(), x -> true) : new UniformPartition();
        List<Integer> splitPoints = partition.split(train, numParts);

        for(int part = 0; part < numParts; ++part)
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

            int currentPart = part;
            System.out.println("Started part " + (part+1) + " /" + numParts);

            // We take the full train set as the preference data
            List<Tuple2<Integer, Integer>> partValid = train.subList(0, splitPoints.get(part));
            int realVal = partition.split(partValid, percTrain);

            // And only a fraction of the ratings as training.
            List<Tuple2<Integer, Integer>> partTrain = train.subList(0, realVal);

            // Build the validation triplets and compute the number of relevant ratings.
            Dataset<Long, String> validDataset = Dataset.load(dataset, partValid);
            int notRel = validDataset.getNumRel(partTrain);

            Initializer<Long, String> initializer = new Initializer<>(validDataset.getPrefData(), partTrain, false, false);
            List<Tuple2<Integer, Integer>> fullTraining = initializer.getFullTraining();
            List<Tuple2<Integer, Integer>> cleanTraining = initializer.getCleanTraining();
            List<IntList> availability = initializer.getAvailability();

            System.out.println("Training: " + splitPoints.get(part) + " recommendations (" + (part + 1) + "/" + numParts + ")");
            System.out.println(validDataset.toString());
            System.out.println("Total recommendations: " + splitPoints.get(part) + " (" + (part + 1) + "/" + numParts + ")");
            System.out.println("Training recommendations: " + realVal + " (" + (part + 1) + "/" + numParts + ")");
            System.out.println("Relevant recommendations (with training): " + (validDataset.getNumRel() - notRel));

            long a = System.currentTimeMillis();
            // Initialize the metrics to compute.
            Map<String, Supplier<CumulativeMetric<Long, String>>> metrics = new HashMap<>();
            metrics.put("recall", () -> new CumulativeRecall<>(validDataset.getPrefData(), validDataset.getNumRel(), 0.5));
            List<String> metricNames = new ArrayList<>(metrics.keySet());
            long b = System.currentTimeMillis();
            System.out.println("Metrics prepared (" + (b - a) + " ms.");

            // Select the algorithms
            AlgorithmSelector<Long, String> algorithmSelector = new AlgorithmSelector<>();
            algorithmSelector.configure(validDataset.getUserIndex(), validDataset.getItemIndex(), validDataset.getPrefData(), realThreshold);
            algorithmSelector.addFile(algorithms);
            Map<String, InteractiveRecommender<Long, String>> recs = algorithmSelector.getRecs();

            // Initialize the algorithm queue.

            b = System.currentTimeMillis();
            System.out.println("Recommenders prepared (" + (b - a) + " ms.)");

            // Store the values for each algorithm.
            Map<String, Double> auxiliarValues = new ConcurrentHashMap<>();

            recs.entrySet().parallelStream().forEach((entry) ->
            {
                String name = entry.getKey();
                InteractiveRecommender<Long, String> rec = entry.getValue();

                UntieRandomNumberReader rngSeedGen = new UntieRandomNumberReader();
                // And the metrics
                Map<String, CumulativeMetric<Long, String>> localMetrics = new HashMap<>();
                metricNames.forEach(metricName -> localMetrics.put(metricName, metrics.get(metricName).get()));

                // Configure and initialize the recommendation loop:
                System.out.println("Starting algorithm " + name);
                long aaa = System.nanoTime();
                EndCondition endcond = iterationsStop ? (auxIter == 0.0 ? new NoLimitsEndCondition() : new NumIterEndCondition(numIter)) : new PercentagePositiveRatingsEndCondition(dataset.getNumRel(), auxIter, 0.5);

                Map<String, Double> averagedLastIteration = new HashMap<>();
                metricNames.forEach(metricName -> averagedLastIteration.put(metricName, 0.0));

                double maxIter = 0;
                // Execute each recommender k times.
                for (int i = 0; i < k; ++i)
                {
                    int rngSeed = rngSeedGen.nextSeed();

                    // Create the recommendation loop:
                    RecommendationLoop<Long, String> loop = new RecommendationLoop<>(validDataset.getUserIndex(), validDataset.getItemIndex(), validDataset.getPrefData(), rec, localMetrics, endcond, rngSeed, false);
                    loop.init(fullTraining, cleanTraining, availability, false);

                    long bbb = System.nanoTime();
                    System.out.println("Algorithm " + name + " for part " + (currentPart+1) + " /" + numParts + " has been initialized (" + (bbb - aaa) / 1000000.0 + " ms.)");

                    Map<String, List<Double>> metricValues = new HashMap<>();
                    metricNames.forEach(metricName -> metricValues.put(metricName, new ArrayList<>()));

                    try {
                        List<Tuple3<Integer, Integer, Long>> list = new ArrayList<>();
                        String fileName = outputFolder + name + "_" + i + ".txt";
                        if (resume) {
                            list = AuxiliarMethods.retrievePreviousIterations(fileName);
                        }

                        Writer writer = new Writer(currentOutputFolder + name + "_" + i + ".txt", metricNames);
                        if (resume && !list.isEmpty()) {
                            metricValues.putAll(AuxiliarMethods.updateWithPrevious(loop, list, writer, interval));
                        }

                        int currentIter = AuxiliarMethods.executeRemaining(loop, writer, interval, metricValues);
                        maxIter = maxIter + (currentIter - maxIter) / (i + 1.0);
                        writer.close();
                    } catch (IOException ioe) {
                        System.err.println("ERROR: Some error occurred when executing algorithm " + name + " (" + i + ") for part " + (currentPart+1) + " /" + numParts);
                    }

                    for (String metric : metricNames)
                    {
                        List<Double> values = metricValues.get(metric);
                        int auxSize = values.size();
                        if (i == 0)
                        {
                            averagedLastIteration.put(metric, values.get(auxSize - 1));
                        }
                        else
                        {
                            double lastIter = averagedLastIteration.get(metric);
                            double newValue = lastIter + (values.get(auxSize - 1) - lastIter) / (i + 1.0);
                            averagedLastIteration.put(metric, newValue);
                        }
                    }

                    System.out.println("Algorithm " + name + " (" + i + ") for part " + (currentPart+1) + " /" + numParts + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
                }

                if(iterationsStop) auxiliarValues.put(name, averagedLastIteration.get("recall"));
                else auxiliarValues.put(name, maxIter);

                long bbb = System.nanoTime();
                System.out.println("Algorithm " + name + " for part " + (currentPart+1) + " /" + numParts + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
            });

            PriorityQueue<Tuple2od<String>> ranking = new PriorityQueue<>(recs.size(), (x, y) -> (int) Math.signum(y.v2() - x.v2()));
            auxiliarValues.forEach((algorithm, value) -> ranking.add(new Tuple2od<>(algorithm, value)));

            // Write the algorithm ranking
            try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(currentOutputFolder + "ranking.txt"))))
            {
                bw.write("Algorithm\t" + (iterationsStop ? "recall@" + numIter : "numIters@" + auxIter));
                while(!ranking.isEmpty())
                {
                    Tuple2od<String> alg = ranking.poll();
                    assert alg != null;
                    bw.write("\n" + alg.v1 + "\t" + alg.v2);
                }
            }
        }
    }
}
