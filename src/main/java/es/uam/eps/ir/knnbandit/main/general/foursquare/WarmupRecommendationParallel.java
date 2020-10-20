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
import es.uam.eps.ir.knnbandit.data.datasets.GeneralDataset;
import es.uam.eps.ir.knnbandit.io.Reader;
import es.uam.eps.ir.knnbandit.io.Writer;
import es.uam.eps.ir.knnbandit.main.AuxiliarMethods;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.partition.Partition;
import es.uam.eps.ir.knnbandit.partition.RelevantPartition;
import es.uam.eps.ir.knnbandit.partition.UniformPartition;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.NoLimitsEndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.NumIterEndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.PercentagePositiveRatingsEndCondition;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import es.uam.eps.ir.knnbandit.warmup.FullWarmup;
import es.uam.eps.ir.knnbandit.warmup.OnlyRatingsWarmup;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import es.uam.eps.ir.knnbandit.warmup.WarmupType;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * Class for executing contact recommender systems in simulated interactive loops (with training)
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class WarmupRecommendationParallel
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
     *                  <li><b>Also without training:</b> True if we also want to execute a version without training</li>
     *             </ol>
     *             Also, as optional arguments, we have two:
     *             <ol>
     *                  <li><b>-k value :</b> The number of times each individual approach has to be executed (by default: 1)</li>
     *                  <li><b>-interval value: </b> In order to print a summary of each approach, the distance between data points (by default: 10*numUsers)</li>
     *             </ol>
     */
    public static void main(String[] args) throws IOException, UnconfiguredException
    {
        if (args.length < 10)
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
            System.err.println("\tAlso without training: true if we also want to execute a version without training");
            System.err.println("Optional arguments:");
            System.err.println("\t-k value : The number of times each individual approach has to be executed (by default: 1)");
            System.err.println("\t-interval value: In order to print a summary of each approach, the distance between data points (by default: 10*numUsers)");
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

        // Training data.
        String trainingData = args[7];
        int auxNumParts = Parsers.ip.parse(args[8]);
        boolean relevantPartition = auxNumParts < 0;
        int numParts = Math.abs(auxNumParts);

        WarmupType warmupType = WarmupType.fromString(args[9]);
        if(warmupType == null)
        {
            System.err.println("ERROR: Invalid warm-up type");
            return;
        }

        boolean alsoWithoutTraining = args[10].equalsIgnoreCase("true");

        int auxinterval = 0;
        int auxK = 1;

        for (int i = 11; i < args.length; ++i)
        {
            switch (args[i])
            {
                case "-k":
                    ++i;
                    auxK = Parsers.ip.parse(args[i]);
                    break;
                case "-interval":
                    ++i;
                    auxinterval = Parsers.ip.parse(args[i]);
                    break;
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
        GeneralDataset<Long, String> dataset = GeneralDataset.load(input, Parsers.lp, Parsers.sp, "::", weightFunction, relevance);
        System.out.println("Read the whole data");
        System.out.println(dataset.toString());

        int interval = auxinterval == 0 ? 10 * dataset.numUsers() : auxinterval;

        // Read the training data
        Reader reader = new Reader();
        List<Tuple2<Integer, Integer>> train = reader.read(trainingData, "\t", true);
        System.out.println("Read the training data");

        // Initialize the metrics to compute.
        Map<String, Supplier<CumulativeMetric<Long, String>>> metrics = new HashMap<>();
        metrics.put("recall", () -> new CumulativeRecall<>(dataset.getPrefData(), dataset.getNumRel(), 0.5));
        metrics.put("gini", () -> new CumulativeGini<>(dataset.numItems()));
        List<String> metricNames = new ArrayList<>(metrics.keySet());

        // Select the algorithms
        long a = System.currentTimeMillis();
        List<String> algorithmNames = readAlgorithmList(algorithms, numParts);
        List<InteractiveRecommender<Long, String>> recs = new ArrayList<>();
        AlgorithmSelector<Long, String> algorithmSelector = new AlgorithmSelector<>();
        algorithmSelector.configure(dataset.getUserIndex(), dataset.getItemIndex(), dataset.getPrefData(), realThreshold);
        for (String algorithm : algorithmNames)
        {
            InteractiveRecommender<Long, String> rec = algorithmSelector.getAlgorithm(algorithm);
            recs.add(rec);
        }
        long b = System.currentTimeMillis();
        System.out.println("Recommenders prepared (" + (b - a) + " ms.)");

        Partition partition = relevantPartition ? new RelevantPartition(dataset.getPrefData(), x -> true) : new UniformPartition();
        List<Integer> splitPoints = partition.split(train, numParts);
        int auxParts = alsoWithoutTraining ? numParts + 1 : numParts;

        IntStream.range(0, auxParts).parallel().forEach(part ->
        {
            // Initialize the random number seed generator.
            UntieRandomNumberReader rngSeedGen = new UntieRandomNumberReader();

            // Obtain the training.
            List<Tuple2<Integer, Integer>> partTrain;
            String extraString = (part != numParts + 1) ? "for the " + (part + 1) + "/" + numParts + " split" : "for the non-training split";
            long aaa = System.nanoTime();

            // Find the training data.
            partTrain = (part == (numParts + 1) ? new ArrayList<>() : train.subList(0, splitPoints.get(part)));

            Warmup warmup;
            switch (warmupType)
            {
                case FULL:
                    warmup = new FullWarmup(dataset.getPrefData(), partTrain, false, false);
                    break;
                case ONLYRATINGS:
                default:
                    warmup = new OnlyRatingsWarmup(dataset.getPrefData(), partTrain, false, false);
            }


            long bbb = System.nanoTime();
            System.out.println("Prepared training data " + extraString + ": " + warmup.getFullTraining().size() + " recommendations ( " + (bbb - aaa) / 1000000.0 + " ms.)");

            // Count the number of relevant items:
            int norel = dataset.getNumRel(warmup.getFullTraining());
            System.out.println("Number of relevant items " + extraString + ": " + (dataset.getNumRel() - norel));

            // If it does not exist, create the directory in which to store the recommendation.
            String outputFolder = output + part + File.separator;
            File folder = new File(outputFolder);
            if (!folder.exists())
            {
                if (!folder.mkdirs())
                {
                    System.err.println("ERROR: Invalid output folder");
                    return;
                }
            }

            // Obtain the algorithm to apply:
            String algorithmName = algorithmNames.get(part);
            InteractiveRecommender<Long, String> rec = recs.get(part);
            // And the metrics
            Map<String, CumulativeMetric<Long, String>> localMetrics = new HashMap<>();
            metricNames.forEach(name -> localMetrics.put(name, metrics.get(name).get()));

            // Configure and initialize the recommendation loop:
            System.out.println("Starting algorithm " + algorithmName + " " + extraString);
            aaa = System.nanoTime();
            EndCondition endcond = iterationsStop ? (auxIter == 0.0 ? new NoLimitsEndCondition() : new NumIterEndCondition(numIter)) : new PercentagePositiveRatingsEndCondition(dataset.getNumRel() - norel, auxIter, 0.5);

            Map<String, List<Double>> averagedValues = new HashMap<>();
            metricNames.forEach(name -> averagedValues.put(name, new ArrayList<>()));
            IntList counter = new IntArrayList();

            Map<String, Double> averagedLastIteration = new HashMap<>();
            metricNames.forEach(metricName -> averagedLastIteration.put(metricName, 0.0));
            double maxIter = 0.0;
            // Execute each recommender k times.
            for (int i = 0; i < k; ++i)
            {
                int rngSeed = rngSeedGen.nextSeed();
                // Create the recommendation loop:
                RecommendationLoop<Long, String> loop = new RecommendationLoop<>(dataset.getUserIndex(), dataset.getItemIndex(), dataset.getPrefData(), rec, localMetrics, endcond, rngSeed, false);

                // Initialize the loop.
                if (partTrain.isEmpty())
                {
                    loop.init(false);
                }
                else
                {
                    loop.init(warmup, false);
                }
                bbb = System.nanoTime();
                System.out.println("Algorithm " + algorithmName + "(" + i + ") " + extraString + " has been initialized (" + (bbb - aaa) / 1000000.0 + " ms.)");

                Map<String, List<Double>> metricValues = new HashMap<>();
                metricNames.forEach(name -> metricValues.put(name, new ArrayList<>()));

                try
                {
                    List<Tuple3<Integer, Integer, Long>> list = new ArrayList<>();
                    String fileName = outputFolder + algorithmName + "_" + i + ".txt";
                    if (resume)
                    {
                        list = AuxiliarMethods.retrievePreviousIterations(fileName);
                    }

                    Writer writer = new Writer(fileName, metricNames);
                    writer.writeHeader();
                    if (resume && !list.isEmpty())
                    {
                        metricValues.putAll(AuxiliarMethods.updateWithPrevious(loop, list, writer, interval));
                    }

                    int currentIter = AuxiliarMethods.executeRemaining(loop, writer, interval, metricValues);
                    maxIter = maxIter + (currentIter - maxIter) / (i + 1.0);

                    writer.close();
                }
                catch (IOException ioe)
                {
                    System.err.println("ERROR: Some error occurred when executing algorithm " + algorithmName + " (" + i + ") " + extraString);
                }

                boolean first = true;
                for (String metric : metricNames)
                {
                    if (i == 0)
                    {
                        List<Double> values = metricValues.get(metric);
                        int auxSize = values.size();
                        averagedValues.get(metric).addAll(values.subList(0, auxSize - 1));
                        averagedLastIteration.put(metric, values.get(auxSize - 1));
                        if (first)
                        {
                            for (int j = 0; j < auxSize - 1; ++j)
                            {
                                counter.add(1);
                            }
                        }
                        first = false;
                    }
                    else
                    {
                        List<Double> oldVals = averagedValues.get(metric);
                        int currentSize = oldVals.size();
                        List<Double> newVals = metricValues.get(metric);
                        int auxSize = newVals.size();
                        for (int j = 0; j < auxSize - 1; ++j)
                        {
                            if (j >= currentSize)
                            {
                                oldVals.add(newVals.get(j));
                                counter.add(1);
                            }
                            else
                            {
                                double oldM = oldVals.get(j);
                                double averaged = oldM + (newVals.get(j) - oldM) / (counter.get(j) + 1);
                                counter.set(j, counter.get(j) + 1);
                                oldVals.set(j, averaged);
                            }

                            double lastIter = averagedLastIteration.get(metric);
                            double newValue = lastIter + (newVals.get(auxSize - 1) - lastIter) / (i + 1.0);
                            averagedLastIteration.put(metric, newValue);
                        }
                    }
                }

                System.out.println("Algorithm " + algorithmName + " (" + i + ") " + extraString + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
            }

            // Write the summary.
            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFolder + algorithmName + "-summary.txt"))))
            {
                int size = counter.size();
                bw.write("Iteration");
                for (String metricName : metricNames)
                {
                    bw.write("\t" + metricName);
                }

                for (int i = 0; i < size; ++i)
                {
                    bw.write("\n" + (interval * (i + 1)));
                    for (String metricName : metricNames)
                    {
                        bw.write("\t" + averagedValues.get(metricName).get(i));
                    }
                }

                bw.write("\n" + maxIter);
                for (String metricName : metricNames)
                {
                    bw.write("\t" + averagedLastIteration.get(metricName));
                }
            }
            catch (IOException ioe)
            {
                System.err.println("Something failed while writing the summary file");
            }


            bbb = System.nanoTime();
            System.out.println("Algorithm " + algorithmName + " " + extraString + " has finished (" + (bbb - aaa) / 1000000.0 + " ms.)");
        });
    }

    /**
     * Reads a list of algorithms.
     *
     * @param file the file containing the algorithms.
     * @param num  number of algorithms to read.
     * @return the list containing the algorithms.
     * @throws IOException if something fails while reading the file.
     */
    private static List<String> readAlgorithmList(String file, int num) throws IOException
    {
        List<String> list = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file))))
        {
            for (int i = 0; i < num; ++i)
            {
                String line = br.readLine();
                list.add(line);
            }
        }

        return list;
    }
}
