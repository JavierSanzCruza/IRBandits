/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.general.movielens;

import es.uam.eps.ir.knnbandit.UntieRandomNumber;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.io.Reader;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
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
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.*;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * Class for executing contact recommender systems in simulated interactive loops (with training)
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class InteractiveRecommendationParallel
{
    /**
     * @param args Execution arguments
     *             <ol>
     *                  <li><b>Algorithms:</b> the recommender systems to apply validation for</li>
     *                  <li><b>Input:</b> Full preference data</li>
     *                  <li><b>Output:</b> Folder in which to store the output</li>
     *                  <li><b>Num. Iter:</b> Number of iterations for the validation. 0 if we want to run out of recommendable items</li>
     *                  <li><b>Threshold:</b> Relevance threshold</li>
     *                  <li><b>Resume:</b> True if we want to resume previous executions, false to overwrite them</li>
     *                  <li><b>Use ratings:</b>True if we want to take the true rating value, false if we want to binarize them</li>
     *                  <li><b>Training data:</b>File containing the training data (a previous execution of a recommender over the cold start problem)</li>
     *                  <li><b>Num. partitions:</b> Number of training partitions we are going to use. Ex: if this argument is equal to 5, we will
     *                         execute the loop 5 times: one with 20% of the training, one with 40%, etc. </li>
     *             </ol>
     */
    public static void main(String[] args) throws IOException, UnconfiguredException
    {
        if (args.length < 10)
        {
            System.err.println("ERROR: Invalid arguments");
            System.err.println("Usage:");
            System.err.println("Algorithms: the recommender systems to apply validation for");
            System.err.println("Input: Full preference data");
            System.err.println("Output: Folder in which to store the output");
            System.err.println("Num. Iter: Number of iterations for the validation. 0 if we want to run out of recommendable items");
            System.err.println("Resume: True if we want to resume previous executions, false to overwrite them");
            System.err.println("Threshold: Relevance threshold");
            System.err.println("Use ratings:True if we want to take the true rating value, false if we want to binarize them");
            System.err.println("Training data:File containing the training data (a previous execution of a recommender over the cold start problem)");
            System.err.println("Num. partitions: Number of training partitions we are going to use. Ex: if this argument is equal to 5, we will execute the loop 5 times: one with 20% of the training, one with 40%, etc. ");
            System.err.println("Also without training: true if we also want to execute a version without training");
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

        boolean resume = args[4].equalsIgnoreCase("true");

        // General recommendation specific arguments
        double threshold = Parsers.dp.parse(args[5]);
        boolean useRatings = args[6].equalsIgnoreCase("true");

        // Training data.
        String trainingData = args[7];
        int auxNumParts = Parsers.ip.parse(args[8]);
        boolean relevantPartition = auxNumParts < 0;
        int numParts = Math.abs(auxNumParts);

        boolean alsoWithoutTraining = args[9].equalsIgnoreCase("true");

        // Configure the weight function and relevance functions.
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x :
                (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
        double realThreshold = useRatings ? threshold : 0.5;

        // Read the training data
        Reader reader = new Reader();
        List<Tuple2<Integer, Integer>> train = reader.read(trainingData, "\t", true);

        // Configure the random number generator for ties.
        UntieRandomNumber.configure(resume, output);

       // Then, we read the ratings.
        Set<Long> users = new HashSet<>();
        Set<Long> items = new HashSet<>();
        List<Tuple3<Long, Long, Double>> triplets = new ArrayList<>();
        int numrel = 0;

        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(input))))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] split = line.split("::");
                Long user = Parsers.lp.parse(split[0]);
                //String item = split[1];
                Long item = Parsers.lp.parse(split[1]);
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

        FastUpdateableUserIndex<Long> uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        FastUpdateableItemIndex<Long> iIndex = SimpleFastUpdateableItemIndex.load(items.stream());
        SimpleFastPreferenceData<Long, Long> prefData = SimpleFastPreferenceData.load(triplets.stream(), uIndex, iIndex);

        System.out.println("Users: " + uIndex.numUsers());
        System.out.println("Items: " + iIndex.numItems());
        System.out.println("Total number of relevant items:" + numrel);
        int numRel = numrel;

        // Initialize the metrics to compute.
        Map<String, Supplier<CumulativeMetric<Long, Long>>> metrics = new HashMap<>();
        metrics.put("recall", () -> new CumulativeRecall<>(prefData, numRel, realThreshold));
        metrics.put("gini", () -> new CumulativeGini<>(items.size()));
        List<String> metricNames = new ArrayList<>(metrics.keySet());

        // Select the algorithms
        long a = System.currentTimeMillis();
        List<String> algorithmNames = readAlgorithmList(algorithms, numParts);
        List<InteractiveRecommender<Long, Long>> recs = new ArrayList<>();
        AlgorithmSelector<Long, Long> algorithmSelector = new AlgorithmSelector<>();
        algorithmSelector.configure(uIndex, iIndex, prefData, realThreshold);
        for(String algorithm : algorithmNames)
        {
            InteractiveRecommender<Long, Long> rec = algorithmSelector.getAlgorithm(algorithm);
            recs.add(rec);
        }
        long b = System.currentTimeMillis();
        System.out.println("Recommenders prepared (" + (b - a) + " ms.)");

        int trainingSize = train.size();

        Partition partition = relevantPartition ? new RelevantPartition(prefData, relevance) : new UniformPartition();
        List<Integer> splitPoints = partition.split(train, numParts);

        int auxParts = alsoWithoutTraining ? numParts + 1 : numParts;
        IntStream.range(0, auxParts).parallel().forEach(part ->
        {
            List<Tuple2<Integer,Integer>> partTrain;

            String data = (part != numParts+1) ? "for the " + (part + 1) + "/" + numParts + " split" : "for the non-training split";
            long aaa = System.nanoTime();
            long bbb;

            if(part == (numParts + 1))
            {
                partTrain = new ArrayList<>();
                System.out.println("Prepared data for the non-training data version");
            }
            else
            {
                // Step 1: Prepare the training data.
                int val = splitPoints.get(part);


                partTrain = train.subList(0, val);
                bbb = System.nanoTime();

                System.out.println("Prepared training data " + (part + 1) + "/" + numParts + ": " + val + " recommendations (" + (bbb - aaa) / 1000000.0 + " ms.)");
            }

            // Count the number of relevant items:
            int norel = partTrain.stream().mapToInt(t ->
            {
                Optional<IdxPref> opt = prefData.getPreference(t.v1, t.v2);
                if(opt.isPresent() && relevance.test(opt.get().v2))
                {
                    return 1;
                }
                return 0;
            }).sum();

            System.out.println("Number of relevant items " + data + ": " + (numRel - norel));

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
            InteractiveRecommender<Long, Long> rec = recs.get(part);
            // And the metrics
            Map<String, CumulativeMetric<Long, Long>> localMetrics = new HashMap<>();
            metricNames.forEach(name -> localMetrics.put(name, metrics.get(name).get()));

            // Configure and initialize the recommendation loop:
            System.out.println("Starting algorithm " + algorithmName + " " + data);
            EndCondition endcond = iterationsStop ? (auxIter == 0.0 ? new NoLimitsEndCondition() : new NumIterEndCondition(numIter)) : new PercentagePositiveRatingsEndCondition(numRel, auxIter, realThreshold);

            RecommendationLoop<Long, Long> loop = new RecommendationLoop<>(uIndex, iIndex, prefData, rec, localMetrics, endcond, UntieRandomNumber.RNG, false);
            if(partTrain.isEmpty())
            {
                loop.init(false);
            }
            else
            {
                loop.init(partTrain, false);
            }
            bbb = System.nanoTime();
            System.out.println("Algorithm " + algorithmName + " " + data + " has been initialized (" + (bbb-aaa)/1000000.0 + " ms.)");

            List<Tuple3<Integer, Integer, Long>> list = new ArrayList<>();
            String fileName = outputFolder + algorithmName + ".txt";

            // If there are previous executions, obtain them
            if (resume)
            {
                File f = new File(fileName);
                if (f.exists()) // if the file exists, then resume:
                {
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fileName))))
                    {
                        String line = br.readLine();
                        int len;
                        if (line != null)
                        {
                            String[] split = line.split("\t");
                            len = split.length;

                            while ((line = br.readLine()) != null)
                            {
                                split = line.split("\t");
                                if (split.length < len)
                                {
                                    break;
                                }

                                int uidx = Parsers.ip.parse(split[1]);
                                int iidx = Parsers.ip.parse(split[2]);
                                long time = Parsers.lp.parse(split[len - 1]);
                                list.add(new Tuple3<>(uidx, iidx, time));
                            }
                        }
                    }
                    catch (IOException ex)
                    {
                        Logger.getLogger(InteractiveRecommendation.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
            }

            // Execute and write into the file.
            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFolder + algorithmName + ".txt"))))
            {
                // Write the header
                bw.write("Num.Iter\tUser\tItem");
                for (String metric : metricNames)
                {
                    bw.write("\t" + metric);
                }
                bw.write("\tTime\n");

                // If we have retrieved any previous iteration, update the loop.
                if (resume && !list.isEmpty())
                {
                    for (Tuple3<Integer, Integer, Long> triplet : list)
                    {
                        StringBuilder builder = new StringBuilder();
                        loop.update(new Tuple2<>(triplet.v1, triplet.v2));
                        int iter = loop.getCurrentIteration();
                        builder.append(iter);
                        builder.append("\t");
                        builder.append(triplet.v1);
                        builder.append("\t");
                        builder.append(triplet.v2);
                        Map<String, Double> metricVals = loop.getMetrics();
                        for (String name : metricNames)
                        {
                            builder.append("\t");
                            builder.append(metricVals.get(name));
                        }
                        builder.append("\t");
                        builder.append(triplet.v3);
                        builder.append("\n");
                        bw.write(builder.toString());
                    }

                    bbb = System.nanoTime();
                    System.out.println("Algorithm " + algorithmName + " " + data + " has finished recovering from previous executions (" + (bbb-aaa)/1000000.0 + " ms.)");
                }

                // Apply it until the end.
                while (!loop.hasEnded())
                {
                    StringBuilder builder = new StringBuilder();
                    long aa = System.currentTimeMillis();
                    Tuple2<Integer, Integer> tuple = loop.nextIteration();
                    long bb = System.currentTimeMillis();
                    if (tuple == null)
                    {
                        break; // The loop has finished
                    }
                    int iter = loop.getCurrentIteration();
                    builder.append(iter);
                    builder.append("\t");
                    builder.append(tuple.v1);
                    builder.append("\t");
                    builder.append(tuple.v2);
                    Map<String, Double> metricVals = loop.getMetrics();
                    for (String name : metricNames)
                    {
                        builder.append("\t");
                        builder.append(metricVals.get(name));
                    }
                    builder.append("\t");
                    builder.append((bb - aa));
                    builder.append("\n");
                    bw.write(builder.toString());
                }
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
            bbb = System.nanoTime();
            System.out.println("Algorithm " + algorithmName + " " + data + " has finished (" + (bbb-aaa)/1000000.0 + " ms.)");
        });
    }

    /**
     * Reads a list of algorithms.
     * @param file the file containing the algorithms.
     * @param num number of algorithms to read.
     * @return the list containing the algorithms.
     * @throws IOException if something fails while reading the file.
     */
    private static List<String> readAlgorithmList(String file, int num) throws IOException
    {
        List<String> list = new ArrayList<>();
        try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file))))
        {
            for(int i = 0; i < num; ++i)
            {
                String line = br.readLine();
                list.add(line);
            }
        }

        return list;
    }
}
