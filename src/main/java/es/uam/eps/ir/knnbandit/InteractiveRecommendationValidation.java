/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit;

import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.io.Reader;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.RecommendationLoop;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2od;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.*;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Class for applying validation to the interactive recommendation approaches.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class InteractiveRecommendationValidation
{
    /**
     * Program for applying validation to the interactive recommendation approaches.
     *
     * @param args Execution arguments:
     *             <ol>
     *                  <li><b>Algorithms:</b> the recommender systems to apply validation for</li>
     *                  <li><b>Input:</b> Full preference data</li>
     *                  <li><b>Output:</b> Folder in which to store the output</li>
     *                  <li><b>Num. Iter:</b> Number of iterations for the validation. 0 if we want to run out of recommendable items</li>
     *                  <li><b>Threshold:</b> Relevance threshold</li>
     *                  <li><b>Resume:</b> True if we want to resume previous executions, false to overwrite them</li>
     *                  <li><b>Use ratings:</b>True if we want to take the true rating value, false if we want to binarize them</li>
     *                  <li><b>Training data:</b>File containing the training data (a previous execution of a recommender over the cold start problem)</li>
     *                  <li><b>Train percentage:</b> Percentage of the training data which will be provided as input to the recommender</li>
     *                  <li><b>Num. partitions:</b> Number of training partitions we are going to use. Ex: if this argument is equal to 5, we will
     *                         execute the loop 5 times: one with 20% of the training, one with 40%, etc. </li>
     *             </ol>
     */
    public static void main(String[] args) throws IOException, UnconfiguredException
    {
        if (args.length < 11)
        {
            System.err.println("ERROR: Invalid arguments");
            System.err.println("Usage:");
            System.err.println("Algorithms: the recommender systems to apply validation for");
            System.err.println("Input: Full preference data");
            System.err.println("Output: Folder in which to store the output");
            System.err.println("Num. Iter: Number of iterations for the validation. 0 if we want to run out of recommendable items");
            System.err.println("Threshold: Relevance threshold");
            System.err.println("Resume: True if we want to resume previous executions, false to overwrite them");
            System.err.println("Use ratings:True if we want to take the true rating value, false if we want to binarize them");
            System.err.println("Training data:File containing the training data (a previous execution of a recommender over the cold start problem)");
            System.err.println("Train percentage: Percentage of the training data which will be provided as input to the recommender");
            System.err.println("Num. partitions: Number of training partitions we are going to use. Ex: if this argument is equal to 5, we will execute the loop 5 times: one with 20% of the training, one with 40%, etc. ");
            return;
        }

        // First, read the program arguments.
        String algorithms = args[0];
        String input = args[1];
        String output = args[2];
        int numIter = Parsers.ip.parse(args[3]);
        double threshold = Parsers.dp.parse(args[4]);
        boolean resume = args[5].equalsIgnoreCase("true");
        boolean useRatings = args[6].equalsIgnoreCase("true");
        String trainingData = args[7];
        double percTrain = Parsers.dp.parse(args[9]);
        int numParts = Parsers.ip.parse(args[8]);
        String algorithmsFile = args[10];

        // Establish functions
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x :
                (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);

        // Read the full training data.
        Reader reader = new Reader();
        List<Tuple2<Integer, Integer>> train = reader.read(trainingData, "\t", true);

        // First, we identify and find the random seed which will be used for unties.
        // This is stored in a file in the output folder named "rngseed". If it does not exist,
        // the number is created.
        if (resume)
        {
            File f = new File(output + "rngseed");
            if (f.exists())
            {
                try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f))))
                {
                    UntieRandomNumber.RNG = Parsers.ip.parse(br.readLine());
                }
            }
            else
            {
                Random rng = new Random();
                UntieRandomNumber.RNG = rng.nextInt();
            }
        }
        else
        {
            Random rng = new Random();
            UntieRandomNumber.RNG = rng.nextInt();
        }

        // Create the new random seed file.
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output + "rngseed"))))
        {
            bw.write("" + UntieRandomNumber.RNG);
        }

        // Then, we read the ratings.
        Set<Long> users = new HashSet<>();
        Set<Long> items = new HashSet<>();
        //Set<String> items = new HashSet<>();
        List<Tuple3<Long, Long, Double>> triplets = new ArrayList<>();
        //List<Tuple3<Long, Long, Double>> triplets = new ArrayList<>();

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

                triplets.add(new Tuple3<>(user, item, rating));
            }
        }

        // First, obtain the full preference information.
        FastUpdateableUserIndex<Long> uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        FastUpdateableItemIndex<Long> iIndex = SimpleFastUpdateableItemIndex.load(items.stream());
        //FastUpdateableItemIndex<String> iIndex = SimpleFastUpdateableItemIndex.load(items.stream());
        //SimpleFastPreferenceData<Long, Long> prefData = SimpleFastPreferenceData.load(triplets.stream(), uIndex, iIndex);
        SimpleFastPreferenceData<Long, Long> prefData = SimpleFastPreferenceData.load(triplets.stream(), uIndex, iIndex);

        System.out.println("Users: " + uIndex.numUsers());
        System.out.println("Items: " + iIndex.numItems());
        int trainingSize = train.size();

        // For each split...
        for (int part = 0; part < numParts; ++part)
        {
            // First, initialize the data.
            int val = trainingSize * (part + 1);
            val /= numParts;

            int realVal = (int) percTrain * val;

            // We take the full train set as the preference data
            List<Tuple2<Integer, Integer>> partValid = train.subList(0, val);
            // And only a fraction of the ratings as training.
            List<Tuple2<Integer, Integer>> partTrain = train.subList(0, realVal);

            // We build the preference data.
            //List<Tuple3<Long, Long, Double>> validationTriplets = new ArrayList<>();
            List<Tuple3<Long, Long, Double>> validationTriplets = new ArrayList<>();
            int defNumRel = partValid.stream().mapToInt(tuple ->
            {
                int uidx = tuple.v1;
                int iidx = tuple.v2;
                if (prefData.numItems(uidx) > 0 && prefData.numUsers(iidx) > 0 && prefData.getPreference(uidx, iidx).isPresent())
                {
                    double value = prefData.getPreference(uidx, iidx).get().v2();
                    validationTriplets.add(new Tuple3<>(prefData.uidx2user(uidx), prefData.iidx2item(iidx), value));
                    if (relevance.test(value))
                    {
                        return 1;
                    }
                }
                return 0;
            }).sum();

            // Now, we obtain the validation data, which will be provided as input for the recommenders and metrics.
            //SimpleFastPreferenceData<Long, Long> validData = SimpleFastPreferenceData.load(validationTriplets.stream(), uIndex, iIndex);
            SimpleFastPreferenceData<Long, Long> validData = SimpleFastPreferenceData.load(validationTriplets.stream(), uIndex, iIndex);
            System.out.println("Training: " + val + " recommendations (" + (part + 1) + "/" + numParts + ")");

            // Initialize the metrics to compute.
            //Map<String, Supplier<CumulativeMetric<Long, Long>>> metrics = new HashMap<>();
            Map<String, Supplier<CumulativeMetric<Long, Long>>> metrics = new HashMap<>();
            metrics.put("recall", () -> new CumulativeRecall<>(validData, defNumRel, 0.5));
            List<String> metricNames = new ArrayList<>(metrics.keySet());

            // Select the algorithms.
            long a = System.currentTimeMillis();
            //AlgorithmSelector<Long, Long> algorithmSelector = new AlgorithmSelector<>();
            AlgorithmSelector<Long, Long> algorithmSelector = new AlgorithmSelector<>();
            algorithmSelector.configure(uIndex, iIndex, validData, useRatings ? threshold : 0.5);
            algorithmSelector.addFile(algorithms);
            //Map<String, InteractiveRecommender<Long, Long>> recs = algorithmSelector.getRecs();
            Map<String, InteractiveRecommender<Long, Long>> recs = algorithmSelector.getRecs();
            long b = System.currentTimeMillis();

            // Initialize the algorithm queue.
            PriorityQueue<Tuple2od<String>> queue = new PriorityQueue<>(recs.size(), (x, y) -> (int) Math.signum(y.v2() - x.v2()));

            System.out.println("Recommenders ready (" + (b - a) + " ms.)");

            File folder = new File(output + part + File.separator);
            if (!folder.exists())
            {
                if (!folder.mkdirs())
                {
                    System.err.println("ERROR: Invalid output folder");
                    return;
                }
            }

            String outputFolder = output + part + File.separator;

            // Execute each recommender.
            recs.entrySet().parallelStream().forEach(re ->
            {
                long aaa = System.currentTimeMillis();
                System.out.println("Algorithm " + re.getKey() + " started");

                // Obtain the recommender and the metrics.
                //InteractiveRecommender<Long, Long> rec = re.getValue();
                InteractiveRecommender<Long, Long> rec = re.getValue();
                Map<String, CumulativeMetric<Long, Long>> localMetrics = new HashMap<>();
                //Map<String, CumulativeMetric<Long, Long>> localMetrics = new HashMap<>();
                metricNames.forEach(name -> localMetrics.put(name, metrics.get(name).get()));

                // Initialize the recommendation loop
                //RecommendationLoop<Long, Long> loop = new RecommendationLoop<>(uIndex, iIndex, prefData, rec, localMetrics, numIter, UntieRandomNumber.RNG, false);
                RecommendationLoop<Long, Long> loop = new RecommendationLoop<>(uIndex, iIndex, prefData, rec, localMetrics, numIter, UntieRandomNumber.RNG, false);

                loop.init(partTrain, false);

                long bbb = System.currentTimeMillis();
                System.out.println("Algorithm " + re.getKey() + " initialized (" + (bbb - aaa) + " ms.)");


                // In case the algorithm had some previous execution, retrieve it if it is indicated as so.
                List<Tuple3<Integer, Integer, Long>> list = new ArrayList<>();
                String fileName = outputFolder + re.getKey() + ".txt";

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

                // Then, execute the algorithm and store it in the corresponding file
                try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFolder + re.getKey() + ".txt"))))
                {
                    // Write the header
                    bw.write("Num.Iter\tUser\tItem");
                    for (String metric : metricNames)
                    {
                        bw.write("\t" + metric);
                    }
                    bw.write("\tTime\n");

                    // If there is something previously retrieved, write it.
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
                    }

                    // Then, until the loop ends...
                    while (!loop.hasEnded())
                    {
                        StringBuilder builder = new StringBuilder();
                        long aa = System.currentTimeMillis();

                        // Obtain the next element and update the recommender and metrics
                        Tuple2<Integer, Integer> tuple = loop.nextIteration();
                        long bb = System.currentTimeMillis();
                        if (tuple == null)
                        {
                            break; // The loop has finished
                        }

                        // Write everything.
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

                bbb = System.currentTimeMillis();

                // Once we've finished with the algorithm, store the final value in the algorithm queue.
                String recName = re.getKey();
                double value = localMetrics.get("recall").compute();

                queue.add(new Tuple2od<>(recName, value));

                System.out.println("Algorithm " + re.getKey() + " finished (" + (bbb - aaa) + " ms.)");
            });

            // Write the algorithm ranking.
            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFolder + algorithmsFile))))
            {
                while (!queue.isEmpty())
                {
                    Tuple2od<String> algData = queue.poll();
                    bw.write(algData.v1 + "\t" + algData.v2 + "\n");
                }
            }
        }
    }
}
