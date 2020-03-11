/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.contact;

import es.uam.eps.ir.knnbandit.UntieRandomNumber;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.io.GraphReader;
import es.uam.eps.ir.knnbandit.graph.io.TextGraphReader;
import es.uam.eps.ir.knnbandit.io.Reader;
import es.uam.eps.ir.knnbandit.main.Initializer;
import es.uam.eps.ir.knnbandit.main.general.movielens.InteractiveRecommendation;
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
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2od;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Class for applying validation to the interactive contact recommendation approaches.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class InteractiveContactRecommendationValidation
{
    /**
     * Program for applying validation to the interactive contact recommendation approaches.
     *
     * @param args Execution arguments:
     *             <ol>
     *                  <li><b>Algorithms:</b> the recommender systems to apply validation for</li>
     *                  <li><b>Input:</b> Full preference data</li>
     *                  <li><b>Output:</b> Folder in which to store the output</li>
     *                  <li><b>Num. Iter:</b> Number of iterations for the validation. 0 if we want to run out of recommendable items</li>
     *                  <li><b>Resume:</b> True if we want to resume previous executions, false to overwrite them</li>
     *                  <li><b>Directed:</b>True if the network is directed, false otherwise</li>
     *                  <li><b>Not reciprocal:</b>In case it is directed, this arguments indicates whether the reciprocal edge is added to training after recommendation (true) or not (false)</li>
     *                  <li><b>Training data:</b>File containing the training data (a previous execution of a recommender over the cold start problem)</li>
     *                  <li><b>Train percentage:</b> Percentage of the training data which will be provided as input to the recommender</li>
     *                  <li><b>Num. partitions:</b> Number of training partitions we are going to use. Ex: if this argument is equal to 5, we will
     *                         execute the loop 5 times: one with 20% of the training, one with 40%, etc. </li>
     *                  <li><b>Algorithms file:</b>Name of the file containing the algorithm ranking.</li>
     *             </ol>
     */
    public static void main(String[] args) throws IOException, UnconfiguredException
    {
        if (args.length < 11)
        {
            System.err.println("ERROR:iInvalid arguments");
            System.err.println("Usage:");
            System.err.println("Algorithms:  the recommender systems to apply validation for ");
            System.err.println("Input:  Full preference data ");
            System.err.println("Output:  Folder in which to store the output ");
            System.err.println("Num. Iter:  Number of iterations for the validation. 0 if we want to run out of recommendable items ");
            System.err.println("Resume:  True if we want to resume previous executions, false to overwrite them ");
            System.err.println("Directed: True if the network is directed, false otherwise ");
            System.err.println("Not reciprocal: In case it is directed, this arguments indicates whether the reciprocal edge is added to training after recommendation (true) or not (false) ");
            System.err.println("Training data: File containing the training data (a previous execution of a recommender over the cold start problem) ");
            System.err.println("Train percentage:  Percentage of the training data which will be provided as input to the recommender ");
            System.err.println("Num. partitions:  Number of training partitions we are going to use. Ex: if this argument is equal to 5, we will execute the loop 5 times: one with 20% of the training, one with 40%, etc.  ");
            System.err.println("Algorithms file: Name of the file containing the algorithm ranking. ");
            return;
        }

        // First, we read the program arguments.
        String algorithms = args[0];
        String input = args[1];
        String output = args[2];

        // Define the stop condition.
        Double auxIter = Parsers.dp.parse(args[3]);
        boolean iterationsStop = auxIter == 0.0 || auxIter >= 1.0;
        int numIter = (iterationsStop && auxIter > 1.0) ? auxIter.intValue() : Integer.MAX_VALUE;

        //Define whether the algorithm has previous executions.
        boolean resume = args[4].equalsIgnoreCase("true");

        // Specific properties of the contact recommendation case.
        boolean directed = args[5].equalsIgnoreCase("true");
        boolean notReciprocal = !directed || args[6].equalsIgnoreCase("true");

        // Training data.
        String trainingData = args[7];
        double percTrain = Parsers.dp.parse(args[8]);
        int auxNumParts = Parsers.ip.parse(args[9]);
        boolean relevantPartition = auxNumParts < 0;
        int numParts = Math.abs(auxNumParts);

        // Output file for the validation.
        String algorithmsFile = args[10];

        // First, we read the training data.
        Reader reader = new Reader();
        List<Tuple2<Integer, Integer>> train = reader.read(trainingData, "\t", true);
        System.out.println("Training data read");

        // First, we identify and find the random seed which will be used for unties.
        UntieRandomNumber.configure(resume, output);

        /* Rating reading */
        Set<Long> users = new HashSet<>();
        List<Tuple3<Long, Long, Double>> triplets = new ArrayList<>();

        // Read the graph
        Graph<Long> graph;
        GraphReader<Long> greader = new TextGraphReader<>(directed, false, false, "\t", Parsers.lp);
        graph = greader.read(input);
        // Configure the user and triplets lists
        graph.getAllNodes().forEach(users::add);
        graph.getAllNodes().forEach(u -> graph.getAdjacentNodes(u).forEach(v -> triplets.add(new Tuple3<>(u, v, 1.0))));

        // Then, we obtain the preference data.
        FastUpdateableUserIndex<Long> uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        FastUpdateableItemIndex<Long> iIndex = SimpleFastUpdateableItemIndex.load(users.stream());
        SimpleFastPreferenceData<Long, Long> prefData = SimpleFastPreferenceData.load(triplets.stream(), uIndex, iIndex);

        Partition partition = relevantPartition ? new RelevantPartition(prefData, x -> true) : new UniformPartition();
        List<Integer> splitPoints = partition.split(train, numParts);

        // And, after that, the splits.
        int trainingSize = train.size();
        for(int part = 0; part < numParts; ++part)
        {
            // We take the full train set as the preference data
            List<Tuple2<Integer, Integer>> partValid = train.subList(0, splitPoints.get(part));

            int realVal = partition.split(partValid, percTrain);
            // And only a fraction of the ratings as training.
            List<Tuple2<Integer, Integer>> partTrain = partValid.subList(0, realVal);


            // We build the preference data.
            List<Tuple3<Long, Long, Double>> validationTriplets = new ArrayList<>();

            // Build the validation triplets and compute the number of relevant ratings.
            int defNumRel = partValid.stream().mapToInt(tuple ->
            {
                int uidx = tuple.v1;
                int iidx = tuple.v2;
                if (prefData.numItems(uidx) > 0 && prefData.numUsers(iidx) > 0 && prefData.getPreference(uidx, iidx).isPresent())
                {
                    validationTriplets.add(new Tuple3<>(prefData.uidx2user(uidx), prefData.iidx2item(iidx), 1.0));

                    // Also add the opposite, if notReciprocal
                    if (notReciprocal)
                    {
                        if (prefData.numItems(iidx) > 0 && prefData.numUsers(uidx) > 0 && prefData.getPreference(iidx, uidx).isPresent())
                        {
                            validationTriplets.add(new Tuple3<>(prefData.uidx2user(uidx), prefData.iidx2item(iidx), 1.0));
                        }
                    }

                    return 1;
                }
                return 0;
            }).sum();




            int notRel = partTrain.stream().mapToInt(tuple ->
            {
                int uidx = tuple.v1;
                int iidx = tuple.v2;
                if (prefData.numItems(uidx) > 0 && prefData.numUsers(iidx) > 0 && prefData.getPreference(uidx, iidx).isPresent())
                {
                    return 1;
                }

                return 0;
            }).sum();

            // Create the validation data, which will be provided as input to recommenders and metrics.
            SimpleFastPreferenceData<Long, Long> validData = SimpleFastPreferenceData.load(validationTriplets.stream(), uIndex, iIndex);

            Initializer<Long, Long> initializer = new Initializer<>(validData, partTrain, true, notReciprocal);

            List<Tuple2<Integer, Integer>> fullTraining = initializer.getFullTraining();
            List<Tuple2<Integer, Integer>> cleanTraining = initializer.getCleanTraining();
            List<IntList> availability = initializer.getAvailability();

            System.out.println("Training: " + splitPoints.get(part) + " recommendations (" + (part + 1) + "/" + numParts + ")");

            System.out.println("Num items:" + users.size());
            System.out.println("Num. users: " + validData.numUsersWithPreferences());
            System.out.println("Total relevant in " + (part + 1) + "/" + numParts + ": " + defNumRel);
            System.out.println("Total validation in " + (part + 1) + "/" + numParts + ": " + (defNumRel - notRel));
            System.out.println("Total recommendations: " + splitPoints.get(part) + " (" + (part + 1) + "/" + numParts + ")");
            System.out.println("Training recommendations: " + realVal + " (" + (part + 1) + "/" + numParts + ")");

            long a = System.currentTimeMillis();
            // Initialize the metrics to compute.
            Map<String, Supplier<CumulativeMetric<Long, Long>>> metrics = new HashMap<>();
            metrics.put("recall", () -> new CumulativeRecall<>(validData, defNumRel, 0.5));
            List<String> metricNames = new ArrayList<>(metrics.keySet());
            long b = System.currentTimeMillis();
            System.out.println("Metrics prepared (" + (b - a) + " ms.");

            // Select the algorithms
            AlgorithmSelector<Long, Long> algorithmSelector = new AlgorithmSelector<>();
            algorithmSelector.configure(uIndex, iIndex, validData, 0.5, notReciprocal);
            algorithmSelector.addFile(algorithms);
            Map<String, InteractiveRecommender<Long, Long>> recs = algorithmSelector.getRecs();

            // Initialize the algorithm queue.
            PriorityQueue<Tuple2od<String>> queue = new PriorityQueue<>(recs.size(), (x, y) -> (int) Math.signum(y.v2() - x.v2()));
            b = System.currentTimeMillis();
            System.out.println("Recommenders prepared (" + (b - a) + " ms.)");

            // Create the directory to use.
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

            AtomicInteger atom = new AtomicInteger(0);
            int numRecs = recs.size();
            // Then, execute each algorithm for this split.
            recs.entrySet().parallelStream().forEach(re ->
            {
                long aaa = System.currentTimeMillis();
                System.out.println("Started algorithm:" + re.getKey());

                // Obtain the recommender.
                InteractiveRecommender<Long, Long> rec = re.getValue();
                // Obtain the metric.
                Map<String, CumulativeMetric<Long, Long>> localMetrics = new HashMap<>();
                metricNames.forEach(name -> localMetrics.put(name, metrics.get(name).get()));

                // Define the ending condition of the loop.
                EndCondition endcond = iterationsStop ? (auxIter == 0.0 ? new NoLimitsEndCondition() : new NumIterEndCondition(numIter)) : new PercentagePositiveRatingsEndCondition(defNumRel-notRel, auxIter, 0.5);

                // Initialize the recommendation loop.
                RecommendationLoop<Long, Long> loop = new RecommendationLoop<>(uIndex, iIndex, validData, rec, localMetrics, endcond, UntieRandomNumber.RNG, notReciprocal);
                loop.init(fullTraining, cleanTraining, availability, true);
                long bbb = System.currentTimeMillis();
                System.out.println("Initialized algorithm " + re.getKey() + " (" + (bbb - aaa) + " ms.)");

                // Then, execute the loop:
                List<Tuple3<Integer, Integer, Long>> list = new ArrayList<>();
                String fileName = outputFolder + re.getKey() + ".txt";
                // If there was a previous execution and we want to recover it, obtain previous values
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

                // Then, execute.
                try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFolder + re.getKey() + ".txt"))))
                {
                    // Write the header.
                    bw.write("Num.Iter\tUser\tItem");
                    for (String metric : metricNames)
                    {
                        bw.write("\t" + metric);
                    }
                    bw.write("\tTime\n");

                    List<Tuple2<Integer, Integer>> updList = new ArrayList<>();
                    // Write the previous values
                    if (resume && !list.isEmpty())
                    {
                        for (Tuple3<Integer, Integer, Long> triplet : list)
                        {
                            StringBuilder builder = new StringBuilder();
                            // First, we update the loop metrics.
                            Tuple2<Integer, Integer> tuple = new Tuple2<>(triplet.v1, triplet.v2);
                            loop.updateMetrics(tuple);
                            updList.add(tuple);
                            // Then, we obtain the values to print.
                            int iter = loop.getCurrentIteration();
                            builder.append(iter);
                            builder.append("\t");
                            builder.append(triplet.v1);
                            builder.append("\t");
                            builder.append(triplet.v2);
                            // Print the metric values.
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

                        bbb = System.currentTimeMillis();
                        System.out.println("Algorithm " + re.getKey() + " finished retrieving data (" + (bbb - aaa) + " ms.)");
                    }


                    if(!loop.hasEnded() && resume && !list.isEmpty())
                    {
                        loop.updateRecs(updList);
                        bbb = System.currentTimeMillis();
                        System.out.println("Algorithm " + re.getKey() + " finished updating itself with retrieved data (" + (bbb - aaa) + " ms.)");
                    }

                    // Then, execute the remaining loop.
                    while (!loop.hasEnded())
                    {
                        StringBuilder builder = new StringBuilder();
                        long aa = System.currentTimeMillis();

                        // Obtain the next element and update.
                        Tuple2<Integer, Integer> tuple = loop.nextIteration();
                        long bb = System.currentTimeMillis();
                        if (tuple == null)
                        {
                            break; // The loop has finished
                        }

                        // Write all the values.
                        int iter = loop.getCurrentIteration();
                        builder.append(iter);
                        builder.append("\t");
                        builder.append(tuple.v1);
                        builder.append("\t");
                        builder.append(tuple.v2);
                        // Print the metric values.
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
                int algorithmId = atom.incrementAndGet();
                System.out.println("Algorithm " + re.getKey() + " finished (" + (bbb - aaa) + " ms., " + algorithmId + "/" + numRecs + ")");
            });

            // Once we have finished all the recommendations, we write the algorithm ranking into a file.
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
