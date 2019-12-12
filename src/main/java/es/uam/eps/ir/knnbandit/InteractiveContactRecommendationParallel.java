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
import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.io.GraphReader;
import es.uam.eps.ir.knnbandit.graph.io.TextGraphReader;
import es.uam.eps.ir.knnbandit.io.Reader;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.RecommendationLoop;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.*;
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
public class InteractiveContactRecommendationParallel
{
    /**
     * Executes contact recommendation systems in simulated interactive loops.
     *
     * @param args Execution arguments:
     *             <ol>
     *                 <li><b>Algorithms:</b> configuration file for the algorithms</li>
     *                 <li><b>Input:</b> preference data</li>
     *                 <li><b>Output:</b> folder in which to store the output</li>
     *                 <li><b>Num. Iter:</b> number of iterations. 0 if we want to apply until full coverage.</li>
     *                 <li><b>Directed:</b> true if the graph is directed, false otherwise</li>
     *                 <li><b>Resume:</b> true if we want to retrieve data from previous executions, false to overwrite</li>
     *                 <li><b>Not reciprocal:</b> true if we don't want to recommend reciprocal edges, false otherwise</li>
     *                 <li><b>Training data:</b> file containing the training data</li>
     *                 <li><b>Num. partitions:</b> number of partitions to make with the training data</li>
     *             </ol>
     *
     * @throws IOException           if something fails while reading / writing.
     * @throws UnconfiguredException if something fails while retrieving the algorithms.
     */
    public static void main(String[] args) throws IOException, UnconfiguredException
    {
        if (args.length < 9)
        {
            System.err.println("ERROR:iInvalid arguments");
            System.err.println("Usage:");
            System.err.println("Algorithms: configuration file for the algorithms");
            System.err.println("Input: preference data");
            System.err.println("Output: folder in which to store the output");
            System.err.println("Num. Iter: number of iterations. 0 if we want to apply until full coverage.");
            System.err.println("Directed: true if the graph is directed, false otherwise");
            System.err.println("Resume: true if we want to retrieve data from previous executions, false to overwrite");
            System.err.println("Not reciprocal: true if we don't want to recommend reciprocal edges, false otherwise");
            System.err.println("Training data: file containing the training data");
            System.err.println("Num. partitions: number of partitions to make with the training data");
            return;
        }

        String algorithms = args[0];
        String input = args[1];
        String output = args[2];
        int auxIter = Parsers.ip.parse(args[3]);
        boolean resume = args[4].equalsIgnoreCase("true");
        int numIter = (auxIter == 0) ? Integer.MAX_VALUE : auxIter;

        boolean directed = args[5].equalsIgnoreCase("true");
        boolean notReciprocal = !directed || args[6].equalsIgnoreCase("true");

        String trainingData = args[7];
        int numParts = Parsers.ip.parse(args[8]);

        Reader reader = new Reader();
        List<Tuple2<Integer, Integer>> train = reader.read(trainingData, "\t", true);


        // First, we identify and find the random seed which will be used for unties.
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

        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output + "rngseed"))))
        {
            bw.write("" + UntieRandomNumber.RNG);
        }

        // Read the ratings.
        Set<Long> users = new HashSet<>();
        List<Tuple3<Long, Long, Double>> triplets = new ArrayList<>();

        Graph<Long> graph;
        GraphReader<Long> greader = new TextGraphReader<>(directed, false, false, "\t", Parsers.lp);
        graph = greader.read(input);

        graph.getAllNodes().forEach(users::add);
        int numEdges = ((int) graph.getEdgeCount()) * (directed ? 1 : 2);
        int numRecipr = graph.getAllNodes().mapToInt(graph::getMutualNodesCount).sum();

        int numrel = numEdges - ((notReciprocal) ? numRecipr / 2 : 0);

        graph.getAllNodes().forEach(u -> graph.getAdjacentNodes(u).forEach(v -> triplets.add(new Tuple3<>(u, v, 1.0))));

        FastUpdateableUserIndex<Long> uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        FastUpdateableItemIndex<Long> iIndex = SimpleFastUpdateableItemIndex.load(users.stream());
        SimpleFastPreferenceData<Long, Long> prefData = SimpleFastPreferenceData.load(triplets.stream(), uIndex, iIndex);

        System.out.println("Num items:" + users.size());
        System.out.println("Num. users: " + prefData.numUsersWithPreferences());

        // Initialize the metrics to compute.
        Map<String, Supplier<CumulativeMetric<Long, Long>>> metrics = new HashMap<>();
        metrics.put("recall", () -> new CumulativeRecall<>(prefData, numrel, 0.5));
        metrics.put("gini", () -> new CumulativeGini<>(users.size()));

        List<String> metricNames = new ArrayList<>(metrics.keySet());

        List<String> algorithmNames = readAlgorithmList(algorithms, numParts);

        List<InteractiveRecommender<Long, Long>> recs = new ArrayList<>();
        // Select the algorithms
        long a = System.currentTimeMillis();
        AlgorithmSelector<Long, Long> algorithmSelector = new AlgorithmSelector<>();
        algorithmSelector.configure(uIndex, iIndex, prefData, 0.5, notReciprocal);
        for(String algorithm : algorithmNames)
        {
            InteractiveRecommender<Long, Long> rec = algorithmSelector.getAlgorithm(algorithm);
            recs.add(rec);
        }
        long b = System.currentTimeMillis();
        System.out.println("Recommenders prepared (" + (b - a) + " ms.)");

        int trainingSize = train.size();

        IntStream.range(0, numParts).parallel().forEach(part ->
        {
            // Step 1: Prepare the training data.
            long aaa = System.nanoTime();
            int val = trainingSize*(part+1);
            val /= numParts;

            List<Tuple2<Integer, Integer>> partTrain = train.subList(0, val);
            long bbb = System.nanoTime();

            System.out.println("Prepared training data " + (part+1) + "/" + numParts + ": " + val + " recommendations (" + (bbb-aaa)/1000000.0 + " ms.)");

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

            // Step 2: Retrieve the algorithm to apply:
            String algorithmName = algorithmNames.get(part);
            InteractiveRecommender<Long,Long> rec = recs.get(part);

            Map<String, CumulativeMetric<Long, Long>> localMetrics = new HashMap<>();
            metricNames.forEach(name -> localMetrics.put(name, metrics.get(name).get()));

            System.out.println("Starting algorithm " + algorithmName + " for the " + (part+1) + "/" + numParts + " split");

            RecommendationLoop<Long, Long> loop = new RecommendationLoop<>(uIndex, iIndex, prefData, rec, localMetrics, numIter, notReciprocal);
            loop.init(partTrain, true);

            bbb = System.nanoTime();
            System.out.println("Algorithm " + algorithmName + " for the " + (part+1) + "/" + numParts + " split has been initialized (" + (bbb-aaa)/1000000.0 + " ms.)");

            List<Tuple3<Integer, Integer, Long>> list = new ArrayList<>();
            String fileName = outputFolder + algorithmName + ".txt";

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
                bw.write("Num.Iter\tUser\tItem");
                for (String metric : metricNames)
                {
                    bw.write("\t" + metric);
                }
                bw.write("\tTime\n");

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
                    System.out.println("Algorithm " + algorithmName + " for the " + (part+1) + "/" + numParts + " split has finished recovering from previous executions (" + (bbb-aaa)/1000000.0 + " ms.)");
                }

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
            System.out.println("Algorithm " + algorithmName + " for the " + (part+1) + "/" + numParts + " split has finished (" + (bbb-aaa)/1000000.0 + " ms.)");
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
