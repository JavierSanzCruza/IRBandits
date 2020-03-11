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
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.userknowledge.fast.SimpleFastUserKnowledgePreferenceData;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
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
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.jooq.lambda.tuple.Tuple4;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.*;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Class for executing recommender systems in simulated interactive loops.
 *
 * @author Javier Sanz-Cruzado Puig (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class InteractiveRecommendation
{
    /**
     * Executes recommendation algorithms in simulated interactive loops.
     *
     * @param args Execution arguments:
     *             <ol>
     *                 <li><b>Algorithms:</b> configuration file for the algorithms</li>
     *                 <li><b>Input:</b> preference data</li>
     *                 <li><b>Output:</b> folder in which to store the output</li>
     *                 <li><b>Num. Iter:</b> number of iterations. 0 if we want to apply until full coverage.</li>
     *                 <li><b>Threshold:</b> relevance threshold</li>
     *                 <li><b>Resume:</b> true if we want to retrieve data from previous executions, false to overwrite</li>
     *                 <li><b>Use ratings:</b> true if we want to use ratings, false for binary values</li>
     *             </ol>
     *
     * @throws IOException           if something fails while reading / writing.
     * @throws UnconfiguredException if something fails while retrieving the algorithms.
     */
    public static void main(String[] args) throws IOException, UnconfiguredException
    {
        if (args.length < 7)
        {
            System.err.println("ERROR: Invalid arguments");
            System.err.println("Usage:");
            System.err.println("\tAlgorithms: recommender systems list");
            System.err.println("\tInput: Preference data input");
            System.err.println("\tOutput: folder in which to store the output");
            System.err.println("\tresume: true if we want to resume previous executions, false if we want to overwrite");
            System.err.println("\tNum. Iter.: number of iterations. 0 if we want to run until we run out of recommendable items");
            System.err.println("\tThreshold: relevance threshold");
            System.err.println("\tUse ratings: true if we want to take the true value of the ratings, false if we want to use binary values");
            System.err.println("\tKnowledge data: ALL, KNOWN or UNKNOWN");
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

        // General recommendation features.
        double threshold = Parsers.dp.parse(args[5]);
        boolean useRatings = args[6].equalsIgnoreCase("true");
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x :
                (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
        double realThreshold = useRatings ? threshold : 0.5;

        KnowledgeDataUse dataUse = KnowledgeDataUse.fromString(args[7]);

        // Configure the random seed for unties:
        UntieRandomNumber.configure(resume, output);

        // Then, we read the ratings.
        Set<Long> users = new HashSet<>();
        Set<Long> items = new HashSet<>();
        List<Tuple4<Long, Long, Double, Boolean>> triplets = new ArrayList<>();
        int numrel = 0;
        int numrelknown = 0;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(input))))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] split = line.split("::");
                Long user = Parsers.lp.parse(split[0]);
                Long item = Parsers.lp.parse(split[1]);
                double val = Parsers.dp.parse(split[2]);
                boolean known = split[3].equals("1");

                users.add(user);
                items.add(item);

                double rating = weightFunction.applyAsDouble(val);
                if (relevance.test(rating))
                {
                    numrel++;
                    if(known) numrelknown++;
                }

                triplets.add(new Tuple4<>(user, item, rating, known));
            }
        }

        // Create the data.
        FastUpdateableUserIndex<Long> uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        FastUpdateableItemIndex<Long> iIndex = SimpleFastUpdateableItemIndex.load(items.stream());

        SimpleFastUserKnowledgePreferenceData<Long, Long> knowledgeData = SimpleFastUserKnowledgePreferenceData.load(triplets.stream(), uIndex, iIndex);
        SimpleFastPreferenceData<Long, Long> prefData = (SimpleFastPreferenceData<Long,Long>) knowledgeData.getPreferenceData();
        SimpleFastPreferenceData<Long, Long> knownData = (SimpleFastPreferenceData<Long,Long>) knowledgeData.getKnownPreferenceData();
        SimpleFastPreferenceData<Long, Long> unknownData = (SimpleFastPreferenceData<Long,Long>) knowledgeData.getUnknownPreferenceData();



        System.out.println("Users: " + uIndex.numUsers());
        System.out.println("Items: " + iIndex.numItems());
        System.out.println("Num. relevant: " + numrel);
        System.out.println("Num. relevant known: " + numrelknown);
        System.out.println("Num. relevant unknown: " + (numrel - numrelknown));
        int numRel = numrel;
        int numRelKnown = numrelknown;

        // Initialize the metrics:
        Map<String, Supplier<CumulativeMetric<Long, Long>>> metrics = new HashMap<>();
        metrics.put("recall", () -> new CumulativeRecall<>(prefData, numRel, realThreshold));
        metrics.put("recall-known", () -> new CumulativeRecall<>(knownData, numRelKnown, realThreshold));
        metrics.put("recall-unknown", () -> new CumulativeRecall<>(unknownData, numRel-numRelKnown, realThreshold));
        metrics.put("gini", () -> new CumulativeGini<>(items.size()));
        List<String> metricNames = new ArrayList<>(metrics.keySet());

        // Select the algorithms.
        long a = System.currentTimeMillis();
        AlgorithmSelector<Long, Long> algorithmSelector = new AlgorithmSelector<>();
        algorithmSelector.configure(uIndex, iIndex, prefData, realThreshold, knowledgeData, dataUse);
        algorithmSelector.addFile(algorithms);
        Map<String, InteractiveRecommender<Long, Long>> recs = algorithmSelector.getRecs();
        long b = System.currentTimeMillis();
        System.out.println("Recommenders ready (" + (b - a) + " ms.)");

        // Execute the recommendations
        recs.entrySet().parallelStream().forEach(re ->
        {
            InteractiveRecommender<Long, Long> rec = re.getValue();
            Map<String, CumulativeMetric<Long, Long>> localMetrics = new HashMap<>();
            metricNames.forEach(name -> localMetrics.put(name, metrics.get(name).get()));

            // Configure, and initialize the recommendation loop:
            EndCondition endcond = iterationsStop ? (auxIter == 0.0 ? new NoLimitsEndCondition() : new NumIterEndCondition(numIter)) : new PercentagePositiveRatingsEndCondition(numRel, auxIter, realThreshold);
            RecommendationLoop<Long, Long> loop = new RecommendationLoop<>(uIndex, iIndex, prefData, rec, localMetrics, endcond, UntieRandomNumber.RNG, false);
            loop.init(false);

            List<Tuple3<Integer, Integer, Long>> list = new ArrayList<>();
            String fileName = output + re.getKey() + ".txt";

            // If there have been previous executions, retrieve them.
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

            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output + re.getKey() + ".txt"))))
            {
                // First, we write the header.
                bw.write("Num.Iter\tUser\tItem");
                for (String metric : metricNames)
                {
                    bw.write("\t" + metric);
                }
                bw.write("\tTime\n");

                // Then, if we have retrieved previous iterations, recompute them.
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

                // Until the loop ends.
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
        });
    }
}
