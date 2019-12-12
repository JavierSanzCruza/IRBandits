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
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parsers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Class for computing the statistics for training data.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class TrainingContactStatistics
{
    /**
     * Program for providing training statistics for contact recommendation.
     *
     * @param args Program arguments
     *             <ul>
     *              <li><b>Input:</b> Input data. Contains all the ratings to consider</li>
     *              <li><b>Training data:</b>File containing the training data (basically, a previous execution of a recommender over no training data)</li>
     *              <li><b>Num. splits:</b> Number of splits of the training data.</li>
     *              <li><b>Directed:</b> True if the network is directed, false otherwise</li>
     *              <li><b>Not reciprocal:</b> True if we want to consider reciprocal links as one single link, false otherwise</li>
     *             </ul>
     */
    public static void main(String[] args) throws IOException
    {
        if (args.length < 5)
        {
            System.err.println("ERROR: Invalid arguments");
            System.err.println("Usage:");
            System.err.println("\tInput: input data");
            System.err.println("\tThreshold: relevance threshold");
            System.err.println("\tTraining data: file containing the training data (basically, a previous execution of a recommender over no training data");
            System.err.println("\tNum. splits: Number of splits");
        }

        String testFile = args[0];
        String trainingFile = args[1];
        int numSplits = Parsers.ip.parse(args[2]);
        boolean directed = args[3].equalsIgnoreCase("true");
        boolean notReciprocal = !directed || args[4].equalsIgnoreCase("true");

        Set<Long> users = new HashSet<>();
        List<Tuple3<Long, Long, Double>> triplets = new ArrayList<>();

        Graph<Long> graph;
        GraphReader<Long> greader = new TextGraphReader<>(directed, false, false, "\t", Parsers.lp);
        graph = greader.read(testFile);

        graph.getAllNodes().forEach(users::add);
        int numEdges = new Long(graph.getEdgeCount()).intValue() * (directed ? 1 : 2);
        int numRecipr = graph.getAllNodes().mapToInt(graph::getMutualNodesCount).sum();
        int numrel = numEdges - ((notReciprocal) ? numRecipr / 2 : 0);

        // Read the training data.
        Reader reader = new Reader();
        List<Tuple2<Integer, Integer>> train = reader.read(trainingFile, "\t", true);
        graph.getAllNodes().forEach(u ->
        {
            graph.getAdjacentNodes(u).forEach(v ->
            {
                triplets.add(new Tuple3<>(u, v, 1.0));
            });
        });

        FastUpdateableUserIndex<Long> uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        FastUpdateableItemIndex<Long> iIndex = SimpleFastUpdateableItemIndex.load(users.stream());
        SimpleFastPreferenceData<Long, Long> prefData = SimpleFastPreferenceData.load(triplets.stream(), uIndex, iIndex);

        // Print the general data:
        System.out.println("General information: ");
        System.out.println("Users\tItems\tRatings\tRel.Ratings");
        System.out.println(uIndex.numUsers() + "\t" + iIndex.numItems() + "\t" + numrel + "\t" + numrel);

        int trainingSize = train.size();

        // Then, for each split:
        System.out.println("Training");
        System.out.println("Num.Split\tNum.Recs\tRatings\tRel.Ratings");

        for (int part = 0; part < numSplits; ++part)
        {
            int val = trainingSize * (part + 1);
            val /= numSplits;

            List<Tuple2<Integer, Integer>> partTrain = train.subList(0, val);

            long auxRelCount = partTrain.stream().filter(t ->
            {
                return prefData.numItems(t.v1) == 0;
            }).count();

            long trainRelCount = partTrain.stream().filter(t ->
            {
                return prefData.numItems(t.v1) > 0 && prefData.getPreference(t.v1, t.v2).isPresent();
            }).count();


            System.out.println((part + 1) + "\t" + val + "\t" + trainRelCount + "\t" + trainRelCount + "\t" + auxRelCount);
        }
    }
}
