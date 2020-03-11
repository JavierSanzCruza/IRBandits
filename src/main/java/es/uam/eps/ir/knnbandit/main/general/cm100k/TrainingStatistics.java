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

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.io.Reader;
import es.uam.eps.ir.knnbandit.partition.Partition;
import es.uam.eps.ir.knnbandit.partition.RelevantPartition;
import es.uam.eps.ir.knnbandit.partition.UniformPartition;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

/**
 * Class for computing the statistics for training data.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class TrainingStatistics
{
    /**
     * Program for providing training statistics for recommendation.
     *
     * @param args Program arguments
     *             <ul>
     *              <li><b>Input:</b> Input data. Contains all the ratings to consider</li>
     *              <li><b>Threshold:</b> Relevance threshold</li>
     *              <li><b>Training data:</b>File containing the training data (basically, a previous execution of a recommender over no training data)</li>
     *              <li><b>Num. splits:</b> Number of splits of the training data.</li>
     *             </ul>
     */
    public static void main(String[] args) throws IOException
    {
        if (args.length < 4)
        {
            System.err.println("ERROR: Invalid arguments");
            System.err.println("Usage:");
            System.err.println("\tInput: input data");
            System.err.println("\tThreshold: relevance threshold");
            System.err.println("\tTraining data: file containing the training data (basically, a previous execution of a recommender over no training data");
            System.err.println("\tNum. splits: Number of splits");
        }


        String testFile = args[0];
        double threshold = Parsers.dp.parse(args[1]);
        String trainingFile = args[2];
        int auxNumParts = Parsers.ip.parse(args[3]);
        boolean relevantPartition = auxNumParts < 0;
        int numParts = Math.abs(auxNumParts);

        DoubleUnaryOperator weightFunction = (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = (double x) -> (x > 0.0);

        // Read the training data.
        Reader reader = new Reader();
        List<Tuple2<Integer, Integer>> train = reader.read(trainingFile, "\t", true);

        Set<Long> users = new HashSet<>();
        Set<Long> items = new HashSet<>();

        List<Tuple3<Long, Long, Double>> triplets = new ArrayList<>();

        int numrel = 0;
        int numrat = 0;

        // Read the whole data.
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(testFile))))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] split = line.split("::");
                Long user = Parsers.lp.parse(split[0]);
                Long item = Parsers.lp.parse(split[1]);
                double val = Parsers.dp.parse(split[2]);

                users.add(user);
                items.add(item);

                double rating = weightFunction.applyAsDouble(val);
                if (relevance.test(rating))
                {
                    numrel++;
                }

                numrat++;

                triplets.add(new Tuple3<>(user, item, rating));
            }
        }

        FastUpdateableUserIndex<Long> uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        FastUpdateableItemIndex<Long> iIndex = SimpleFastUpdateableItemIndex.load(items.stream());
        SimpleFastPreferenceData<Long, Long> prefData = SimpleFastPreferenceData.load(triplets.stream(), uIndex, iIndex);

        // Print the general data:
        System.out.println("General information: ");
        System.out.println("Users\tItems\tRatings\tRel.Ratings");
        System.out.println(uIndex.numUsers() + "\t" + iIndex.numItems() + "\t" + numrat + "\t" + numrel);

        int trainingSize = train.size();

        // Then, for each split:
        System.out.println("Training");
        System.out.println("Num.Split\tNum.Recs\tRatings\tRel.Ratings");

        Partition partition = relevantPartition ? new RelevantPartition(prefData, relevance) : new UniformPartition();
        List<Integer> splitPoints = partition.split(train, numParts);

        for (int part = 0; part < numParts; ++part)
        {
            int val = splitPoints.get(part);
            List<Tuple2<Integer, Integer>> partTrain = train.subList(0, val);

            long trainCount = partTrain.stream().filter(t ->
            {
                if (prefData.numItems(t.v1) > 0)
                {
                    Optional<IdxPref> pref = prefData.getPreference(t.v1, t.v2);
                    return pref.isPresent();
                }
                return false;
            }).count();

            long trainRelCount = partTrain.stream().filter(t ->
            {
                if (prefData.numItems(t.v1) > 0)
                {
                    Optional<IdxPref> pref = prefData.getPreference(t.v1, t.v2);
                    return pref.isPresent() && pref.get().v2 > 0.0;
                }
                return false;
            }).count();

            System.out.println((part + 1) + "\t" + val + "\t" + trainCount + "\t" + trainRelCount);
        }
    }
}
