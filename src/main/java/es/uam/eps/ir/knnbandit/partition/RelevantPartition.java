/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.partition;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import org.jooq.lambda.tuple.Tuple2;
import org.ranksys.fast.preference.FastPointWisePreferenceData;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.DoublePredicate;

/**
 * Partitions the data acording to the relevant ratings.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RelevantPartition implements Partition
{
    @Override
    public List<Integer> split(Dataset<?,?> dataset, List<Pair<Integer>> trainingData, int numParts)
    {
        List<Integer> splitPoints = new ArrayList<>();

        // Count the total number of relevant pairs
        int numRel = trainingData.stream().mapToInt(tuple ->
        {
            Optional<Double> optional = dataset.getPreference(tuple.v1(), tuple.v2());
            return (optional.isPresent() && dataset.getRelevanceChecker().test(optional.get())) ? 1 : 0;
        }).sum();

        int nextPoint = numRel / numParts;
        int counter = 1;
        int i = 0;
        int j = 0;
        for (Pair<Integer> tuple : trainingData)
        {
            Optional<Double> optional = dataset.getPreference(tuple.v1(), tuple.v2());
            if (optional.isPresent() && dataset.getRelevanceChecker().test(optional.get()))
            {
                i++;
            }

            ++j;
            if (i == nextPoint && counter < numParts)
            {
                splitPoints.add(j);
                counter++;
                nextPoint = numRel * (counter) / numParts;
            }
        }
        splitPoints.add(j);
        return splitPoints;
    }

    @Override
    public int split(Dataset<?,?> dataset, List<Pair<Integer>> trainingData, double percentage)
    {
        int size = trainingData.size();

        // Count the total number of relevant pairs
        int numRel = trainingData.stream().mapToInt(tuple ->
        {
            Optional<Double> optional = dataset.getPreference(tuple.v1(), tuple.v2());
            return (optional.isPresent() && dataset.getRelevanceChecker().test(optional.get())) ? 1 : 0;
        }).sum();

        Double point = percentage * numRel;
        int splitPoint = point.intValue();

        int count = 0;
        int j = 0;
        for (Pair<Integer> tuple : trainingData)
        {
            Optional<Double> optional = dataset.getPreference(tuple.v1(), tuple.v2());
            if (optional.isPresent() && dataset.getRelevanceChecker().test(optional.get()))
            {
                count++;
            }
            ++j;
            if (count == splitPoint)
            {
                return j;
            }
        }

        return point.intValue();
    }
}
