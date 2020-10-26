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
import org.jooq.lambda.tuple.Tuple2;

import java.util.ArrayList;
import java.util.List;


/**
 * Partitions the data uniformly.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class UniformPartition implements Partition
{
    @Override
    public List<Integer> split(Dataset<?,?> dataset, List<Pair<Integer>> trainingData, int numParts)
    {
        List<Integer> splitPoints = new ArrayList<>();
        int size = trainingData.size();
        for (int part = 1; part <= numParts; ++part)
        {
            int point = (size * part) / numParts;
            splitPoints.add(point);
        }

        return splitPoints;
    }

    @Override
    public int split(Dataset<?,?> dataset, List<Pair<Integer>> trainingData, double percentage)
    {
        int size = trainingData.size();
        Double point = percentage * size;
        return point.intValue();
    }
}
