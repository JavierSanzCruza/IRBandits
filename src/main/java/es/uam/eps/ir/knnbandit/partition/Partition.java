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

import java.util.List;

/**
 * Interface for partitioning training data.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface Partition
{
    /**
     * Given a list of tuples, divides it in a given number of parts.
     *
     * @param trainingData the training data.
     * @param numParts     the number of parts.
     * @return a list containing the split points.
     */
    List<Integer> split(Dataset<?,?> dataset, List<Pair<Integer>> trainingData, int numParts);

    /**
     * Given a list of tuples, divides it in two parts given a percentage.
     *
     * @param trainingData the training data.
     * @param percentage   the percentage of training.
     * @return the split point.
     */
    int split(Dataset<?,?> dataset, List<Pair<Integer>> trainingData, double percentage);
}
