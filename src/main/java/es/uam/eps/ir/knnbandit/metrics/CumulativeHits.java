/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.metrics;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;

import java.util.List;

/**
 * Counts the number of positive recommendations. Used for replayer loop, where metrics
 * are only updated when recommendations are.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class CumulativeHits<U,I> implements CumulativeMetric<U, I>
{
    private double counter;
    @Override
    public double compute()
    {
        return counter;
    }

    @Override
    public void initialize(Dataset<U, I> dataset)
    {
        counter = 0;
    }

    @Override
    public void initialize(Dataset<U, I> dataset, List<FastRating> train)
    {
        counter = 0;
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        counter += value;
    }

    @Override
    public void reset()
    {
        counter = 0;
    }
}
