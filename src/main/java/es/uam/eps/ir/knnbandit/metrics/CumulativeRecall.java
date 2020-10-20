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
import es.uam.eps.ir.knnbandit.data.datasets.GeneralOfflineDataset;
import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;

import java.util.List;

/**
 * Cumulative implementation of global recall.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado Puig (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class CumulativeRecall<U, I> implements CumulativeMetric<U, I>
{
    /**
     * Number of relevant (user,item) pairs.
     */
    private int numRel;
    /**
     * Relevance threshold.
     */
    private double threshold;
    /**
     * Number of training relevant (user,item) pairs.
     */
    private int toRemove;
    /**
     * Number of currently discovered (user, item) pairs.
     */
    private double current;

    /**
     * Constructor.
     *
     * @param numRel    Number of relevant (user, item) pairs.
     * @param threshold Relevance threshold.
     */
    public CumulativeRecall(int numRel, double threshold)
    {
        this.numRel = numRel;
        this.current = 0.0;
        this.toRemove = 0;
        this.threshold = threshold;
    }

    @Override
    public double compute()
    {
        if (numRel == 0)
        {
            return 0.0;
        }
        return this.current / (this.numRel - this.toRemove + 0.0);
    }


    @Override
    public void initialize(Dataset<U,I> dataset)
    {
        OfflineDataset<U,I> general = ((OfflineDataset<U,I>) dataset);
        this.numRel = general.getNumRel();
        this.current = 0.0;
    }

    @Override
    public void initialize(Dataset<U,I> dataset, List<FastRating> train)
    {
        OfflineDataset<U,I> general = ((OfflineDataset<U,I>) dataset);
        this.numRel = general.getNumRel();
        this.current = 0.0;
        this.toRemove = train.stream().mapToInt(pref -> pref.value() >= threshold ? 1 : 0).sum();
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        if(value >= threshold) this.current++;
    }

    @Override
    public void reset()
    {
        this.current = 0.0;
    }

}
