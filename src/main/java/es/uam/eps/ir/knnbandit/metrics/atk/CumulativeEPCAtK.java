/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.metrics.atk;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.Int2IntMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2LongOpenHashMap;

import java.util.List;

// TODO: Think how to compute this

/**
 * Cumulative version of Expected Popularity Complement at cutoff k
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class CumulativeEPCAtK<U, I> extends CumulativeMetricAtK<U, I>
{
    /**
     * Number of users.
     */
    private int numUsers;
    /**
     * Number of items
     */
    private int numItems;
    /**
     * A map containing the popularity of each item.
     */
    private final Int2IntMap popularities;
    /**
     * A map containing the frequency of the items in the top K.
     */
    private final Int2IntMap frequencies;
    /**
     * Current number of ratings
     */
    private double numRatings;
    /**
     * The EPC main sum.
     */
    private double sum;

    /**
     * Constructor.
     *
     * @param k number of recommendations to consider.
     */
    public CumulativeEPCAtK(int k, int numUsers, int numItems)
    {
        super(k);
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.sum = 0.0;
        this.numRatings = k;
        this.popularities = new Int2IntOpenHashMap();
        this.frequencies = new Int2IntOpenHashMap();
        this.popularities.defaultReturnValue(0);
        this.frequencies.defaultReturnValue(0);
    }

    @Override
    public void initialize(Dataset<U, I> dataset)
    {
        this.numUsers = dataset.numUsers();
        this.numItems = dataset.numItems();
        this.sum = 0.0;
        this.popularities.clear();
        this.frequencies.clear();
    }

    @Override
    public void initialize(Dataset<U,I> dataset, List<FastRating> train)
    {
        this.numUsers = dataset.numUsers();
        this.numItems = dataset.numItems();
        this.sum = 0.0;
        this.popularities.clear();
        this.frequencies.clear();


    }


    @Override
    protected void updateAdd(int uidx, int iidx)
    {
        int popularity = this.popularities.getOrDefault(iidx, this.popularities.defaultReturnValue());
        sum += popularity;

        ((Int2IntOpenHashMap) this.frequencies).addTo(iidx, 1);
        ((Int2IntOpenHashMap) this.popularities).addTo(iidx, 1);
    }

    @Override
    protected void updateDel(int uidx, int iidx)
    {
        // First, we obtain the frequency:
        int frequency = this.frequencies.getOrDefault(iidx, this.frequencies.defaultReturnValue());
        int popularity = this.popularities.getOrDefault(iidx, this.popularities.defaultReturnValue());
        if (frequency < 1)
        {
            return; // An error ocurred.
        }

        // as we are removing a single item, the popularity

        // Then: update the value of the sum.
        sum -= frequency * popularity;
        sum += (frequency - 1) * (popularity + 1);
        // And update the frequency of the item in the last k elements.
        this.frequencies.put(iidx, frequency - 1);

        ((Int2IntOpenHashMap) this.frequencies).addTo(iidx, -1);

    }

    @Override
    protected void resetMetric()
    {
        this.frequencies.clear();
        this.popularities.clear();
        this.sum = 0.0;
    }

    @Override
    public double compute()
    {
        if (this.numUsers <= 0 || this.numRatings <= 0)
        {
            return Double.NaN;
        }
        return 1 - this.sum / (this.numRatings * this.numUsers);
    }
}
