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
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2LongMap;
import it.unimi.dsi.fastutil.ints.Int2LongOpenHashMap;
import org.jooq.lambda.tuple.Tuple2;

import java.util.List;


// TODO: Finish

/**
 * Cumulative Expected Popularity Complement (EPC) metric. Finds how popular are the different
 * recommended items.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class CumulativeEPC<U, I> implements CumulativeMetric<U, I>
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
    private final Int2LongMap popularities;
    /**
     * Current number of ratings
     */
    private double numRatings;
    /**
     * The EPC main sum.
     */
    private double sum;
    /**
     * The value for EPC for the previous iteration (which is the value that must be returned).
     */
    private double epcValue;

    /**
     * Constructor.
     *
     * @param numUsers the number of users.
     * @param numItems the number of items.
     */
    public CumulativeEPC(int numUsers, int numItems)
    {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.numRatings = 0.0;
        this.popularities = new Int2LongOpenHashMap();
        this.popularities.defaultReturnValue(0L);
        this.epcValue = Double.NaN;
        this.sum = 0.0;
    }

    @Override
    public void initialize(Dataset<U,I> dataset)
    {
        this.numUsers = dataset.numUsers();
        this.numItems = dataset.numItems();
        this.numRatings = 0.0;
        this.popularities.clear();
        this.epcValue = Double.NaN;
        this.sum = 0.0;
    }

    @Override
    public void initialize(Dataset<U,I> dataset, List<FastRating> train)
    {
        this.initialize(dataset);
        // Initialize the popularity values.
        train.forEach(rating -> ((Int2LongOpenHashMap) this.popularities).addTo(rating.iidx(), 1));
    }

    @Override
    public double compute()
    {
        return epcValue;
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        if (numUsers > 0 && numRatings > 0.0)
        {
            this.epcValue = 1 - 1 / (numUsers * numRatings) * sum;
        }

        long pop = this.popularities.getOrDefault(iidx, this.popularities.defaultReturnValue());
        sum += 2 * pop + 1;
        numRatings++;
        this.popularities.put(iidx, pop + 1);
    }

    @Override
    public void reset()
    {
        this.numRatings = 0.0;
        this.popularities.clear();
        this.sum = 0.0;
        this.epcValue = Double.NaN;
    }
}
