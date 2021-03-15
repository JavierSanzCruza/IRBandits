/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.basic;

import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Interactive version of an average rating recommendation algorithm.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class AvgRecommender<U, I> extends AbstractBasicInteractiveRecommender<U, I>
{
    /**
     * Number of times an arm has been selected.
     */
    private double[] numTimes;

    private double numIter;

    private double mu;
    private double p;

    /**
     * Constructor.
     *
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     */
    public AvgRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated)
    {
        super(uIndex, iIndex, ignoreNotRated);
        this.numTimes = new double[iIndex.numItems()];
        IntStream.range(0, iIndex.numItems()).forEach(iidx -> this.numTimes[iidx] = 0);
    }

    /**
     * Constructor.
     *
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param rngSeed   Random number generator seed.
     */
    public AvgRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed);
        this.numTimes = new double[iIndex.numItems()];
        IntStream.range(0, iIndex.numItems()).forEach(iidx -> this.numTimes[iidx] = 0);
    }

    @Override
    public void init()
    {
        super.init();
        IntStream.range(0, iIndex.numItems()).forEach(iidx ->
        {
            this.numTimes[iidx] = 0.0;
            this.values[iidx] = 0.0;
        });

        this.mu = 0.0;
        this.p = 0.0;
        this.numIter = 0;
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();
        values.forEach(triplet ->
        {
            int iidx = triplet.iidx();
            double oldvalue = this.values[iidx];
            this.values[iidx] = oldvalue + triplet.value();
            this.numTimes[iidx] += 1.0;
            this.numIter++;

            this.p += triplet.value();
            this.mu += 1.0;
        });
    }

    @Override
    public int next(int uidx, IntList availability)
    {
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }
        else
        {
            double val = Double.NEGATIVE_INFINITY;
            IntList top = new IntArrayList();

            for (int item : availability)
            {
                double actP = p/(numItems()+0.0);
                double actMu = mu/(numItems()+0.0);

                double value = (values[item] + actP)/(this.numTimes[item]+actMu);

                if (value > val)
                {
                    val = value;
                    top = new IntArrayList();
                    top.add(item);
                }
                else if (value == val)
                {
                    top.add(item);
                }
            }

            int nextItem;
            int size = top.size();
            if (size == 1)
            {
                nextItem = top.get(0);
            }
            else
            {
                nextItem = top.get(rng.nextInt(size));
            }

            return nextItem;
        }
    }

    @Override
    public IntList next(int uidx, IntList availability, int k)
    {
        if (availability == null || availability.isEmpty())
        {
            return new IntArrayList();
        }
        else
        {
            IntList top = new IntArrayList();

            int num = Math.min(availability.size(), k);
            PriorityQueue<Tuple2id> queue = new PriorityQueue<>(num, Comparator.comparingDouble(x -> x.v2));

            for (int iidx : availability)
            {
                double actP = p/(numItems()+0.0);
                double actMu = mu/(numItems()+0.0);

                double value = (values[iidx] + actP)/(this.numTimes[iidx]+actMu);

                if(queue.size() < num)
                {
                    queue.add(new Tuple2id(iidx, value));
                }
                else
                {
                    Tuple2id newTuple = new Tuple2id(iidx, value);
                    if(queue.comparator().compare(queue.peek(), newTuple) < 0)
                    {
                        queue.poll();
                        queue.add(newTuple);
                    }
                }
            }

            while(!queue.isEmpty())
            {
                top.add(0, queue.poll().v1);
            }

            return top;
        }
    }

    @Override
    public void fastUpdate(int uidx, int iidx, double value)
    {
        double newValue;
        if(!Double.isNaN(value) && value != Constants.NOTRATEDRATING)
            newValue = value;
        else if(!ignoreNotRated)
            newValue = Constants.NOTRATEDNOTIGNORED;
        else
            return;


        double oldValue = values[iidx];
        if (numTimes[iidx] <= 0.0)
        {
            this.values[iidx] = newValue;
        }
        else
        {
            this.values[iidx] = oldValue + newValue;
        }
        this.p += newValue;
        this.mu += 1.0;
        this.numIter++;
        this.numTimes[iidx]++;
    }
}
