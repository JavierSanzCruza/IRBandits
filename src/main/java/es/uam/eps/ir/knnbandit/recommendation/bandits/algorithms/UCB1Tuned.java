/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms;

import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.utils.Pair;
import it.unimi.dsi.fastutil.PriorityQueue;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import it.unimi.dsi.fastutil.objects.ObjectHeapPriorityQueue;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Comparator;

/**
 * Simple multi-armed bandit using the UCB1-tuned algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class UCB1Tuned extends AbstractMultiArmedBandit
{
    /**
     * The values for each arm.
     */
    double[] values;
    /**
     * The variances of each arm.
     */
    double[] variances;
    /**
     * The number of times each item has been selected.
     */
    int[] numTimes;
    /**
     * The number of iterations.
     */
    int numIter;

    /**
     * Constructor.
     *
     * @param numArms the number of arms.
     */
    public UCB1Tuned(int numArms)
    {
        super(numArms);
        this.values = new double[numArms];
        this.numTimes = new int[numArms];
        this.variances = new double[numArms];
        this.numIter = 0;
    }

    @Override
    public int next(int[] available, ValueFunction valF)
    {
        if (available == null || available.length == 0)
        {
            return -1;
        }
        else if (available.length == 1)
        {
            return available[0];
        }
        else
        {
            double max = Double.NEGATIVE_INFINITY;
            IntList top = new IntArrayList();
            for (int i : available)
            {
                double val;
                if (this.numTimes[i] == 0)
                {
                    val = Double.POSITIVE_INFINITY;
                }
                else
                {
                    double ucb = this.variances[i] - values[i] * values[i] + Math.sqrt(2 * Math.log(numIter + 1) / (numTimes[i]+0.0));
                    val = valF.apply(i, values[i] + Math.sqrt((Math.log(numIter + 1) / numTimes[i]+0.0) * Math.min(0.25, ucb)), numTimes[i]+0.0);
                }

                if (val > max)
                {
                    max = val;
                    top = new IntArrayList();
                    top.add(i);
                }
                else if (val == max)
                {
                    top.add(i);
                }
            }

            int item;
            int size = top.size();
            if (size == 1)
            {
                item = top.get(0);
            }
            else
            {
                item = top.get(untierng.nextInt(size));
            }
            return item;
        }
    }

    @Override
    public int next(IntList available, ValueFunction valF)
    {
        if (available == null || available.isEmpty())
        {
            return -1;
        }
        else if (available.size() == 1)
        {
            return available.get(0);
        }
        else
        {
            double max = Double.NEGATIVE_INFINITY;
            IntList top = new IntArrayList();
            for (int i : available)
            {
                double val;
                if (this.numTimes[i] == 0)
                {
                    val = Double.POSITIVE_INFINITY;
                }
                else
                {
                    double ucb = this.variances[i] - values[i] * values[i] + Math.sqrt(2 * Math.log(numIter + 1) / (numTimes[i]+0.0));
                    val = valF.apply(i, values[i] + Math.sqrt((Math.log(numIter + 1) / numTimes[i]+0.0) * Math.min(0.25, ucb)), numTimes[i]+0.0);
                }

                if (val > max)
                {
                    max = val;
                    top = new IntArrayList();
                    top.add(i);
                }
                else if (val == max)
                {
                    top.add(i);
                }
            }

            int item;
            int size = top.size();
            if (size == 1)
            {
                item = top.get(0);
            }
            else
            {
                item = top.get(untierng.nextInt(size));
            }
            return item;
        }
    }

    @Override
    public IntList next(IntList available, ValueFunction valFunc, int k)
    {
        if (available == null || available.isEmpty())
        {
            return new IntArrayList();
        }
        else
        {
            int num = Math.min(k, available.size());
            IntList top = new IntArrayList();

            PriorityQueue<Tuple2id> queue = new ObjectHeapPriorityQueue<>(num, Comparator.comparingDouble(x -> x.v2));

            for(int i : available)
            {
                double val;
                if (this.numTimes[i] == 0)
                {
                    val = Double.POSITIVE_INFINITY;
                }
                else
                {
                    double ucb = this.variances[i] - values[i] * values[i] + Math.sqrt(2 * Math.log(numIter + 1) / (numTimes[i]+0.0));
                    val = valFunc.apply(i, values[i] + Math.sqrt((Math.log(numIter + 1) / numTimes[i]+0.0) * Math.min(0.25, ucb)), numTimes[i]+0.0);
                }

                if(queue.size() < num)
                {
                    queue.enqueue(new Tuple2id(i, val));
                }
                else
                {
                    Tuple2id topElem = queue.dequeue();
                    Tuple2id newElem = new Tuple2id(i, val);
                    if(queue.comparator().compare(topElem, newElem) >= 0)
                    {
                        queue.enqueue(topElem);
                    }
                    else
                    {
                        queue.enqueue(newElem);
                    }
                }

            }

            while(!queue.isEmpty())
            {
                top.add(0, queue.dequeue().v1);
            }

            return top;
        }
    }

    @Override
    public void update(int i, double value)
    {
        double oldM = values[i];
        double oldS = variances[i];
        numTimes[i]++;
        numIter++;

        values[i] = oldM + (value - oldM) / (numTimes[i]+0.0);
        variances[i] = oldS + (value - oldM) * (value - values[i]+0.0);
    }

    @Override
    public void reset()
    {
        numIter = 0;
        for (int i = 0; i < numArms; ++i)
        {
            this.values[i] = 0.0;
            this.numTimes[i] = 0;
            this.variances[i] = 0.0;
        }
    }

    @Override
    public Pair<Integer> getStats(int arm)
    {
        if(arm < 0 || arm >= numArms) return null;

        int numHits = Double.valueOf(numTimes[arm]*values[arm]).intValue();
        return new Pair<>(numHits, numTimes[arm] - numHits);
    }

}
