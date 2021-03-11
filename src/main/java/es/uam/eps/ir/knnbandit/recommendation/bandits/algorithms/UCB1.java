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
 * Simple multi-armed bandit using the UCB1 algorithm.
 *
 * @author Javier Sanz-Cruzado Puig (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class UCB1 extends AbstractMultiArmedBandit
{
    /**
     * The values for each user.
     */
    private double[] values;
    /**
     * The number of times each item has been selected.
     */
    private int[] numTimes;
    /**
     * The number of iterations.
     */
    private int numIter;

    /**
     * A parameter regulating the upper confidence bound for the algorithm.
     */
    private final double alpha;

    /**
     * Constructor.
     *
     * @param numArms the number of arms.
     */
    public UCB1(int numArms)
    {
        super(numArms);
        this.values = new double[numArms];
        this.numTimes = new int[numArms];
        this.alpha = 2.0;
    }

    /**
     * Constructor.
     *
     * @param numArms the number of arms.
     * @param alpha   the value of the parameter that regulates the upper confidence bound.
     */
    public UCB1(int numArms, double alpha)
    {
        super(numArms);
        this.values = new double[numArms];
        this.numTimes = new int[numArms];
        this.alpha = alpha;
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
                    val = valF.apply(i, values[i] + Math.sqrt(alpha * Math.log(numIter + 1) / (numTimes[i]+0.0)), numTimes[i]+0.0);
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

            int size = top.size();
            int item;
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
                    val = valF.apply(i, values[i] + Math.sqrt(alpha * Math.log(numIter + 1) / (numTimes[i]+0.0)), numTimes[i]+0.0);
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

            int size = top.size();
            int item;
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
                if(this.numTimes[i] == 0)
                    val = Double.POSITIVE_INFINITY;
                else
                    val = valFunc.apply(i, values[i] + Math.sqrt(alpha * Math.log(numIter + 1) / (numTimes[i]+0.0)), numTimes[i]+0.0);

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
        numTimes[i]++;
        numIter++;
        values[i] = values[i] + 1.0 / (numTimes[i] + 0.0) * (value - values[i]);
    }

    @Override
    public void reset()
    {
        this.values = new double[numArms];
        this.numTimes = new int[numArms];
    }

    @Override
    public Pair<Integer> getStats(int arm)
    {
        if(arm < 0 || arm >= numArms) return null;

        int numHits = Double.valueOf(numTimes[arm]*values[arm]).intValue();
        return new Pair<>(numHits, numTimes[arm] - numHits);
    }
}
