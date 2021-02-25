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
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.Random;

/**
 * Variable Epsilon-Greedy item bandit.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class EpsilonTGreedyItemBandit extends AbstractMultiArmedBandit
{
    /**
     * Slope parameter.
     */
    private final double alpha;
    /**
     * Random number generator.
     */
    private final Random rng = new Random();
    /**
     * Epsilon greedy update function.
     */
    private final EpsilonGreedyUpdateFunction updateFunction;
    /**
     * Values of each arm.
     */
    double[] values;
    /**
     * Number of times an arm has been selected.
     */
    int[] numTimes;
    /**
     * The sum of the values.
     */
    double sumValues;
    /**
     * Number of iterations.
     */
    private int numIter;

    /**
     * Constructor.
     *
     * @param numArms           the number of arms of the bandit.
     * @param alpha             slope parameter.
     * @param updateFunction    the update function for the arms.
     */
    public EpsilonTGreedyItemBandit(int numArms, double alpha, EpsilonGreedyUpdateFunction updateFunction)
    {
        super(numArms);
        this.alpha = alpha;
        this.sumValues = 0.0;
        this.values = new double[numArms];
        this.numTimes = new int[numArms];
        this.updateFunction = updateFunction;
        this.numIter = 1;
    }

    @Override
    public int next(int[] available, ValueFunction valF)
    {
        if (available == null || available.length == 0)
        {
            return -1;
        }
        if (available.length == 1)
        {
            return available[0];
        }
        else
        {
            double epsilon = Math.min(1.0, this.alpha * numArms / (numIter + 0.0));
            if (rng.nextDouble() < epsilon)
            {
                int item = untierng.nextInt(available.length);
                return available[item];
            }
            else
            {
                double max = Double.NEGATIVE_INFINITY;
                IntList top = new IntArrayList();

                for (int i : available)
                {
                    double val = valF.apply(i, values[i], numTimes[i]);
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
                int iidx;
                if (size == 1)
                {
                    iidx = top.get(0);
                }
                else
                {
                    iidx = top.get(untierng.nextInt(size));
                }

                return iidx;
            }
        }
    }

    @Override
    public int next(IntList available, ValueFunction valF)
    {
        if (available == null || available.isEmpty())
        {
            return -1;
        }
        if (available.size() == 1)
        {
            return available.get(0);
        }
        else
        {
            double epsilon = Math.min(1.0, this.alpha * numArms / (numIter + 0.0));
            if (rng.nextDouble() < epsilon)
            {
                int item = untierng.nextInt(available.size());
                return available.get(item);
            }
            else
            {
                double max = Double.NEGATIVE_INFINITY;
                IntList top = new IntArrayList();

                for (int i : available)
                {
                    double val = valF.apply(i, values[i], numTimes[i]+0.0);
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
                int iidx;
                if (size == 1)
                {
                    iidx = top.get(0);
                }
                else
                {
                    iidx = top.get(untierng.nextInt(size));
                }

                return iidx;
            }
        }
    }

    @Override
    public void update(int i, double increment)
    {
        double oldSum = this.sumValues;
        double nTimes = this.numTimes[i] + 1.0;
        double oldVal = this.values[i];

        numTimes[i]++;
        numIter++;
        double newVal = this.updateFunction.apply(oldVal, increment, oldSum, increment, nTimes);
        this.values[i] = newVal;
        this.sumValues += (newVal - oldVal);
    }

    @Override
    public void reset()
    {
        this.sumValues = 0.0;
        this.values = new double[numArms];
        this.numTimes = new int[numArms];
        this.numIter = 1;
    }

    @Override
    public Pair<Integer> getStats(int arm)
    {
        if(arm < 0 || arm >= numArms) return null;

        int numHits = Double.valueOf(numTimes[arm]*values[arm]).intValue();
        return new Pair<>(numHits, numTimes[arm] - numHits);
    }
}
