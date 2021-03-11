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
import es.uam.eps.ir.knnbandit.stats.BetaDistribution;
import es.uam.eps.ir.knnbandit.utils.Pair;
import it.unimi.dsi.fastutil.PriorityQueue;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import it.unimi.dsi.fastutil.objects.ObjectHeapPriorityQueue;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Comparator;

/**
 * Multi-armed bandit using the Thompson sampling algorithm.
 * It considers that rewards of each arm follow a Bernoulli distribution.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ThompsonSampling extends AbstractMultiArmedBandit
{
    /**
     * A Beta distribution for each possible arm.
     */
    private final BetaDistribution[] betas;

    /**
     * The initial alphas for each arm.
     */
    private final double[] initialAlphas;
    /**
     * The initial betas for each arm.
     */
    private final double[] initialBetas;
    /**
     * The unique initial alpha value for all the arms.
     */
    private final double initialAlpha;
    /**
     * The unique initial beta value for all the arms.
     */
    private final double initialBeta;

    /**
     * Constructor.
     *
     * @param numArms The number of arms.
     */
    public ThompsonSampling(int numArms)
    {
        super(numArms);
        this.betas = new BetaDistribution[numArms];
        for (int i = 0; i < numArms; ++i)
        {
            betas[i] = new BetaDistribution(1.0, 1.0);
        }

        this.initialAlpha = 1.0;
        this.initialBeta = 1.0;
        this.initialAlphas = null;
        this.initialBetas = null;
    }

    /**
     * Constructor.
     *
     * @param numArms      Number of arms.
     * @param initialAlpha The initial value for the alpha parameter of Beta distributions.
     * @param initialBeta  The initial value for the beta parameter of the Beta distributions.
     */
    public ThompsonSampling(int numArms, double initialAlpha, double initialBeta)
    {
        super(numArms);
        this.betas = new BetaDistribution[numArms];
        for (int i = 0; i < numArms; ++i)
        {
            betas[i] = new BetaDistribution(initialAlpha, initialBeta);
        }

        this.initialAlpha = initialAlpha;
        this.initialBeta = initialBeta;
        this.initialAlphas = null;
        this.initialBetas = null;
    }

    /**
     * Constructor.
     *
     * @param numArms       Number of arms.
     * @param initialAlphas The initial values for the alpha parameters of Beta distributions.
     * @param initialBetas  The initial values for the beta parameters of Beta distributions.
     */
    public ThompsonSampling(int numArms, double[] initialAlphas, double[] initialBetas)
    {
        super(numArms);
        this.betas = new BetaDistribution[numArms];
        for (int i = 0; i < numArms; ++i)
        {
            betas[i] = new BetaDistribution(initialAlphas[i], initialBetas[i]);
        }

        this.initialAlpha = 1.0;
        this.initialBeta = 1.0;
        this.initialAlphas = initialAlphas;
        this.initialBetas = initialBetas;
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
                double val = valF.apply(i, this.betas[i].sample(), 0);
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
            if (size == 1)
            {
                return top.get(0);
            }
            else
            {
                return top.get(untierng.nextInt(size));
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
                double val = valF.apply(i, this.betas[i].sample(), 0);
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
            if (size == 1)
            {
                return top.get(0);
            }
            else
            {
                return top.get(untierng.nextInt(size));
            }
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
                double val = valFunc.apply(i, this.betas[i].sample(), 0);
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
        this.betas[i].updateAdd(value, (1.0 - value));
    }

    @Override
    public void reset()
    {
        if (initialAlphas == null || initialBetas == null)
        {
            for (int i = 0; i < numArms; ++i)
            {
                betas[i] = new BetaDistribution(initialAlpha, initialBeta);
            }
        }
        else
        {
            for (int i = 0; i < numArms; ++i)
            {
                betas[i] = new BetaDistribution(initialAlphas[i], initialBetas[i]);
            }
        }
    }

    @Override
    public Pair<Integer> getStats(int arm)
    {
        if(arm < 0 || arm >= numArms) return null;

        if(initialAlphas == null || initialBetas == null)
        {
            int numHits = Double.valueOf(this.betas[arm].getAlpha() - initialAlpha).intValue();
            int numMisses = Double.valueOf(this.betas[arm].getBeta() - initialBeta).intValue();
            return new Pair<>(numHits, numMisses);
        }
        else
        {
            int numHits = Double.valueOf(this.betas[arm].getAlpha() - initialAlphas[arm]).intValue();
            int numMisses = Double.valueOf(this.betas[arm].getBeta() - initialBetas[arm]).intValue();
            return new Pair<>(numHits, numMisses);
        }
    }
}
