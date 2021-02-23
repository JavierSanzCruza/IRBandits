/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.bandits.item;

import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.stats.BetaDistribution;
import it.unimi.dsi.fastutil.PriorityQueue;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import it.unimi.dsi.fastutil.objects.ObjectHeapPriorityQueue;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Comparator;

/**
 * Item bandit using the Thompson sampling algorithm.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ThompsonSamplingItemBandit<U, I> extends ItemBandit<U, I>
{
    /**
     * A Beta distribution for each possible item.
     */
    private final BetaDistribution[] betas;

    /**
     * The number of items.
     */
    private final int numItems;

    private final double[] initialAlphas;
    private final double[] initialBetas;
    private final double initialAlpha;
    private final double initialBeta;

    /**
     * Constructor.
     *
     * @param numItems The number of items.
     */
    public ThompsonSamplingItemBandit(int numItems)
    {
        this.numItems = numItems;
        this.betas = new BetaDistribution[numItems];
        for (int i = 0; i < numItems; ++i)
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
     * @param numItems     Number of items.
     * @param initialAlpha The initial value for the alpha parameter of Beta distributions.
     * @param initialBeta  The initial value for the beta parameter of the Beta distributions.
     */
    public ThompsonSamplingItemBandit(int numItems, double initialAlpha, double initialBeta)
    {
        this.numItems = numItems;
        this.betas = new BetaDistribution[numItems];
        for (int i = 0; i < numItems; ++i)
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
     * @param numItems      Number of items.
     * @param initialAlphas The initial values for the alpha parameters of Beta distributions.
     * @param initialBetas  The initial values for the beta parameters of Beta distributions.
     */
    public ThompsonSamplingItemBandit(int numItems, double[] initialAlphas, double[] initialBetas)
    {
        this.numItems = numItems;
        this.betas = new BetaDistribution[numItems];
        for (int i = 0; i < numItems; ++i)
        {
            betas[i] = new BetaDistribution(initialAlphas[i], initialBetas[i]);
        }

        this.initialAlpha = 1.0;
        this.initialBeta = 1.0;
        this.initialAlphas = initialAlphas;
        this.initialBetas = initialBetas;
    }

    @Override
    public int next(int uidx, int[] available, ValueFunction valF)
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
                double val = valF.apply(uidx, i, this.betas[i].sample(), 0);
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
    public int next(int uidx, IntList available, ValueFunction valF)
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
                double val = valF.apply(uidx, i, this.betas[i].sample(), 0);
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
    public IntList next(int uidx, IntList available, ValueFunction valFunc, int k)
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
                double val = valFunc.apply(uidx, i, this.betas[i].sample(), 0);
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
            for (int i = 0; i < numItems; ++i)
            {
                betas[i] = new BetaDistribution(initialAlpha, initialBeta);
            }
        }
        else
        {
            for (int i = 0; i < numItems; ++i)
            {
                betas[i] = new BetaDistribution(initialAlphas[i], initialBetas[i]);
            }
        }
    }
}
