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
import it.unimi.dsi.fastutil.ints.*;

/**
 * Item bandit using the Thompson sampling algorithm, delaying the updates
 *
 * @param <U> User type.
 * @param <I> Item type.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class DelayedThompsonSamplingItemBandit<U, I> extends ItemBandit<U, I>
{
    /**
     * A Beta distribution for each possible item.
     */
    private final BetaDistribution[] betas;


    private final Int2IntMap delays;
    private final Int2DoubleMap currentScores;
    private final int delay;
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
     * @param delay    The time.
     */
    public DelayedThompsonSamplingItemBandit(int numItems, int delay)
    {
        this.numItems = numItems;
        this.betas = new BetaDistribution[numItems];
        this.delays = new Int2IntOpenHashMap();
        this.currentScores = new Int2DoubleOpenHashMap();
        this.delay = delay;
        for (int i = 0; i < numItems; ++i)
        {
            betas[i] = new BetaDistribution(1.0, 1.0);
            this.delays.put(i, delay);
            this.currentScores.put(i, betas[i].sample());
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
    public DelayedThompsonSamplingItemBandit(int numItems, double initialAlpha, double initialBeta, int delay)
    {
        this.numItems = numItems;
        this.betas = new BetaDistribution[numItems];
        this.delays = new Int2IntOpenHashMap();
        this.currentScores = new Int2DoubleOpenHashMap();
        this.delay = delay;
        for (int i = 0; i < numItems; ++i)
        {
            betas[i] = new BetaDistribution(initialAlpha, initialBeta);
            this.delays.put(i, delay);
            this.currentScores.put(i, betas[i].sample());
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
    public DelayedThompsonSamplingItemBandit(int numItems, double[] initialAlphas, double[] initialBetas, int delay)
    {
        this.numItems = numItems;
        this.betas = new BetaDistribution[numItems];
        this.delays = new Int2IntOpenHashMap();
        this.currentScores = new Int2DoubleOpenHashMap();
        this.delay = delay;
        for (int i = 0; i < numItems; ++i)
        {
            betas[i] = new BetaDistribution(initialAlphas[i], initialBetas[i]);
            this.delays.put(i, delay);
            this.currentScores.put(i, betas[i].sample());
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
                int currentDelay = this.delays.get(i);
                double val;
                if (currentDelay > 0)
                {
                    val = valF.apply(uidx, i, this.currentScores.get(i), 0);
                    this.delays.put(i, currentDelay - 1);
                }
                else
                {
                    double aux = this.betas[i].sample();
                    this.delays.put(i, delay);
                    this.currentScores.put(i, aux);
                    val = valF.apply(uidx, i, this.currentScores.get(i), 0);
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
                int currentDelay = this.delays.get(i);
                double val;
                if (currentDelay > 0)
                {
                    val = valF.apply(uidx, i, this.currentScores.get(i), 0);
                    this.delays.put(i, currentDelay - 1);
                }
                else
                {
                    double aux = this.betas[i].sample();
                    this.delays.put(i, delay);
                    this.currentScores.put(i, aux);
                    val = valF.apply(uidx, i, this.currentScores.get(i), 0);
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
    public void update(int i, double value)
    {
        this.betas[i].updateAdd(value, (1.0 - value));
        this.currentScores.put(i, this.betas[i].sample());
        this.delays.put(i, delay);
    }

    @Override
    public void reset()
    {
        if (initialAlphas == null)
        {
            for (int i = 0; i < numItems; ++i)
            {
                betas[i] = new BetaDistribution(initialAlpha, initialBeta);
                this.currentScores.put(i, betas[i].sample());
                this.delays.put(i, delay);
            }
        }
        else
        {
            for (int i = 0; i < numItems; ++i)
            {
                betas[i] = new BetaDistribution(initialAlphas[i], initialBetas[i]);
                this.currentScores.put(i, betas[i].sample());
                this.delays.put(i, delay);
            }
        }
    }
}
