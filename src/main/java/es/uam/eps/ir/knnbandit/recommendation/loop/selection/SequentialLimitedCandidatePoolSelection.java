/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.selection;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.StreamDataset;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.io.IOException;
import java.util.Random;

/**
 * Target user / candidate item selection mechanism for sequential offline datasets,
 * i.e. for the cases where the order of the ratings in the dataset is important
 * (the next user in the log is selected). It only takes as candidate items the
 * featured item in the log, and a random selection of other items.
 *
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class SequentialLimitedCandidatePoolSelection<U,I> implements Selection<U,I>
{
    /**
     * An stream dataset for reading everything.
     */
    protected StreamDataset<U,I> dataset;

    /**
     * The number of extra items to add to the candidate list.
     */
    private final int numExtra;
    /**
     * The number of random number generator seeds.
     */
    private final int rngSeed;

    /**
     * Random number generator.
     */
    private Random rng;

    /**
     * List of all items.
     */
    IntList allItems = new IntArrayList();

    /**
     * Constructor.
     */
    public SequentialLimitedCandidatePoolSelection(int numExtra, int rngSeed)
    {
        this.numExtra = numExtra;
        this.rngSeed = rngSeed;
        this.dataset = null;
        this.rng = new Random(rngSeed);
    }

    @Override
    public int selectTarget()
    {
        return this.dataset.getCurrentUidx();
    }

    @Override
    public IntList selectCandidates(int uidx)
    {
        if(this.dataset.getFeaturedIidx() > 0)
        {
            if (this.numExtra >= this.dataset.numItems() - 1)
            {
                return new IntArrayList(allItems);
            }
            else
            {
                IntList candidateList = new IntArrayList();
                candidateList.add(this.dataset.getFeaturedIidx());

                for (int j = 0; j < numExtra; ++j)
                {
                    int next = rng.nextInt(this.dataset.numItems());
                    int iidx = this.allItems.get(next);
                    if (!candidateList.contains(iidx))
                    {
                        candidateList.add(iidx);
                    }
                    else
                    {
                        --j;
                    }
                }

                return candidateList;
            }
        }

        return null;
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        try
        {
            this.dataset.advance();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    @Override
    public void init(Dataset<U, I> dataset)
    {
        try
        {
            this.dataset = (StreamDataset<U, I>) dataset;
            this.dataset.restart();
            this.rng = new Random();
            this.allItems = new IntArrayList();
            dataset.getAllIidx().forEach(allItems::add);
        }
        catch(IOException ignored)
        {
        }
    }

    @Override
    public void init(Dataset<U, I> dataset, Warmup warmup)
    {
        this.init(dataset);
    }

    @Override
    public boolean isAvailable(int uidx, int iidx)
    {
        return true;
    }
}
