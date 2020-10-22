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
import es.uam.eps.ir.knnbandit.data.datasets.GeneralOfflineDataset;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.Collections;
import java.util.Random;

/**
 * Target user / candidate item selection mechanism for non-sequential offline datasets.
 * Here, we allow repetition, and we select a fixed number of items each iteration with
 * (at least) one positive rating.
 *
 * This follows the evaluation protocol from the following reference:
 *
 * <p>
 *     <b>Reference: </b> C. Gentile, S. Li, G. Zapella. Online clustering of bandits. 29th conference on Neural Information Processing Systems (NeurIPS 2015). Montréal, Canada (2015).
 * </p>
 *
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class LimitedCandidatePoolSelection<U,I> implements Selection<U,I>
{
    /**
     * List of users we can recommend items to.
     */
    protected final IntList userList;

    /**
     * Random seed.
     */
    private final int rngSeed;

    /**
     * Random number generator.
     */
    private Random rng;

    /**
     * The current number of target users.
     */
    private int numUsers;

    /**
     * The number of candidate items to select.
     */
    private final int numCandidates;
    /**
     * The dataset.
     */
    private GeneralOfflineDataset<U,I> dataset;
    /**
     * The item list.
     */
    private final IntList allItems;

    /**
     * Constructor.
     * @param rngSeed random seed.
     */
    public LimitedCandidatePoolSelection(int rngSeed, int numCandidates)
    {
        this.rngSeed = rngSeed;
        this.rng = new Random(rngSeed);
        this.userList = new IntArrayList();
        this.numUsers = 0;
        this.numCandidates = numCandidates;
        this.allItems = new IntArrayList();
    }

    @Override
    public int selectTarget()
    {
        int index = rng.nextInt(numUsers);
        return this.userList.get(index);
    }

    @Override
    public IntList selectCandidates(int uidx)
    {
        IntList positiveItems = new IntArrayList();
        this.dataset.getUidxPreferences(uidx).filter(item -> this.dataset.isRelevant(item.v2)).forEach(item -> positiveItems.add(item.v1));
        if(positiveItems.isEmpty()) return null;

        IntList candidateItems = new IntArrayList();
        if(this.numCandidates >= this.dataset.numItems())
        {
            this.dataset.getAllIidx().forEach(candidateItems::add);
        }
        else
        {
            int iidx = positiveItems.getInt(rng.nextInt(positiveItems.size()));
            candidateItems.add(iidx);
            for(int i = 1; i < numCandidates; ++i)
            {
                int index = rng.nextInt(dataset.numItems());
                iidx = allItems.getInt(index);
                if(!candidateItems.contains(iidx)) candidateItems.add(iidx);
                else --i;
            }
        }
        return candidateItems;
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
    }

    @Override
    public void init(Dataset<U, I> dataset)
    {
        this.dataset = ((GeneralOfflineDataset<U,I>) dataset);
        this.userList.clear();
        this.allItems.clear();
        this.rng = new Random(rngSeed);

        this.dataset.getUidxWithPreferences().forEach(userList::add);
        this.dataset.getAllIidx().forEach(allItems::add);

        Collections.shuffle(this.userList, rng);
        this.numUsers = userList.size();
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
