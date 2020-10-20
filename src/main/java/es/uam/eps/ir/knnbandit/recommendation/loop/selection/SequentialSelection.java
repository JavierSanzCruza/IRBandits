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
import it.unimi.dsi.fastutil.ints.*;

import java.io.IOException;

/**
 * Target user / candidate item selection mechanism for non-sequential offline datasets,
 * i.e. for the cases where the order of the ratings in the dataset is not important.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class SequentialSelection<U,I> implements Selection<U,I>
{
    protected StreamDataset<U,I> dataset;

    /**
     * Constructor.
     */
    public SequentialSelection()
    {
        this.dataset = null;
    }

    @Override
    public int selectTarget()
    {
        return this.dataset.getCurrentUidx();
    }

    @Override
    public IntList selectCandidates(int uidx)
    {
        return this.dataset.getCandidateIidx();
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
        }
        catch(IOException ioe)
        {
            return;
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
