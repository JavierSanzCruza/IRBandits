/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.warmup;

import es.uam.eps.ir.knnbandit.data.datasets.StreamDataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Warm-up for the streaming data.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class StreamWarmup implements Warmup
{
    /**
     * The full set of ratings.
     */
    private final List<FastRating> fullTraining;
    /**
     * The number of relevant ratings.
     */
    private final int numRel;

    /**
     * Constructor.
     * @param fullTraining the full set of tuples.
     * @param numRel the number of relevant ratings.
     */
    protected StreamWarmup(List<FastRating> fullTraining, int numRel)
    {
        this.fullTraining = fullTraining;
        this.numRel = numRel;
    }

    @Override
    public int getNumRel()
    {
        return numRel;
    }

    @Override
    public List<FastRating> getFullTraining()
    {
        return fullTraining;
    }

    @Override
    public List<FastRating> getCleanTraining()
    {
        return fullTraining;
    }

    /**
     * Loads the stream warmup.
     * @param dataset   the streaming dataset.
     * @param training  the list of pairs.
     * @return the warmup for the streaming dataset.
     * @throws IOException if something fails while reading the data.
     */
    public StreamWarmup load(StreamDataset<?,?> dataset, List<Pair<Integer>> training) throws IOException
    {
        int numRel = 0;
        List<FastRating> fullTraining = new ArrayList<>();
        dataset.restart();
        for(Pair<Integer> t : training)
        {
            dataset.advance();
            int uidx = t.v1();
            int iidx = t.v2();

            if(dataset.getCurrentUidx() == uidx && dataset.getFeaturedIidx() == iidx)
            {
                double value = dataset.getFeaturedItemRating();
                if(dataset.getRelevanceChecker().test(value)) numRel++;
                fullTraining.add(new FastRating(uidx, iidx, value));
            }
        }

        return new StreamWarmup(fullTraining, numRel);
    }

}
