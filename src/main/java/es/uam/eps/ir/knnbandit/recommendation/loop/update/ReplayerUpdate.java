/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.update;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.StreamDataset;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.Selection;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.Warmup;

import java.util.ArrayList;
import java.util.List;

/**
 * Update mechanism for the Replayer evaluation strategy: given a stream dataset,
 * it only updates the values if the currently selected target user and candidate item
 * correspond with the current user and featured items in such stream.
 *
 * In other case, (uidx, iidx) pair is not used for updating neither the recommendation
 * nor the metrics.
 *
 * @param <U> the type of the users.
 * @param <I> the type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ReplayerUpdate<U,I> implements UpdateStrategy<U,I>
{
    /**
     * The stream dataset.
     */
    private StreamDataset<U,I> dataset;

    @Override
    public void init(Dataset<U, I> dataset)
    {
        this.dataset = ((StreamDataset<U,I>) dataset);
    }

    @Override
    public Pair<List<FastRating>> selectUpdate(int uidx, int iidx, Selection<U,I> selection)
    {
        List<FastRating> list = new ArrayList<>();
        List<FastRating> metricList = new ArrayList<>();

        if(dataset.getCurrentUidx() == uidx && dataset.getFeaturedIidx() == iidx)
        {
            FastRating rating = new FastRating(uidx, iidx, dataset.getFeaturedItemRating());
            list.add(rating);
            metricList.add(rating);
        }

        return new Pair<>(list, metricList);
    }

    @Override
    public List<FastRating> getList(Warmup warmup)
    {
        return warmup.getFullTraining();
    }
}
