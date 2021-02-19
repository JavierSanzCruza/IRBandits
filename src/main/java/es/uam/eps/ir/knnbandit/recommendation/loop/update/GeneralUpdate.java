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
import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.Selection;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.Warmup;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * General recommendation update. It assumes that, if we do receive the (uidx, iidx) value,
 * we take its payoff, and use it for updating both recommenders and metrics.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 */
public class GeneralUpdate<U,I> implements UpdateStrategy<U,I>
{
    /**
     * The offline dataset.
     */
    private OfflineDataset<U,I> dataset;

    @Override
    public void init(Dataset<U, I> dataset)
    {
        this.dataset = ((OfflineDataset<U,I>) dataset);
    }

    @Override
    public Pair<List<FastRating>> selectUpdate(int uidx, int iidx, Selection<U,I> selection)
    {
        if(selection.isAvailable(uidx, iidx))
        {
            Optional<Double> value = dataset.getPreference(uidx, iidx);
            List<FastRating> list = new ArrayList<>();
            list.add(new FastRating(uidx, iidx, value.orElse(Double.NaN)));
            return new Pair<>(list, list);
        }
        return new Pair<>(new ArrayList<>(), new ArrayList<>());
    }

    @Override
    public List<FastRating> getList(Warmup warmup)
    {
        return warmup.getFullTraining();
    }
}
