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
import es.uam.eps.ir.knnbandit.data.datasets.DatasetWithKnowledge;
import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.Selection;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.Warmup;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Class that allows the selection of a subset of ratings depending
 * on whether the users knew the items or not before the recommendation
 * was done.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class WithKnowledgeUpdate<U,I> implements UpdateStrategy<U,I>
{
    /**
     * A dataset containing the information we are considering.
     */
    private OfflineDataset<U,I> dataset;
    /**
     * The selection of ratings we are going to use.
     */
    private final KnowledgeDataUse dataUse;

    /**
     * Constructor.
     * @param dataUse a selection of the subset of ratings we are going to use.
     */
    public WithKnowledgeUpdate(KnowledgeDataUse dataUse)
    {
        this.dataUse = dataUse;
    }

    @Override
    public void init(Dataset<U, I> dataset)
    {
        this.dataset = ((DatasetWithKnowledge<U,I>) dataset).getDataset(dataUse);
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
