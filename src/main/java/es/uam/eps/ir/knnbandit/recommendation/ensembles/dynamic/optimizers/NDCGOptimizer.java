/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.ensembles.dynamic.optimizers;

import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.basic.NDCG;

/**
 * In a dynamic ensemble, this method computes the nDCG@k metric to select the next recommender to use.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 */
public class NDCGOptimizer<U,I> implements DynamicOptimizer<U,I>
{
    /**
     * The NDCG metric to use.
     */
    private NDCG<U,I> ndcg;

    @Override
    public void init(OfflineDataset<U, I> dataset, int cutoff)
    {
        NDCG.NDCGRelevanceModel<U,I> relModel = new NDCG.NDCGRelevanceModel<>(false,dataset.getPreferenceData(), 1.0);
        // Choose the recommender maximizing the metric.
        this.ndcg = new NDCG<>(cutoff, relModel);
    }

    @Override
    public double evaluate(Recommendation<U, I> rec)
    {
        return ndcg.evaluate(rec);
    }
}