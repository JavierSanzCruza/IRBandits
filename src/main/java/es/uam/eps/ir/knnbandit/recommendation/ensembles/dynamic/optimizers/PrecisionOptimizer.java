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
import es.uam.eps.ir.ranksys.metrics.basic.Precision;
import es.uam.eps.ir.ranksys.metrics.rel.IdealRelevanceModel;
import java.util.Set;
import java.util.function.DoublePredicate;
import java.util.stream.Collectors;

/**
 * In a dynamic ensemble, this method computes the P@k metric to select the next recommender to use.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 */
public class PrecisionOptimizer<U,I> implements DynamicOptimizer<U,I>
{
    /**
     * The precision metric.
     */
    private Precision<U,I> precision;

    @Override
    public void init(OfflineDataset<U, I> dataset, int cutoff)
    {
        IdealRelevanceModel<U,I> relModel = new IdealRelevanceModel<U, I>()
        {
            @Override
            protected UserIdealRelevanceModel<U, I> get(U u)
            {
                DoublePredicate pred = dataset.getRelevanceChecker();
                return new UserIdealRelevanceModel<U, I>()
                {
                    @Override
                    public Set<I> getRelevantItems()
                    {
                        return dataset.getUserPreferences(u).filter(i -> pred.test(i.v2)).map(i -> i.v1).collect(Collectors.toSet());
                    }

                    @Override
                    public boolean isRelevant(I i)
                    {
                        return dataset.isRelevant(dataset.getPreference(u,i).orElse(0.0));
                    }

                    @Override
                    public double gain(I i)
                    {
                        return dataset.getPreference(u,i).orElse(0.0);
                    }
                };
            }
        };

        // Choose the recommender maximizing the metric.
        this.precision = new Precision<>(cutoff, relModel);
    }

    @Override
    public double evaluate(Recommendation<U, I> rec)
    {
        return precision.evaluate(rec);
    }
}