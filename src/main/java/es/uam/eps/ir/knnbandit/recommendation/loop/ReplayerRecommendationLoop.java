/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.loop;

import es.uam.eps.ir.knnbandit.data.datasets.StreamDataset;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.SequentialSelection;
import es.uam.eps.ir.knnbandit.recommendation.loop.update.ReplayerUpdate;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;

import java.util.List;
import java.util.Map;

/**
 * An interactive recommendation loop following the replayer strategy.
 *
 * <p>
 *     <b>Reference: </b> L. Li, W. Chu, J. Langford, X. Wang. Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms. 4th ACM Conference on Web search and data mining (WSDM 2011). Hong Kong, China, pp. 297-306 (2011).
 * </p>
 *
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ReplayerRecommendationLoop<U,I> extends GenericRecommendationLoop<U,I>
{
    /**
     * Constructor.
     *
     * @param dataset       the dataset containing all the information.
     * @param recommender   the interactive recommendation algorithm.
     * @param metrics       the set of metrics we want to study.
     * @param endCondition  the condition that establishes whether the loop has finished or not.
     */
    public ReplayerRecommendationLoop(StreamDataset<U,I> dataset, InteractiveRecommenderSupplier<U, I> recommender, Map<String, CumulativeMetric<U, I>> metrics, EndCondition endCondition)
    {
        super(dataset, new SequentialSelection<>(), recommender, new ReplayerUpdate<>(), endCondition, metrics);
    }

    @Override
    public void fastUpdate(int uidx, int iidx)
    {
        Pair<List<FastRating>> updateValues = this.update.selectUpdate(uidx, iidx, this.selection);
        // First, update the recommender:
        List<FastRating> recValues = updateValues.v1();

        // advance the loop
        selection.update(uidx, iidx, 0.0);

        for(FastRating value : recValues)
        {
            recommender.update(value.uidx(), value.iidx(),value.value());
        }

        // Then, update the metrics:
        List<FastRating> metricValues = updateValues.v2();
        for(FastRating value : metricValues)
        {
            metrics.forEach((name, metric) -> metric.update(value.uidx(), value.iidx(),value.value()));
            endCond.update(value.uidx(), value.iidx(),value.value());
        }
        numIter++;
    }
}
