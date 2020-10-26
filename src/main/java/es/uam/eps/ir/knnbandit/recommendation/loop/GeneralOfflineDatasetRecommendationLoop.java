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

import es.uam.eps.ir.knnbandit.data.datasets.GeneralDataset;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.NonSequentialSelection;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.user.RandomUserSelector;
import es.uam.eps.ir.knnbandit.recommendation.loop.update.GeneralUpdate;

import java.util.Map;

/**
 * An interactive recommendation loop for non sequential, general domain datasets.
 * Each iteration, randomly selects a user and recommends an item that she has not rated (been recommended)
 * before.
 *
 * @param <U> type of the users
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class GeneralOfflineDatasetRecommendationLoop<U,I> extends GenericRecommendationLoop<U,I>
{
    /**
     * Constructor.
     *
     * @param dataset      the dataset containing all the information.
     * @param recommender  the interactive recommendation algorithm.
     * @param metrics      the set of metrics we want to study.
     * @param endCondition the condition that establishes whether the loop has finished or not.
     * @param rngSeed      a random number generator seed.
     */
    public GeneralOfflineDatasetRecommendationLoop(GeneralDataset<U, I> dataset, InteractiveRecommenderSupplier<U, I> recommender, Map<String, CumulativeMetric<U, I>> metrics, EndCondition endCondition, int rngSeed)
    {
        super(dataset, new NonSequentialSelection<>(rngSeed, new RandomUserSelector(rngSeed)), recommender, new GeneralUpdate<>(), endCondition, metrics);
    }
}
