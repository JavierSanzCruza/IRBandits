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

import es.uam.eps.ir.knnbandit.data.datasets.DatasetWithKnowledge;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.NonSequentialSelection;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.user.RandomUserSelector;
import es.uam.eps.ir.knnbandit.recommendation.loop.update.WithKnowledgeUpdate;

import java.util.Map;

/**
 * An interactive recommendation loop for non sequential, general domain datasets with information
 * about whether the user knew the items before he provided them a rating.
 * Each iteration, randomly selects a user and recommends an item that she has not rated (been recommended)
 * before.
 *
 * @param <U> type of the users
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class OfflineDatasetWithKnowledgeRecommendationLoop<U,I> extends GenericRecommendationLoop<U,I>
{
    /**
     * Constructor.
     *
     * @param dataset      the dataset containing all the information.
     * @param recommender  the interactive recommendation algorithm.
     * @param metrics      the set of metrics we want to study.
     * @param endCondition the condition that establishes whether the loop has finished or not.
     * @param dataUse      a selection of the subset of the ratings we shall use for updates.
     * @param rngSeed      a seed for a random number generator
     */
    public OfflineDatasetWithKnowledgeRecommendationLoop(DatasetWithKnowledge<U, I> dataset, InteractiveRecommenderSupplier<U, I> recommender, Map<String, CumulativeMetric<U, I>> metrics, EndCondition endCondition, KnowledgeDataUse dataUse, int rngSeed)
    {
        super(dataset, new NonSequentialSelection<>(rngSeed, new RandomUserSelector(rngSeed), false), recommender, new WithKnowledgeUpdate<>(dataUse), endCondition, metrics, rngSeed);
    }

    /**
     * Constructor.
     *
     * @param dataset      the dataset containing all the information.
     * @param recommender  the interactive recommendation algorithm.
     * @param metrics      the set of metrics we want to study.
     * @param endCondition the condition that establishes whether the loop has finished or not.
     * @param dataUse      a selection of the subset of the ratings we shall use for updates.
     * @param rngSeed      a seed for a random number generator
     * @param cutoff       the number of items to recommend each iteration.
     */
    public OfflineDatasetWithKnowledgeRecommendationLoop(DatasetWithKnowledge<U, I> dataset, InteractiveRecommenderSupplier<U, I> recommender, Map<String, CumulativeMetric<U, I>> metrics, EndCondition endCondition, KnowledgeDataUse dataUse, int rngSeed, int cutoff)
    {
        super(dataset, new NonSequentialSelection<>(rngSeed, new RandomUserSelector(rngSeed), false), recommender, new WithKnowledgeUpdate<>(dataUse), endCondition, metrics, rngSeed, cutoff);
    }
}
