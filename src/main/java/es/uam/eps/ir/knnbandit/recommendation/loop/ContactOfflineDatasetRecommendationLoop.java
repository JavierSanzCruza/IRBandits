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

import es.uam.eps.ir.knnbandit.data.datasets.ContactDataset;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.NonSequentialSelection;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.user.RandomUserSelector;
import es.uam.eps.ir.knnbandit.recommendation.loop.update.ContactUpdate;

import java.util.Map;

/**
 * An interactive recommendation loop for non sequential, contact recommendation in social networks datasets.
 * Each iteration, randomly selects a user and recommends an user that she has not rated (been recommended)
 * before.
 *
 * @param <U> type of the users
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ContactOfflineDatasetRecommendationLoop<U> extends GenericRecommendationLoop<U,U>
{
    /**
     * Constructor.
     *
     * @param dataset       the dataset containing all the information.
     * @param recommender   the interactive recommendation algorithm.
     * @param metrics       the set of metrics we want to study.
     * @param endCondition  the condition that establishes whether the loop has finished or not.
     * @param rngSeed       a random number generator seed.
     */
    public ContactOfflineDatasetRecommendationLoop(ContactDataset<U> dataset, InteractiveRecommenderSupplier<U, U> recommender, Map<String, CumulativeMetric<U, U>> metrics, EndCondition endCondition, int rngSeed)
    {
        super(dataset,new NonSequentialSelection<>(rngSeed, new RandomUserSelector(rngSeed), true), recommender, new ContactUpdate<>(), endCondition, metrics, rngSeed);
    }

    /**
     * Constructor.
     *
     * @param dataset       the dataset containing all the information.
     * @param recommender   the interactive recommendation algorithm.
     * @param metrics       the set of metrics we want to study.
     * @param endCondition  the condition that establishes whether the loop has finished or not.
     * @param rngSeed       a random number generator seed.
     * @param cutoff        the number of items to recommend each iteration.
     */
    public ContactOfflineDatasetRecommendationLoop(ContactDataset<U> dataset, InteractiveRecommenderSupplier<U, U> recommender, Map<String, CumulativeMetric<U, U>> metrics, EndCondition endCondition, int rngSeed, int cutoff)
    {
        super(dataset,new NonSequentialSelection<>(rngSeed, new RandomUserSelector(rngSeed), true), recommender, new ContactUpdate<>(), endCondition, metrics, rngSeed, cutoff);
    }
}
