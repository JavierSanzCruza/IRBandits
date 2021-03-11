/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;

/**
 * Interface for configuring interactive recommendation algorithms from JSON objects.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface AlgorithmConfigurator<U,I>
{
    /**
     * Given a JSON array, obtains a list of algorithms.
     * @param array the array containing the algorithm configurators.
     * @return a list of the configured interactive recommender suppliers.
     */
    List<InteractiveRecommenderSupplier<U,I>> getAlgorithms(JSONArray array);

    /**
     * Given a JSON object, obtains a single algorithm.
     * @param object the object containing the algorithm configuration.
     * @return the configured interactive recommender supplier.
     */
    InteractiveRecommenderSupplier<U,I> getAlgorithm(JSONObject object);
}
