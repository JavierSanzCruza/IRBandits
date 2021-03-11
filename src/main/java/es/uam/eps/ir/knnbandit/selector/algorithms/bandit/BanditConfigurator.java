/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;

/**
 * Interface for configuring bandit algorithms from JSON objects.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface BanditConfigurator
{
    /**
     * Given a JSON array, obtains a list of multi-armed bandit suppliers.
     * @param array the JSON array.
     * @return the list of multi-armed bandit suppliers.
     */
    List<BanditSupplier> getBandits(JSONArray array);
    /**
     * Given a JSON object, obtains a multi-armed bandit supplier.
     * @param object the JSON object.
     * @return the multi-armed bandit supplier.
     */
    BanditSupplier getBandit(JSONObject object);
}
