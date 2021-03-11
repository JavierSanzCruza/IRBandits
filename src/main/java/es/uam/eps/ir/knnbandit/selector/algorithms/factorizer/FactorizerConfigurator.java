/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms.factorizer;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;

/**¡
 * Interface for configuring factorizers for matrix factorization approaches.
 * I
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface FactorizerConfigurator<U,I>
{
    /**
     * Given a JSON array, obtains a list of suppliers.
     * @param array the JSON array.
     * @return the list of factorizer suppliers.
     */
    List<FactorizerSupplier<U,I>> getFactorizers(JSONArray array);
    /**
     * Given a JSON objects, obtains a factorizer supplier
     * @param object the JSON object.
     * @return the supplier of the factorizer.
     */
    FactorizerSupplier<U,I> getFactorizer(JSONObject object);
}
