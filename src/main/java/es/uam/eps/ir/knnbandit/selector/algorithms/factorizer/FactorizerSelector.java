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

import org.json.JSONObject;

import java.util.List;

/**
 * Class for selecting a factorizer for matrix factorization algorithms.
 *
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class FactorizerSelector<U,I>
{
    /**
     * The identifier for the algorithm name.
     */
    private final static String NAME = "name";
    /**
     * The identifier for the parameters of the algorithms.
     */
    private final static String PARAMS = "params";

    /**
     * Obtains the suppliers for factorizers.
     * @param json the JSON object.
     * @return the list of factorizer suppliers.
     */
    public List<FactorizerSupplier<U,I>> getFactorizers(JSONObject json)
    {
        String algorithm = json.getString(NAME);
        FactorizerConfigurator<U,I> conf = this.getConfigurator(algorithm);
        if(conf == null) return null;
        return conf.getFactorizers(json.getJSONArray(PARAMS));
    }

    /**
     * Obtains a single multi-armed bandit supplier from a JSON object.
     * @param json the JSON object containing the bandit configuration parameters.
     * @return the supplier for the algorithm under the corresponding configuration.
     */
    public FactorizerSupplier<U,I> getFactorizer(JSONObject json)
    {
        String algorithm = json.getString(NAME);
        FactorizerConfigurator<U,I> conf = this.getConfigurator(algorithm);
        if(conf == null) return null;
        return conf.getFactorizer(json.getJSONObject(PARAMS));
    }

    /**
     * Obtains the configurator of a factorizer.
     * @param name the name of the factorizer.
     * @return the configurator.
     */
    private  FactorizerConfigurator<U,I> getConfigurator(String name)
    {
        switch(name)
        {
            case FactorizerIdentifiers.IMF:
                return new HKVFactorizerConfigurator<>();
            case FactorizerIdentifiers.FASTIMF:
                return new PZTFactorizerConfigurator<>();
            case FactorizerIdentifiers.PLSA:
                return new PLSAFactorizerConfigurator<>();
            default:
                return null;
        }
    }
}