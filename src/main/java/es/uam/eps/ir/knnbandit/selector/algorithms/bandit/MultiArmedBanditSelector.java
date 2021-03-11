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

import org.json.JSONObject;

import java.util.List;

/**
 * Class for selecting the multi-armed bandits in the experiments.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class MultiArmedBanditSelector
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
     * Obtains the suppliers for multi-armed bandits.
     * @param json the JSON object.
     * @return the list of multi-armed bandit suppliers.
     */
    public List<BanditSupplier> getBandits(JSONObject json)
    {
        String algorithm = json.getString(NAME);
        BanditConfigurator conf = this.getConfigurator(algorithm);
        if(conf == null) return null;
        return conf.getBandits(json.getJSONArray(PARAMS));
    }

    /**
     * Obtains a single multi-armed bandit supplier from a JSON object.
     * @param json the JSON object containing the bandit configuration parameters.
     * @return the supplier for the algorithm under the corresponding configuration.
     */
    public BanditSupplier getBandit(JSONObject json)
    {
        String algorithm = json.getString(NAME);
        BanditConfigurator conf = this.getConfigurator(algorithm);
        if(conf == null) return null;
        return conf.getBandit(json.getJSONObject(PARAMS));
    }

    /**
     * Obtains the configurator of a multi-armed bandit.
     * @param name the name of the multi-armed bandit.
     * @return the configurator.
     */
    private BanditConfigurator getConfigurator(String name)
    {
        switch(name)
        {
            case MultiArmedBanditIdentifiers.EGREEDY:
                return new EpsilonGreedyConfigurator();
            case MultiArmedBanditIdentifiers.ETGREEDY:
                return new EpsilonTGreedyConfigurator();
            case MultiArmedBanditIdentifiers.UCB1:
                return new UCB1Configurator();
            case MultiArmedBanditIdentifiers.UCB1TUNED:
                return new UCB1TunedConfigurator();
            case MultiArmedBanditIdentifiers.THOMPSON:
                return new ThompsonSamplingConfigurator();
            case MultiArmedBanditIdentifiers.DELAYTHOMPSON:
                return new DelayThompsonSamplingConfigurator();
            case MultiArmedBanditIdentifiers.MLEPOP:
                return new MLECategoricalItemBanditConfigurator();
            case MultiArmedBanditIdentifiers.MLEAVG:
                return new MLECategoricalAverageItemBanditConfigurator();
            default:
                return null;
        }
    }


}