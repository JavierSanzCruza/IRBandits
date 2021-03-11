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
import es.uam.eps.ir.knnbandit.selector.PMFBanditIdentifiers;
import es.uam.eps.ir.knnbandit.selector.algorithms.pmfbandit.EpsilonGreedyPMFBanditConfigurator;
import es.uam.eps.ir.knnbandit.selector.algorithms.pmfbandit.GeneralizedLinearUCBPMFBanditConfigurator;
import es.uam.eps.ir.knnbandit.selector.algorithms.pmfbandit.LinearUCBPMFBanditConfigurator;
import es.uam.eps.ir.knnbandit.selector.algorithms.pmfbandit.ThompsonSamplingPMFBanditConfigurator;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * Configurator for the interactive contact recommendation algorithm based on the combination of probabilistic
 * matrix factorization with multi-armed bandit algorithms for selecting items.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.mf.icf.InteractivePMFRecommender
 */
public class PMFBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for the prior standard deviation for the user factors.
     */
    private static final String LAMBDAP = "lambdaP";
    /**
     * Identifier for the prior standard deviation for the item factors.
     */
    private static final String LAMBDAQ = "lambdaQ";
    /**
     * Standard rating deviation.
     */
    private static final String STDEV = "stdev";
    /**
     * Number of iterations for training the latent factors.
     */
    private static final String NUMITER = "numIter";
    /**
     * Identifier for selecting whether the algorithm is updated with items unknown by the system or not.
     */
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    /**
     * Identifier for the variant to apply.
     */
    private static final String VARIANT = "variant";
    /**
     * Identifier for the number of latent factors.
     */
    private static final String K = "k";
    /**
     * Identifier for the name of the variant.
     */
    private static final String NAME = "name";
    /**
     * Identifier for the parameters of the variant.
     */
    private static final String PARAMS = "params";

    @Override
    public List<InteractiveRecommenderSupplier<U, I>> getAlgorithms(JSONArray array)
    {
        List<InteractiveRecommenderSupplier<U,I>> list = new ArrayList<>();
        int numConfigs = array.length();
        for(int i = 0; i < numConfigs; ++i)
        {
            JSONObject object = array.getJSONObject(i);
            boolean ignoreUnknown = true;
            if(object.has(IGNOREUNKNOWN))
            {
                ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
            }
            int k = object.getInt(K);
            double lambdaP = object.getDouble(LAMBDAP);
            double lambdaQ = object.getDouble(LAMBDAQ);
            double stdev = object.getDouble(STDEV);
            int numIter = object.getInt(NUMITER);

            // And, now, we obtain
            JSONObject variant = object.getJSONObject(VARIANT);
            String name = variant.getString(NAME);

            AlgorithmConfigurator<U,I> conf = this.selectInterPMFVariant(name, k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
            list.addAll(conf.getAlgorithms(variant.getJSONArray(PARAMS)));
        }
        return list;
    }

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }
        int k = object.getInt(K);
        double lambdaP = object.getDouble(LAMBDAP);
        double lambdaQ = object.getDouble(LAMBDAQ);
        double stdev = object.getDouble(STDEV);
        int numIter = object.getInt(NUMITER);

        // And, now, we obtain
        JSONObject variant = object.getJSONObject(VARIANT);
        String name = variant.getString(NAME);

        AlgorithmConfigurator<U,I> conf = this.selectInterPMFVariant(name, k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
        assert conf != null;
        return conf.getAlgorithm(variant.getJSONObject(PARAMS));
    }

    /**
     * Determines the variant of the algorithm that we need to use.
     * @param name          the name of the variant.
     * @param k             the number of latent factors.
     * @param lambdaP       the prior standard deviation of the user factors.
     * @param lambdaQ       the prior standard deviation of the item factors.
     * @param stdev         the standard deviation of the ratings.
     * @param numIter       the number of iterations.
     * @param ignoreUnknown true if we ignore the ratings that we do not know about, false otherwise.
     * @return the configurator for the selected PMF bandit variant.
     */
    private AlgorithmConfigurator<U,I> selectInterPMFVariant(String name, int k, double lambdaP, double lambdaQ, double stdev, int numIter, boolean ignoreUnknown)
    {
        switch(name)
        {
            case PMFBanditIdentifiers.EGREEDY:
                return new EpsilonGreedyPMFBanditConfigurator<>(k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
            case PMFBanditIdentifiers.UCB:
                return new LinearUCBPMFBanditConfigurator<>(k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
            case PMFBanditIdentifiers.GENERALIZEDUCB:
                return new GeneralizedLinearUCBPMFBanditConfigurator<>(k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
            case PMFBanditIdentifiers.THOMPSON:
                return new ThompsonSamplingPMFBanditConfigurator<>(k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
            default:
                return null;
        }
    }
}
