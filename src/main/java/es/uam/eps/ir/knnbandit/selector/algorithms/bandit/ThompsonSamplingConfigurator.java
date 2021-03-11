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

import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.AbstractMultiArmedBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.ThompsonSampling;
import org.json.JSONObject;

/**
 * Class for configuring the Thompson sampling multi-armed bandit algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.ThompsonSampling
 */
public class ThompsonSamplingConfigurator extends AbstractBanditConfigurator
{
    /**
     * Identifier for the initial alpha value for the Beta distributions.
     */
    private final static String ALPHA = "alpha";
    /**
     * Identifier for the initial beta value for the Beta distributions.
     */
    private final static String BETA = "beta";

    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);
        return new ThompsonSamplingBanditSupplier(alpha, beta);
    }

    /**
     * Class for configuring Thompson sampling multi-armed bandits.
     */
    private static class ThompsonSamplingBanditSupplier implements BanditSupplier
    {
        /**
         * Initial alpha value for the Beta distribution.
         */
        private final double alpha;
        /**
         * Initial beta value for the Beta distribution.
         */
        private final double beta;

        /**
         * Constructor.
         * @param alpha the initial alpha value for the Beta distributions.
         * @param beta  the initial beta value for the Beta distributions.
         */
        public ThompsonSamplingBanditSupplier(double alpha, double beta)
        {
            this.alpha = alpha;
            this.beta = beta;
        }

        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new ThompsonSampling(numItems, alpha, beta);
        }

        @Override
        public String getName()
        {
            return MultiArmedBanditIdentifiers.THOMPSON + "-" + alpha + "-" + beta;
        }
    }
}
