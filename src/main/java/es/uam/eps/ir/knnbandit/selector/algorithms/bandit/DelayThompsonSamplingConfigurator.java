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
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.DelayedThompsonSampling;
import org.json.JSONObject;

/**
 * Class for configuring delayed versions of the Thompson sampling multi-armed bandit algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.DelayedThompsonSampling
 */
public class DelayThompsonSamplingConfigurator extends AbstractBanditConfigurator
{
    /**
     * Identifier for the initial alpha value of the Beta distributions.
     */
    private final static String ALPHA = "alpha";
    /**
     * Identifier for the initial beta value of the Beta distributions.
     */
    private final static String BETA = "beta";
    /**
     * Identifier for the number of steps before updating an estimation.
     */
    private final static String DELAY = "delay";

    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);
        int delay = object.getInt(DELAY);
        return new DelayThompsonSamplingBanditSupplier(alpha, beta, delay);
    }

    /**
     * Class for configuring a delayed Thompson sampling multi-armed bandit.
     */
    private static class DelayThompsonSamplingBanditSupplier implements BanditSupplier
    {
        /**
         * The initial alpha value of the Beta distributions.
         */
        private final double alpha;
        /**
         * The initial beta value of the Beta distributions.
         */
        private final double beta;
        /**
         * The number of steps before updating an estimation.
         */
        private final int delay;

        /**
         * Constructor.
         * @param alpha the initial alpha value of the Beta distributions.
         * @param beta  the initial beta value of the Beta distributions.
         * @param delay the number of steps before updating an estimation.
         */
        public DelayThompsonSamplingBanditSupplier(double alpha, double beta, int delay)
        {
            this.alpha = alpha;
            this.beta = beta;
            this.delay = delay;
        }

        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new DelayedThompsonSampling(numItems, alpha, beta, delay);
        }

        @Override
        public String getName()
        {
            return MultiArmedBanditIdentifiers.DELAYTHOMPSON + "-" + alpha + "-" + beta + "-" + delay;
        }
    }
}
