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
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.AverageRatingMLE;
import org.json.JSONObject;


/**
 * Class for configuring a bandit that selects an item proportionally to its average success.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.AverageRatingMLE
 */
public class MLECategoricalAverageItemBanditConfigurator extends AbstractBanditConfigurator
{
    /**
     * Identifier of the initial value of the alpha parameter of the Beta distribution.
     */
    private final static String ALPHA = "alpha";
    /**
     * Identifier of the initial value of the beta parameter of the Beta distribution.
     */
    private final static String BETA = "beta";

    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);
        return new MLECategoricalItemBanditSupplier(alpha, beta);
    }

    /**
     * Class for configuring a bandit that selects an item proportionally to its average rating.
     */
    private static class MLECategoricalItemBanditSupplier implements BanditSupplier
    {
        /**
         * The initial value for the alpha parameter of Beta distributions.
         */
        private final double alpha;
        /**
         * The initial value for the beta parameter of Beta distributions.
         */
        private final double beta;

        /**
         * @param alpha the initial value for the alpha parameter of Beta distributions.
         * @param beta  the initial value for the beta parameter of Beta distributions.
         */
        public MLECategoricalItemBanditSupplier(double alpha, double beta)
        {
            this.alpha = alpha;
            this.beta = beta;
        }

        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new AverageRatingMLE(numItems, alpha, beta);
        }

        @Override
        public String getName()
        {
            return MultiArmedBanditIdentifiers.MLEAVG + "-" + alpha + "-" + beta;
        }
    }
}
