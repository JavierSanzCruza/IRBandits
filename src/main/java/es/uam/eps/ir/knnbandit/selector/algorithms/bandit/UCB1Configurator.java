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

import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.*;
import org.json.JSONObject;

/**
 * Class for configuring the UCB1 multi-armed bandit algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.UCB1
 */
public class UCB1Configurator extends AbstractBanditConfigurator
{
    /**
     * Identifier for the parameter regulating the upper confidence bound.
     */
    private final static String ALPHA = "alpha";

    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        double alpha = object.has(ALPHA) ? object.getDouble(ALPHA) : 2.0;
        return new UCB1BanditSupplier(alpha);
    }

    /**
     * Class for configuring the UCB1 multi-armed bandit.
     */
    private static class UCB1BanditSupplier implements BanditSupplier
    {
        /**
         * The parameter regulating the upper confidence bound.
         */
        private final double alpha;

        /**
         * Constructor.
         * @param alpha the parameter regulating the upper confidence bound.
         */
        public UCB1BanditSupplier(double alpha)
        {
            this.alpha = alpha;
        }
        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new UCB1(numItems, alpha);
        }

        @Override
        public String getName()
        {
            return MultiArmedBanditIdentifiers.UCB1;
        }
    }
}
