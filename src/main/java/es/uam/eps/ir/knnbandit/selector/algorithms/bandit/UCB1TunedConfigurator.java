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
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.UCB1Tuned;
import org.json.JSONObject;

/**
 * Class for configuring the UCB1-tuned multi-armed bandit algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.UCB1Tuned
 */
public class UCB1TunedConfigurator extends AbstractBanditConfigurator
{
    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        return new UCB1TunedBanditSupplier();
    }

    /**
     * Class for configuring the UCB1 tuned multi-armed bandit algorithm.
     */
    private static class UCB1TunedBanditSupplier implements BanditSupplier
    {
        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new UCB1Tuned(numItems);
        }

        @Override
        public String getName()
        {
            return MultiArmedBanditIdentifiers.UCB1TUNED;
        }
    }
}
