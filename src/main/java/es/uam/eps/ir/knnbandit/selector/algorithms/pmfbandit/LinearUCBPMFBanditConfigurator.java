/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms.pmfbandit;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.mf.icf.LinearUCBInteractivePMFRecommender;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.PMFBanditIdentifiers;
import es.uam.eps.ir.knnbandit.selector.algorithms.AbstractAlgorithmConfigurator;
import org.json.JSONObject;

/**
 * Configures the linear UCB variant of the PMF bandit.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.mf.icf.LinearUCBInteractivePMFRecommender
 */
public class LinearUCBPMFBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier of the parameter that regularizes the upper confidence bound.
     */
    private final static String ALPHA = "alpha";
    /**
     * The number of latent factors.
     */
    private final int k;
    /**
     * The prior standard deviation of the user factors.
     */
    private final double lambdaP;
    /**
     * The prior standard deviation of the item factors.
     */
    private final double lambdaQ;
    /**
     * The standard deviation of the ratings.
     */
    private final double stdev;
    /**
     * The number of iterations.
     */
    private final int numIter;
    /**
     * True if we ignore the ratings we do not know about, false otherwise.
     */
    private final boolean ignoreUnknown;

    /**
     * Constructor.
     * @param k             the number of latent factors.
     * @param lambdaP       the prior standard deviation of the user factors.
     * @param lambdaQ       the prior standard deviation of the item factors.
     * @param stdev         the standard deviation of the ratings.
     * @param numIter       the number of iterations.
     * @param ignoreUnknown true if we ignore the ratings that we do not know about, false otherwise.
     */
    public LinearUCBPMFBanditConfigurator(int k, double lambdaP, double lambdaQ, double stdev, int numIter, boolean ignoreUnknown)
    {
        this.k = k;
        this.lambdaP = lambdaP;
        this.lambdaQ = lambdaQ;
        this.stdev = stdev;
        this.numIter = numIter;
        this.ignoreUnknown = ignoreUnknown;
    }

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        return new GeneralizedUCBPMFBanditInteractiveRecommenderSupplier(alpha);
    }

    /**
     * Configures the linear UCB variant of the PMF bandit.
     */
    private class GeneralizedUCBPMFBanditInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        /**
         * The parameter regularizing the upper confidence bound.
         */
        private final double alpha;

        /**
         * Constructor.
         * @param alpha regularizes the upper confidence bound.
         */
        public GeneralizedUCBPMFBanditInteractiveRecommenderSupplier(double alpha)
        {
            this.alpha = alpha;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new LinearUCBInteractivePMFRecommender<>(userIndex, itemIndex, ignoreUnknown, k, lambdaP, lambdaQ, stdev, numIter, alpha);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new LinearUCBInteractivePMFRecommender<>(userIndex, itemIndex, ignoreUnknown, rngSeed, k, lambdaP, lambdaQ, stdev, numIter, alpha);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.PMFBANDIT + "-" + k + "-" + lambdaP + "-" + lambdaQ + "-" + stdev + "-" + numIter + "-" + PMFBanditIdentifiers.UCB + "-" + alpha + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
