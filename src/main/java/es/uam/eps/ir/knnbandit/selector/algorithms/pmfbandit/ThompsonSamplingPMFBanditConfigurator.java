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
import es.uam.eps.ir.knnbandit.recommendation.mf.icf.ThompsonSamplingInteractivePMFRecommender;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.PMFBanditIdentifiers;
import es.uam.eps.ir.knnbandit.selector.algorithms.AbstractAlgorithmConfigurator;
import org.json.JSONObject;

/**
 * Configures the Thompson sampling variant of the PMF bandit.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.mf.icf.LinearUCBInteractivePMFRecommender
 */
public class ThompsonSamplingPMFBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{

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
    public ThompsonSamplingPMFBanditConfigurator(int k, double lambdaP, double lambdaQ, double stdev, int numIter, boolean ignoreUnknown)
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
        return new ThompsonSamplingPMFBanditInteractiveRecommenderSupplier();
    }

    /**
     * Configures the Thompson sampling variant of the PMF bandit.
     */
    private class ThompsonSamplingPMFBanditInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new ThompsonSamplingInteractivePMFRecommender<>(userIndex, itemIndex, ignoreUnknown, k, lambdaP, lambdaQ, stdev, numIter);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new ThompsonSamplingInteractivePMFRecommender<>(userIndex, itemIndex, ignoreUnknown, rngSeed, k, lambdaP, lambdaQ, stdev, numIter);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.PMFBANDIT + "-" + k + "-" + lambdaP + "-" + lambdaQ + "-" + stdev + "-" + numIter + "-" + PMFBanditIdentifiers.THOMPSON + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
