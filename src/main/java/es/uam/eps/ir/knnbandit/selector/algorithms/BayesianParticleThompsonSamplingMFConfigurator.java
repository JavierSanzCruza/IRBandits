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

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.ParticleThompsonSamplingMF;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles.PTSMFParticleFactory;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles.PTSMParticleFactories;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

/**
 * Class for configuring the Bayesian variant of the Particle Thompson sampling algorithm.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.ParticleThompsonSamplingMF
 */
public class BayesianParticleThompsonSamplingMFConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for selecting whether the algorithm is updated with items unknown by the system or not.
     */
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    /**
     * Identifier for the standard deviation of the item factors.
     */
    private static final String SIGMAQ = "sigmaQ";
    /**
     * Identifier for overall standard deviation.
     */
    private static final String STDEV = "stdev";
    /**
     * Identifier for the number of latent factors.
     */
    private static final String K = "k";
    /**
     * Identifier for the number of particles to use.
     */
    private static final String NUMP = "numParticles";
    /**
     * Identifier for the shape of the inverse gamma distributions for the user latent factor variance.
     */
    private static final String ALPHA = "alpha";
    /**
     * Identifier for the rate of the inverse gamma distributions for the user latent factor variance.
     */
    private static final String BETA = "beta";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        int k = object.getInt(K);
        double sigmaQ = object.getDouble(SIGMAQ);
        double stdev = object.getDouble(STDEV);
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);
        int numP = object.getInt(NUMP);

        return new BayesianParticleThompsonSamplingMFInteractiveRecommenderSupplier(k, sigmaQ, alpha, beta, stdev, numP, ignoreUnknown);
    }

    /**
     * Class that configures an particle Thompson sampling matrix factorization approach with Bayesian particles.
     */
    private class BayesianParticleThompsonSamplingMFInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        /**
         * True if we only use ratings we know about for the update procedure.
         */
        private final boolean ignoreUnknown;
        /**
         * The number of latent factors.
         */
        private final int k;
        /**
         * The variance for the item vectors.
         */
        private final double sigmaQ;
        /**
         * The shape of the inverse gamma distributions for the user latent factor variance.
         */
        private final double alpha;
        /**
         * The rate of the inverse gamma distributions for the user latent factor variance.
         */
        private final double beta;
        /**
         * The variance for the ratings.
         */
        private final double stdev;
        /**
         * The number of particles.
         */
        private final int numP;

        /**
         * Constructor.
         *
         * @param k             the number of latent factors.
         * @param sigmaQ        the variance for the item vectors.
         * @param alpha         the shape of the inverse gamma distributions for the user latent factor variance.
         * @param beta          the rate of the inverse gamma distributions for the user latent factor variance.
         * @param stdev         the variance for the ratings.
         * @param numP          the number of particles.
         * @param ignoreUnknown true if we only use ratings we know about for the update procedure, false otherwise.
         */
        public BayesianParticleThompsonSamplingMFInteractiveRecommenderSupplier(int k, double sigmaQ, double alpha, double beta, double stdev, int numP, boolean ignoreUnknown)
        {
            this.ignoreUnknown = ignoreUnknown;
            this.k = k;
            this.sigmaQ = sigmaQ;
            this.numP = numP;
            this.alpha = alpha;
            this.beta = beta;
            this.stdev = stdev;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            PTSMFParticleFactory<U, I> factory = PTSMParticleFactories.bayesianFactory(k, stdev, sigmaQ, alpha, beta);
            return new ParticleThompsonSamplingMF<>(userIndex, itemIndex, ignoreUnknown, numP, factory);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            PTSMFParticleFactory<U, I> factory = PTSMParticleFactories.bayesianFactory(k, stdev, sigmaQ, alpha, beta);
            return new ParticleThompsonSamplingMF<>(userIndex, itemIndex, ignoreUnknown,  rngSeed, numP, factory);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.BAYESIANPTS + "-" + k + "-" + numP + "-" + sigmaQ + "-" + alpha + "-" + beta + "-" + stdev + "-"  + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
