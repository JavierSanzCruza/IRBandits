/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles;

/**
 * Class that provides particle factories for the Particle Thompson Sampling algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class PTSMParticleFactories
{
    /**
     * Factory for the normal PTS particles.
     *
     * @param k      the number of latent factors.
     * @param sigma  the variance for the ratings.
     * @param sigmaP the variance for the user vectors.
     * @param sigmaQ the variance for the item vectors.
     * @param <U>    Type of the users.
     * @param <I>    Type of the items.
     * @return The particle factory.
     */
    public static <U, I> PTSMFParticleFactory<U, I> normalFactory(int k, double sigma, double sigmaP, double sigmaQ)
    {
        return (uIndex, iIndex) ->
        {
            PTSMFParticle<U, I> particle = new NormalPTSMFParticle<>(uIndex, iIndex, k, sigma, sigmaP, sigmaQ);
            particle.initialize();
            return particle;
        };
    }

    /**
     * Factory for the bayesian PTS particles.
     *
     * @param k      the number of latent factors.
     * @param sigma  the variance for the ratings.
     * @param sigmaQ the variance for the item vectors.
     * @param alpha  the shape of the inverse gamma distributions for the user latent factor variance.
     * @param beta   the rate of the inverse gamma distributions for the user latent factor variance.
     * @param <U>    Type of the users.
     * @param <I>    Type of the items.
     * @return The particle factory.
     */
    public static <U, I> PTSMFParticleFactory<U, I> bayesianFactory(int k, double sigma, double sigmaQ, double alpha, double beta)
    {
        return (uIndex, iIndex) ->
        {
            PTSMFParticle<U, I> particle = new BayesianPTSMFParticle<>(uIndex, iIndex, k, sigma, sigmaQ, alpha, beta);
            particle.initialize();
            return particle;
        };
    }
}
