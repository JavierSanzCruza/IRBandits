/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import es.uam.eps.ir.knnbandit.recommendation.mf.FastParticle;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;

import java.util.Random;

/**
 * Particle for the Particle Thompson Sampling matrix factorization algorithm.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public abstract class PTSMFParticle<U, I> extends FastParticle<U, I>
{
    /**
     * Algebra for calculations.
     */
    protected final static Algebra ALG = new Algebra();
    /**
     * Number of latent factors.
     */
    protected final int K;
    /**
     * Random number generator.
     */
    protected final Random rng;
    /**
     * True if the particle uses the Bayesian algorithm, false otherwise.
     */
    private final boolean bayesian;
    /**
     * User matrix
     */
    protected DoubleMatrix2D P;
    /**
     * Variance of the ratings.
     */
    protected double sigma;
    /**
     * Variance of the user latent factors.
     */
    protected double sigmaP;
    /**
     * Variance of the item latent factors.
     */
    protected double sigmaQ;
    /**
     * Item matrix
     */
    private DoubleMatrix2D Q;
    /**
     * Inverse of the covariance matrix for the user vectors.
     */
    private DoubleMatrix2D[] Au;
    /**
     * Vector representing the tastes of each user.
     */
    private DoubleMatrix1D[] bu;
    /**
     * Mean value for producing the user vector.
     */
    private DoubleMatrix1D[] muU;
    /**
     * Inverse of the covariance matrix for the item vectors.
     */
    private DoubleMatrix2D[] Ai;
    /**
     * Vector representing the tastes of each item.
     */
    private DoubleMatrix1D[] bi;

    /**
     * Constructor.
     *
     * @param uIndex User index.
     * @param iIndex Item index.
     * @param K      Number of latent factors.
     * @param sigma  Variance of the ratings.
     * @param sigmaP Variance of the user latent factors.
     * @param sigmaQ Variance of the item latent factors.
     */
    public PTSMFParticle(FastUserIndex<U> uIndex, FastItemIndex<I> iIndex, int K, double sigma, double sigmaP, double sigmaQ, boolean bayesian)
    {
        super(uIndex, iIndex);
        this.sigma = sigma;
        this.sigmaP = sigmaP;
        this.sigmaQ = sigmaQ;
        this.K = K;
        this.rng = new Random();

        this.bayesian = bayesian;
    }

    @Override
    public void initialize()
    {
        // We initialize the different users.
        this.P = new DenseDoubleMatrix2D(this.numUsers, this.K);
        this.Q = new DenseDoubleMatrix2D(this.numItems, this.K);

        // Initialize users
        this.Au = new DenseDoubleMatrix2D[numUsers];
        this.bu = new DenseDoubleMatrix1D[numUsers];

        DoubleMatrix2D auxU = DoubleFactory2D.sparse.identity(this.K);
        for (int j = 0; j < this.K; ++j)
        {
           auxU.setQuick(j, j, 1.0 / sigmaP);
        }
        DoubleMatrix2D auxUInv = ALG.inverse(auxU);
        DoubleMatrix1D auxB = new DenseDoubleMatrix1D(this.K);
        MultivariateNormalDistribution mndU = new MultivariateNormalDistribution(auxB.toArray(), auxUInv.toArray());

        for(int uidx = 0; uidx < numUsers; ++uidx)
        {
            this.Au[uidx] = new DenseDoubleMatrix2D(this.K, this.K);
            this.Au[uidx].assign(auxU);
            this.bu[uidx] = new DenseDoubleMatrix1D(this.K);

            double[] pu = mndU.sample();
            this.P.viewRow(uidx).assign(pu);
        }

        // Initialize items
        this.Ai = new DenseDoubleMatrix2D[numItems];
        this.bi = new DenseDoubleMatrix1D[numItems];

        DoubleMatrix2D auxI = DoubleFactory2D.sparse.identity(this.K);
        for (int j = 0; j < this.K; ++j)
        {
            auxI.setQuick(j, j, 1.0 / sigmaQ);
        }
        DoubleMatrix2D auxIInv = ALG.inverse(auxI);
        MultivariateNormalDistribution mndI = new MultivariateNormalDistribution(auxB.toArray(), auxIInv.toArray());


        for(int iidx = 0; iidx < numItems; ++iidx)
        {
            this.Ai[iidx] = new DenseDoubleMatrix2D(this.K, this.K);
            this.Ai[iidx].assign(auxI);
            this.bi[iidx] = new DenseDoubleMatrix1D(this.K);

            double[] qi = mndI.sample();
            this.Q.viewRow(iidx).assign(qi);
        }
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        // We first update the user:
        DoubleMatrix2D aux = new DenseDoubleMatrix2D(this.K, this.K);
        DoubleMatrix1D qi = this.Q.viewRow(iidx);
        ALG.multOuter(qi,qi,aux);

        // Update the Au and bu matrices
        this.Au[uidx].assign(aux, (x, y) -> x + 1 / sigma * y);
        this.bu[uidx].assign(qi, (x, y) -> x + value * y);

        aux.assign(this.Au[uidx]);
        if (bayesian) // If bayesian, add 1/sigmaP to the diagonal (otherwise it is already added)
        {
            for (int k = 0; k < K; ++k)
            {
                aux.setQuick(k, k, 1 / sigmaP);
            }
        }

        DoubleMatrix1D mult = new DenseDoubleMatrix1D(this.K);
        DoubleMatrix2D inverse = ALG.inverse(this.Au[uidx]);
        inverse.zMult(this.bu[uidx], mult);

        // Update vector pu
        DoubleMatrix1D pu = this.gaussianSample(mult, inverse, 1.0 / sigma);
        // Update the norm of U
        this.updateNormP(uidx, pu);
        // Assign the new pu.
        this.P.viewRow(uidx).assign(pu);

        ALG.multOuter(pu, pu, aux);
        this.Ai[iidx].assign(aux, (x, y) -> x + 1 / sigma * y);
        this.bi[iidx].assign(pu, (x, y) -> x + value * y);

        mult = new DenseDoubleMatrix1D(this.K);
        inverse = ALG.inverse(this.Ai[iidx]);
        inverse.zMult(this.bi[iidx], mult);

        // Update vector qi
        DoubleMatrix1D newQi = this.gaussianSample(mult, inverse, 1.0 / sigma);
        this.Q.viewRow(iidx).assign(newQi);

        this.sigmaP = this.updateSigmaP();
    }

    /**
     * Updates the value of sigmaP
     *
     * @return the new value of sigmaP.
     */
    protected abstract double updateSigmaP();

    /**
     * Updates the norm of the user matrix.
     *
     * @param uidx the modified user.
     * @param pu   the new vector for the user.
     */
    protected abstract void updateNormP(int uidx, DoubleMatrix1D pu);

    @Override
    public double getEstimatedReward(int uidx, int iidx)
    {
        return ALG.mult(this.P.viewRow(uidx), this.Q.viewRow(iidx));
    }

    @Override
    public double getWeight(int uidx, int iidx, double value)
    {
        double mean = ALG.mult(this.Q.viewRow(iidx), this.muU[uidx]);

        DoubleMatrix1D aux = new DenseDoubleMatrix1D(this.K);
        DoubleMatrix2D A;
        if(bayesian)
        {
            A = new DenseDoubleMatrix2D(this.K, this.K);
            A.assign(Au[uidx]);
            for(int k = 0; k < this.K; ++k)
            {
                A.setQuick(k, k, A.getQuick(k, k) + 1.0 / sigmaP);
            }
        }
        else
        {
            A = Au[uidx];
        }

        A.zMult(this.Q.viewRow(iidx), aux);
        double variance = 1.0/sigma + ALG.mult(aux, this.Q.viewRow(iidx));

        NormalDistribution nd = new NormalDistribution(mean, Math.sqrt(variance));
        return nd.density(value);
    }


    /**
     * Samples from a Gaussian distribution, with mean mu, and covariance matrix
     * sigma*covariance.
     *
     * @param mu         the mean of the distribution.
     * @param covariance the covariance matrix.
     * @param sigma      a product of the distribution.
     * @return the sample from the Gaussian distribution.
     */
    private DoubleMatrix1D gaussianSample(DoubleMatrix1D mu, DoubleMatrix2D covariance, double sigma)
    {
        EigenvalueDecomposition decomp = new EigenvalueDecomposition(covariance);

        DoubleMatrix1D eigen = decomp.getRealEigenvalues();
        DoubleMatrix1D matrix = new DenseDoubleMatrix1D(this.K);
        for (int k = 0; k < this.K; ++k)
        {
            matrix.setQuick(k, Math.sqrt(eigen.getQuick(k) * sigma) * rng.nextGaussian());
        }

        decomp.getV().zMult(matrix, matrix);
        matrix.assign(mu, Double::sum);
        return matrix;
    }

    /**
     * Given a particle, clones the different fields.
     *
     * @param particle the particle.
     */
    public void clone(PTSMFParticle<U, I> particle)
    {
        particle.P = new DenseDoubleMatrix2D(this.numUsers, this.K);
        particle.P.assign(this.P);
        particle.Q = new DenseDoubleMatrix2D(this.numItems, this.K);
        particle.Q.assign(this.Q);

        particle.Au = new DoubleMatrix2D[numUsers];
        particle.bu = new DoubleMatrix1D[numUsers];
        particle.muU = new DoubleMatrix1D[numUsers];
        for (int uidx = 0; uidx < numUsers; ++uidx)
        {
            particle.Au[uidx] = new DenseDoubleMatrix2D(this.K, this.K);
            particle.bu[uidx] = new DenseDoubleMatrix1D(this.K);
            particle.muU[uidx] = new DenseDoubleMatrix1D(this.K);
            particle.Au[uidx].assign(this.Au[uidx]);
            particle.bu[uidx].assign(this.bu[uidx]);
            particle.muU[uidx].assign(this.muU[uidx]);
        }

        particle.Ai = new DoubleMatrix2D[numItems];
        particle.bi = new DoubleMatrix1D[numItems];
        for (int iidx = 0; iidx < numItems; ++iidx)
        {
            particle.Ai[iidx] = new DenseDoubleMatrix2D(this.K, this.K);
            particle.bi[iidx] = new DenseDoubleMatrix1D(this.K);
            particle.Ai[iidx].assign(this.Ai[iidx]);
            particle.bi[iidx].assign(this.bi[iidx]);
        }
    }


}
