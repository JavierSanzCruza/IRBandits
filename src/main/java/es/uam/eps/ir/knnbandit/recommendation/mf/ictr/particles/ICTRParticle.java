package es.uam.eps.ir.knnbandit.recommendation.mf.ictr.particles;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import es.uam.eps.ir.knnbandit.recommendation.mf.FastParticle;
import es.uam.eps.ir.knnbandit.recommendation.mf.Particle;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;

import java.util.Random;

/**
 * Abstract class for the definition of an individual particle for the Interactive Collaborative Topic Regression (ICTR)
 * approach.
 * <p>
 * Wang, Q. et al. Online Interactive Collaborative Filtering Using Multi-Armed Bandit with Dependent Arms. IEEE TKDE (2019)
 *
 * @author Javier Sanz-Cruzado
 * @author Pablo Castells
 * <p>
 * TODO: Remaining doubts about the correctness of the implementation:
 * TODO: is lambda, alpha, beta unique for each particle? i.e. do they depend on the user/item? Otherwise, they would not be personalized...
 * TODO: is it necessary to update everything each time? At least, p_m must only be updated once
 * TODO: as I do not know the value of z_{m,t}, how do I estimate E[p_{m,k}|\lambda_k, r_m,t]?
 */
public class ICTRParticle<U, I> extends FastParticle<U, I>
{
    /**
     * Algebra for computing matrix operations.
     */
    protected static final Algebra ALG = new Algebra();
    /**
     * The number of latent factors.
     */
    protected final int K;
    /**
     * Random number generator.
     */
    protected final Random rng;
    /**
     * User matrix.
     */
    protected DoubleMatrix2D P;
    /**
     * Item matrix.
     */
    protected DoubleMatrix2D Q;
    /**
     * Matrix for modelling dependencies between items (arms).
     */
    protected DoubleMatrix2D phi;
    /**
     * Variance of the rating prediction for each item.
     */
    protected DoubleMatrix1D sigma;
    /**
     * Hyperparameters to determine the Gaussian distribution of q_n (means)
     */
    protected DoubleMatrix2D muQ;
    /**
     * Hyperparameters to determine the Gaussian distribution of q_n (covariance matrices)
     */
    protected DoubleMatrix2D[] sigmaQ;
    /**
     * Hyperparameters to determine the Dirichlet distribution of the user factors.
     */
    protected DoubleMatrix2D lambdas;
    /**
     * Hyperparameters to determine the Dirichlet distributions of the factors according to the item ratings.
     */
    protected DoubleMatrix2D etas;
    /**
     * Hyperparameter for determining the variance of the rating prediction for each item.
     */
    protected DoubleMatrix1D alpha;
    /**
     * Hyperparameter for determining the variance of the rating prediction for each item.
     */
    protected DoubleMatrix1D beta;

    /**
     * Constructor.
     *
     * @param uIndex user index.
     * @param iIndex item index.
     * @param K      the number of latent factors.
     */
    public ICTRParticle(FastUserIndex<U> uIndex, FastItemIndex<I> iIndex, int K)
    {
        super(uIndex, iIndex);
        this.K = K;
        this.rng = new Random();
    }

    /**
     * Initializes the different elements of a particle.
     */
    public void initialize()
    {
        DoubleFactory2D factory = DoubleFactory2D.dense;
        // Initialize the lambdas
        this.lambdas = new DenseDoubleMatrix2D(K, numUsers);
        this.lambdas.assign(1.0);

        // Initialize the etas
        this.etas = new DenseDoubleMatrix2D(K, numItems);
        this.etas.assign(1.0);

        // Initialize the phi values
        this.phi = new DenseDoubleMatrix2D(K, numItems);
        for (int k = 1; k < K; ++k)
        {
            DoubleMatrix1D phi = dirichletSampling(etas.viewRow(k));
            this.phi.viewRow(k).assign(phi);
        }

        // Then, we initialize the p_u
        this.P = new DenseDoubleMatrix2D(numUsers, K);
        for (int uidx = 0; uidx < numUsers; ++uidx)
        {
            DoubleMatrix1D pu = dirichletSampling(lambdas.viewRow(uidx));
            this.P.viewRow(uidx).assign(pu);
        }

        // We initialize the mean vectors as 0.
        this.muQ = new DenseDoubleMatrix2D(numItems, K);
        this.sigmaQ = new DenseDoubleMatrix2D[numItems];
        // Initialize the alpha and beta parameters:
        this.alpha = new DenseDoubleMatrix1D(numItems);
        this.beta = new DenseDoubleMatrix1D(numItems);
        this.Q = new DenseDoubleMatrix2D(numItems, K);
        this.sigma = new DenseDoubleMatrix1D(numItems);

        for (int iidx = 0; iidx < numItems; ++iidx)
        {
            double a = 1;
            double b = 1;

            alpha.setQuick(iidx, a);
            beta.setQuick(iidx, b);

            double val = this.inverseGammaSample(a, b);
            this.sigma.setQuick(iidx, val);

            // We initialize Sigma_q as the identity matrix.
            this.sigmaQ[iidx] = factory.identity(K);

            // Initialize qi
            for (int j = 0; j < K; ++j)
            {
                double aux = Math.sqrt(val) * rng.nextGaussian();
                Q.setQuick(iidx, j, aux);
            }
        }
    }

    /**
     * Returns a copy of the particle.
     *
     * @return the copy of the particle.
     */
    public Particle<U, I> clone()
    {
        ICTRParticle<U, I> particle = new ICTRParticle<>(this.getUserIndex(), this.getItemIndex(), this.K);

        // Clone the user, item and interdependence matrices:
        particle.P = new DenseDoubleMatrix2D(this.numUsers, this.K);
        particle.P.assign(this.P);

        particle.Q = new DenseDoubleMatrix2D(this.numItems, this.K);
        particle.Q.assign(this.Q);

        particle.phi = new DenseDoubleMatrix2D(this.K, this.numItems);
        particle.phi.assign(this.phi);

        // Clone the sigma vector
        particle.sigma = new DenseDoubleMatrix1D(this.numItems);
        particle.sigma.assign(this.sigma);

        // Clone the hyperparameters
        particle.muQ = new DenseDoubleMatrix2D(this.numItems, this.K);
        particle.muQ.assign(this.muQ);

        particle.sigmaQ = new DoubleMatrix2D[numItems];
        for (int i = 0; i < numItems; ++i)
        {
            particle.sigmaQ[i] = new DenseDoubleMatrix2D(this.K, this.K);
            particle.sigmaQ[i].assign(this.sigmaQ[i]);
        }

        particle.alpha = new DenseDoubleMatrix1D(this.numItems);
        particle.alpha.assign(this.alpha);

        particle.beta = new DenseDoubleMatrix1D(this.numItems);
        particle.beta.assign(this.beta);

        particle.lambdas = new DenseDoubleMatrix2D(this.numUsers, this.K);
        particle.lambdas.assign(this.lambdas);

        return particle;
    }

    /**
     * Given a triplet (user, item, value), updates the values of the different variables of the system.
     *
     * @param uidx  The identifier of the user.
     * @param iidx  The identifier of the item.
     * @param value The value of the interaction.
     */
    public void update(int uidx, int iidx, double value)
    {
        // Step 1: Update the sufficient statistics for z_{m,t}:
        DenseDoubleMatrix1D thetas = new DenseDoubleMatrix1D(this.K);
        double sumLambdas = 0.0;

        // First, we compute the sum of the lambdas
        for (int k = 0; k < this.K; ++k)
        {
            sumLambdas += this.lambdas.getQuick(uidx, k);
        }

        double sumThetas = 0.0;
        // Then, find the theta[k] value
        for (int k = 0; k < this.K; ++k)
        {
            double sumEtas = 0.0;
            for (int n = 0; n < this.numItems; ++n)
            {
                sumEtas += this.etas.getQuick(k, n);
            }

            /* TODO: Not truly sure about this...*/
            double expectedP = (this.lambdas.getQuick(uidx, k) + value) / (sumLambdas + value);
            double expectedPhi = (this.phi.getQuick(k, iidx) + value) / (sumEtas + value);

            thetas.setQuick(k, expectedP * expectedPhi);
            sumThetas += thetas.getQuick(k);
        }

        // Step 2: Now, we sample the value of z_{m,t} using a multinomial distribution with \theta parameter.
        double rnd = rng.nextDouble();
        int z = this.multinomialSampling(thetas, sumThetas);

        // Step 3: Update the statistics for the hyperparameters:
        // sigmaQ^-1
        DoubleMatrix2D oldSigmaQ = this.sigmaQ[iidx];

        DoubleMatrix2D newSigmaQ = new DenseDoubleMatrix2D(this.K, this.K);
        ALG.multOuter(this.P.viewRow(uidx), this.P.viewRow(uidx), newSigmaQ);
        newSigmaQ.assign(oldSigmaQ, (x, y) -> x + y);

        // Now, find sigmaQ...
        DoubleMatrix2D inverse = ALG.inverse(newSigmaQ);

        // Find muQ
        DoubleMatrix1D oldMuQ = this.muQ.viewRow(iidx);

        DoubleMatrix1D newMuQ = new DenseDoubleMatrix1D(this.K);
        newMuQ.assign(this.P.viewRow(uidx));
        oldSigmaQ.zMult(oldMuQ, newMuQ, 1.0, value, false);
        inverse.zMult(newMuQ, newMuQ);

        // Update alpha
        double alpha = this.alpha.getQuick(iidx) + 0.5;
        this.alpha.setQuick(iidx, alpha);

        // Update beta
        DoubleMatrix1D newVal = new DenseDoubleMatrix1D(this.K);
        newSigmaQ.zMult(newMuQ, newVal);
        double beta = ALG.mult(newMuQ, newVal);

        oldSigmaQ.zMult(oldMuQ, newVal);
        beta += ALG.mult(oldMuQ, newVal);
        beta += value * value;
        beta /= 2.0;
        beta += this.beta.getQuick(iidx);

        this.beta.setQuick(iidx, beta);

        lambdas.setQuick(uidx, z, lambdas.getQuick(uidx, z) + value);
        for (int k = 0; k < K; ++k)
        {
            etas.setQuick(k, iidx, etas.getQuick(k, iidx) + value);
        }

        this.sigmaQ[iidx] = newSigmaQ;
        this.muQ.viewRow(iidx).assign(newMuQ);

        // Step 4: Sample the random variables.
        // First, sample sigma_n^2

        double sigma_n = this.inverseGammaSample(alpha, beta);
        this.sigma.setQuick(iidx, sigma_n);
        DoubleMatrix1D newP = this.dirichletSampling(lambdas.viewRow(uidx));
        this.P.viewRow(uidx).assign(newP);

        for (int k = 0; k < this.K; ++k)
        {
            DoubleMatrix1D newPhi = this.dirichletSampling(etas.viewRow(k));
            this.phi.viewRow(k).assign(newPhi);
        }

        for (int idx = 0; idx < this.numItems; ++idx)
        {
            DoubleMatrix1D matrix = this.gaussianSample(this.muQ.viewRow(idx), this.sigmaQ[idx], this.sigma.getQuick(idx));
            this.Q.viewRow(idx).assign(matrix);
        }
    }

    @Override
    public double getEstimatedReward(int uidx, int iidx)
    {
        return ALG.mult(this.P.viewRow(uidx), this.Q.viewRow(iidx));
    }

    @Override
    public double getWeight(int uidx, int iidx, double value)
    {
        // First, we find the average and variance of the Gaussian of the rating value.
        double mean = ALG.mult(this.P.viewRow(uidx), this.Q.viewRow(iidx));
        double var = this.sigma.getQuick(iidx);

        // Then, we find the value of the density function:
        double diff = (mean - value);
        double aux = 2 * var * var;

        double gaussian = 1.0 / Math.sqrt(Math.PI * aux);
        gaussian *= Math.exp(-diff / aux);


        double sum = 0.0;
        double lambdasSum = 0.0;
        double etasSum = 0.0;
        // We iterate then over the different aspects:
        for (int k = 0; k < K; ++k)
        {
            double lambda = this.lambdas.getQuick(uidx, k);
            double eta = this.etas.getQuick(k, iidx);
            lambdasSum += lambda;
            etasSum += eta;

            sum += lambda * eta;
        }

        // return the value of the fitness function.
        return gaussian * sum / (lambdasSum * etasSum);
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
        matrix.assign(mu, (x, y) -> x + y);
        return matrix;
    }


    /**
     * Function for sampling from a multinomial distribution.
     *
     * @param thetas the not-normalized probabilities.
     * @param sum    the sum of the not-normalized probabilities.
     * @return the sampled element.
     */
    private int multinomialSampling(DenseDoubleMatrix1D thetas, double sum)
    {
        double rnd = rng.nextDouble();
        int idx = -1;
        double w = 0.0;
        do
        {
            ++idx;
            w+= thetas.getQuick(idx);
        }
        while (rnd <= w);

        return idx;
    }

    /**
     * Function for sampling from a Dirichlet distribution.
     *
     * @param lambda the lambda hyperparameters.
     * @return the vector containing the sample.
     */
    private DoubleMatrix1D dirichletSampling(DoubleMatrix1D lambda)
    {
        int k = lambda.size();
        DoubleMatrix1D sample = new DenseDoubleMatrix1D(k);
        double sum = 0.0;
        for (int i = 0; i < k; ++i)
        {
            double val = this.gammaSample(lambda.getQuick(k));
            sum += val;
            sample.setQuick(i, val);
        }

        for (int i = 0; i < k; ++i)
        {
            sample.setQuick(i, sample.getQuick(i) / sum);
        }

        return sample;
    }

    /**
     * Function for sampling from an Inverse Gamma distribution.
     *
     * @param alpha the shape of the distribution.
     * @param beta  the scale of the distribution.
     * @return the sampled value.
     */
    private double inverseGammaSample(double alpha, double beta)
    {
        double sample = gammaSample(alpha) / beta;
        return 1.0 / sample;

    }

    /**
     * Function for sampling from a Gamma distribution.
     * <p>
     * This implementation was adapted from https://github.com/gesiscss/promoss.
     *
     * @param shape the shape of the distribution.
     * @return the sampled value.
     */
    private double gammaSample(double shape)
    {
        if (shape <= 0) // Not well defined, set to zero and skip
        {
            return 0;
        }
        else if (shape == 1) // Exponential
        {
            return -Math.log(rng.nextDouble());
        }
        else if (shape < 1) // Use Johnks generator
        {
            double c = 1.0 / shape;
            double d = 1.0 / (1 - shape);
            while (true)
            {
                double x = Math.pow(rng.nextDouble(), c);
                double y = x + Math.pow(rng.nextDouble(), d);
                if (y <= 1)
                {
                    return -Math.log(rng.nextDouble()) * x / y;
                }
            }
        }
        else // Bests algorithm
        {
            double b = shape - 1;
            double c = 3 * shape - 0.75;
            while (true)
            {
                double u = rng.nextDouble();
                double v = rng.nextDouble();
                double w = u * (1 - u);
                double y = Math.sqrt(c / w) * (u - 0.5);
                double x = b + y;
                if (x >= 0)
                {
                    double z = 64 * w * w * w * v * v;
                    if ((z <= (1 - 2 * y * y / x))
                            || (Math.log(z) <= 2 * (b * Math.log(x / b) - y)))
                    {
                        return x;
                    }
                }
            }
        }
    }

    /**
     * Obtains the variance for the item ratings.
     *
     * @param i the item.
     * @return the variance.
     */
    public double getVariance(I i)
    {
        int iidx = this.getItemIndex().item2iidx(i);
        return this.sigma.getQuick(iidx);
    }

    /**
     * Obtains the variance for the item ratings.
     *
     * @param iidx the item identifier.
     * @return the variance.
     */
    public double getVariance(int iidx)
    {
        return this.sigma.getQuick(iidx);
    }
}
