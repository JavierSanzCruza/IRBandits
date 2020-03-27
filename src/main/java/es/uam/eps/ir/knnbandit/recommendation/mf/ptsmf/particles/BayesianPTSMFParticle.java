package es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles;

import cern.colt.matrix.DoubleMatrix1D;
import es.uam.eps.ir.knnbandit.recommendation.mf.Particle;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;

/**
 * Particle for the Bayesian version of the PTS algorithm.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public class BayesianPTSMFParticle<U, I> extends PTSMFParticle<U, I>
{
    /**
     * Shape of the inverse gamma distribution of the variance of user factors.
     */
    private final double alpha;
    /**
     * Shape of the inverse gamma distribution of the variance of item factors.
     */
    private final double beta;
    /**
     * Squared Frobenius norm of the user matrix.
     */
    private double normP;

    /**
     * Constructor.
     *
     * @param uIndex User index.
     * @param iIndex Item index.
     * @param K      Number of latent factors.
     * @param sigma  Variance of the ratings.
     * @param sigmaQ Variance of the item latent factors.
     * @param alpha  Shape of the sigmaP inverse gamma distribution.
     * @param beta   Rate of the sigmaP inverse gamma distribution.
     */
    public BayesianPTSMFParticle(FastUserIndex<U> uIndex, FastItemIndex<I> iIndex, int K, double sigma, double sigmaQ, double alpha, double beta)
    {
        super(uIndex, iIndex, K, sigma, 1.0, sigmaQ, true);
        this.alpha = alpha;
        this.beta = beta;
    }

    @Override
    public void initialize()
    {
        super.initialize();
        this.normP = ALG.normF(this.P);
        this.normP *= this.normP;
    }

    @Override
    protected double updateSigmaP()
    {
        double alphaP = this.alpha + (this.numUsers * this.K) / 2.0;
        double betaP = this.beta + this.normP / 2.0;

        double lambdaP = this.gammaSample(alphaP) / betaP;
        return 1 / lambdaP;
    }

    @Override
    protected void updateNormP(int uidx, DoubleMatrix1D pu)
    {
        DoubleMatrix1D oldPu = this.P.viewRow(uidx);
        double toDel = ALG.mult(oldPu, oldPu);

        double toAdd = ALG.mult(pu, pu);

        this.normP += toAdd - toDel;
    }

    @Override
    public Particle<U, I> clone()
    {
        return null;
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
}
