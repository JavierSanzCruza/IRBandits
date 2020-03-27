package es.uam.eps.ir.knnbandit.recommendation.mf.icf;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Interactive contact recommendation algorithm based on the combination of probabilistic
 * matrix factorization with multi-armed bandit algorithms for selecting items.
 * <p>
 * Uses Thompson sampling as the method for selecting the item to recommend.
 * <p>
 * Zhao, X., Zhang, W., Wang, J. Interactive Collaborative filtering. CIKM 2013.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public class ThompsonSamplingPMFBanditRecommender<U, I> extends PMFBanditRecommender<U, I>
{
    /**
     * Sampled vector from the item distribution.
     */
    private DoubleMatrix1D lastqi;
    /**
     * Matrices L such that A = L^T L, where A is the user covariance matrix.
     */
    private DoubleMatrix2D[] userDecomposed;

    private DoubleMatrix2D[] userEigenvalues;
    /**
     * Matrices L such that A = L^T L, where A is the item covariance matrix.
     */
    private DoubleMatrix2D[] itemDecomposed;

    private DoubleMatrix2D[] itemEigenvalues;

    /**
     * Constructor.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param prefData  Preference data.
     * @param hasRating True if we must ignore unknown items when updating.
     * @param k         Number of latent factors to use
     * @param stdevP    Prior standard deviation for the user factors.
     * @param stdevQ    Prior standard deviation for the item factors.
     * @param stdev     Prior standard deviation for the ratings.
     * @param numIter   Number of training iterations.
     */
    public ThompsonSamplingPMFBanditRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRating, int k, double stdevP, double stdevQ, double stdev, int numIter)
    {
        super(uIndex, iIndex, prefData, hasRating, k, stdevP, stdevQ, stdev, numIter);
    }

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      Preference data.
     * @param hasRating     True if we must ignore unknown items when updating.
     * @param k             Number of latent factors to use
     * @param stdevP        Prior standard deviation for the user factors.
     * @param stdevQ        Prior standard deviation for the item factors.
     * @param stdev         Prior standard deviation for the ratings.
     * @param notReciprocal Not reciprocal
     * @param numIter       Number of training iterations.
     */
    public ThompsonSamplingPMFBanditRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRating, boolean notReciprocal, int k, double stdevP, double stdevQ, double stdev, int numIter)
    {
        super(uIndex, iIndex, prefData, hasRating, notReciprocal, k, stdevP, stdevQ, stdev, numIter);
    }

    /**
     * Given a covariance matrix A, finds a matrix L such that A = L^T L
     *
     * @param covarianceMatrix the covariance matrix.
     * @return the covariance matrix.
     */
    private static Pair<DoubleMatrix2D> findL(DoubleMatrix2D covarianceMatrix)
    {
        int k = covarianceMatrix.rows();

        // First, compute the eigenvalue decomposition for the covariance matrix.
        EigenvalueDecomposition eigen = new EigenvalueDecomposition(covarianceMatrix);
        // Find the eigenvector matrix
        DoubleMatrix2D V = eigen.getV();
        // Find the eigenvalues diagonal matrix (they exist and they are real, since the covariance
        // matrix is symmetrical)
        DoubleMatrix2D D = eigen.getD();

        // Find the square root of L
        /*for(int i = 0; i < k; ++i)
        {
            D.setQuick(i,i,Math.sqrt(D.getQuick(i,i)));
        }*
         */

        // Multiply both matrices
        //DoubleMatrix2D L = ALG.mult(V,D);
        return new Pair<>(V, D);
    }

    @Override
    public void initializeMethod()
    {
        super.initializeMethod();

        userDecomposed = new DoubleMatrix2D[this.numUsers()];
        userEigenvalues = new DoubleMatrix2D[this.numUsers()];
        for (int uidx = 0; uidx < this.numUsers(); ++uidx)
        {
            Pair<DoubleMatrix2D> pair = findL(this.stdevP[uidx]);
            userDecomposed[uidx] = pair.v1();
            userEigenvalues[uidx] = pair.v2();
        }

        itemDecomposed = new DoubleMatrix2D[this.numItems()];
        itemEigenvalues = new DoubleMatrix2D[this.numItems()];

        for (int iidx = 0; iidx < this.numItems(); ++iidx)
        {

            Pair<DoubleMatrix2D> pair = findL(this.stdevQ[iidx]);
            itemDecomposed[iidx] = pair.v1();
            itemEigenvalues[iidx] = pair.v2();
        }
    }

    @Override
    public int next(int uidx)
    {
        IntList list = this.availability.get(uidx);
        if (list == null || list.isEmpty())
        {
            return -1;
        }

        // First, we estimate the user vector from a Multivariate Gaussian distribution
        DoubleMatrix1D originalPU = this.P.viewRow(uidx);
        DoubleMatrix1D pu = this.sampleMultivariateNormalDistrib(originalPU, this.userDecomposed[uidx], this.userEigenvalues[uidx]);

        // Next, for each item, we sample the item vector, and compute the score.
        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();
        List<DoubleMatrix1D> itemVectors = new ArrayList<>();

        for (int iidx : list)
        {
            DoubleMatrix1D originalQi = this.Q.viewRow(iidx);
            // First, we sample an item vector, using a Multivariate Gaussian distribution
            DoubleMatrix1D qi = this.sampleMultivariateNormalDistrib(originalQi, this.itemDecomposed[iidx], this.itemEigenvalues[iidx]);

            // Find the product of the sampled vectors: p_u^t *
            double val = ALG.mult(pu, qi);

            if (Double.isNaN(val))
            {
                val = Double.NEGATIVE_INFINITY;
            }
            if (top.isEmpty() || max < val)
            {
                top = new IntArrayList();
                itemVectors = new ArrayList<>();
                top.add(iidx);
                itemVectors.add(qi);
                max = val;
            }
            else if (max == val)
            {
                top.add(iidx);
                itemVectors.add(qi);
            }
        }

        int topSize = top.size();
        if (topSize == 1)
        {
            this.lastqi = itemVectors.get(0);
            return top.get(0);
        }
        else
        {
            int idx = rng.nextInt(top.size());
            this.lastqi = itemVectors.get(idx);
            return top.get(idx);
        }
    }

    @Override
    public void updateMethod(int uidx, int iidx, double value)
    {
        if (this.lastqi != null)
        {
            // First, obtain the last qi
            DenseDoubleMatrix2D aux = new DenseDoubleMatrix2D(this.k, this.k);
            ALG.multOuter(this.lastqi, this.lastqi, aux);

            // First, update the values for the A and b matrices for user u
            As[uidx].assign(aux, (x, y) -> x + y);
            bs[uidx].assign(this.lastqi, (x, y) -> x + value * y);

            // Then, find A^-1 b and A^-1 sigma^2

            ALG.inverse(As[uidx]);
            DenseDoubleMatrix1D c = new DenseDoubleMatrix1D(this.k);


            /*LUDecompositionQuick lu = new LUDecompositionQuick(0);
            DenseDoubleMatrix1D c = new DenseDoubleMatrix1D(this.k);
            c.assign(bs[uidx]);

            lu.decompose(As[uidx]);
            lu.solve(c);

            DenseDoubleMatrix2D sigmaI = new DenseDoubleMatrix2D(this.k, this.k);
            for (int i = 0; i < k; ++i)
            {
                sigmaI.setQuick(i, i, this.stdev);
            }
            lu.solve(sigmaI);

            this.P.viewRow(uidx).assign(c);
            this.stdevP[uidx] = sigmaI;

            // Update the decomposition for the covariance matrix.
            Pair<DoubleMatrix2D> pair = findL(sigmaI);
            this.userDecomposed[uidx] = pair.v1();
            this.userEigenvalues[uidx] = pair.v2();

            this.lastqi = null;*/
        }
    }

    /**
     * Samples from a Multivariate Normal distribution.
     *
     * @param mean        mean.
     * @param eigenvector eigenvector matrix.
     * @param eigenvalues eigenvalues matrix.
     * @return the sampled vector.
     */
    private DoubleMatrix1D sampleMultivariateNormalDistrib(DoubleMatrix1D mean, DoubleMatrix2D eigenvector, DoubleMatrix2D eigenvalues)
    {
        // Sample the values.
        Random rng = new Random();
        DenseDoubleMatrix1D dense = new DenseDoubleMatrix1D(this.k);
        for (int i = 0; i < this.k; ++i)
        {
            dense.setQuick(i, Math.sqrt(eigenvalues.getQuick(i, i)) * rng.nextGaussian());
        }

        DoubleMatrix1D res = ALG.mult(eigenvector, dense);
        eigenvector.zMult(dense, dense);
        res.assign(mean, (x, y) -> x + y);
        return res;
    }
}
