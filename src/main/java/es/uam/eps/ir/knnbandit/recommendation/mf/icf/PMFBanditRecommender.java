package es.uam.eps.ir.knnbandit.recommendation.mf.icf;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;

/**
 * Interactive contact recommendation algorithm based on the combination of probabilistic
 * matrix factorization with multi-armed bandit algorithms for selecting items.
 * <p>
 * Zhao, X., Zhang, W., Wang, J. Interactive Collaborative filtering. CIKM 2013.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public abstract class PMFBanditRecommender<U, I> extends InteractiveRecommender<U, I>
{
    /**
     * An algebra to perform matrix operations.
     */
    protected static final Algebra ALG = new Algebra();
    /**
     * Prior standard deviation for the whole set of ratings
     */
    protected final double stdev;
    /**
     * Function depending on the prior standard deviation for the user factors.
     */
    protected final double lambdaP;
    /**
     * Function depending on the prior standard deviation for the item factors.
     */
    protected final double lambdaQ;
    /**
     * Number of latent factors.
     */
    protected final int k;
    /**
     * Number of iterations for training
     */
    private final int numIter;
    /**
     * List of A matrices
     */
    protected DoubleMatrix2D[] As;
    /**
     * List of B vectors
     */
    protected DoubleMatrix1D[] bs;

    /**
     * User matrix
     */
    protected DoubleMatrix2D P;
    /**
     * Item matrix
     */
    protected DoubleMatrix2D Q;
    /**
     * Standard deviation matrix for the user factors.
     */
    protected DoubleMatrix2D[] stdevP;
    /**
     * Standard deviation matrix for the item factors.
     */
    protected DoubleMatrix2D[] stdevQ;

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      Preference data.
     * @param ignoreUnknown True if we must ignore unknown items when updating.
     * @param k             Number of latent factors to use
     * @param stdevP        Prior standard deviation for the user factors.
     * @param stdevQ        Prior standard deviation for the item factors.
     * @param stdev         Prior standard deviation for the ratings.
     * @param numIter       Number of training iterations.
     */
    public PMFBanditRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreUnknown, int k, double stdevP, double stdevQ, double stdev, int numIter)
    {
        super(uIndex, iIndex, prefData, ignoreUnknown);

        this.stdev = stdev;
        this.lambdaP = stdevP / stdev;
        this.lambdaQ = stdevQ / stdev;

        this.numIter = numIter;

        this.k = k;

        //this.P = new DenseDoubleMatrix2D(uIndex.numUsers(), k);
        //this.Q = new DenseDoubleMatrix2D(iIndex.numItems(), k);
    }

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      Preference data.
     * @param ignoreUnknown True if we must ignore unknown items when updating.
     * @param k             Number of latent factors to use
     * @param stdevP        Prior standard deviation for the user factors.
     * @param stdevQ        Prior standard deviation for the item factors.
     * @param stdev         Prior standard deviation for the ratings.
     * @param notReciprocal Not reciprocal
     * @param numIter       Number of training iterations.
     */
    public PMFBanditRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreUnknown, boolean notReciprocal, int k, double stdevP, double stdevQ, double stdev, int numIter)
    {
        super(uIndex, iIndex, prefData, ignoreUnknown, notReciprocal);

        this.stdev = stdev * stdev;
        this.lambdaP = stdevP * stdevP / this.stdev;
        this.lambdaQ = stdevQ * stdevQ / this.stdev;

        this.numIter = numIter;

        this.k = k;

        //this.P = new DenseDoubleMatrix2D(uIndex.numUsers(), k);
        //this.Q = new DenseDoubleMatrix2D(iIndex.numItems(), k);
    }
    // First, we factorize the data. The algorithm evolves as follows: first, we evaluate over


    @Override
    protected void initializeMethod()
    {
        // First, we initialize the matrices
        this.P = new DenseDoubleMatrix2D(uIndex.numUsers(), k);

        // It is enough to initialize the Q matrix with random values.
        this.Q = new DenseDoubleMatrix2D(iIndex.numItems(), k);
        this.Q.assign(x -> Math.sqrt(1.0 / k) * Math.random());

        // Then, we declare the standard deviation matrices for the users.
        this.stdevP = new DenseDoubleMatrix2D[uIndex.numUsers()];
        for (int i = 0; i < numUsers(); ++i)
        {
            stdevP[i] = new DenseDoubleMatrix2D(k, k);
            for (int j = 0; j < k; j++)
            {
                stdevP[i].setQuick(j, j, this.lambdaP / this.stdev);
            }
        }

        // Then, we declare the standard deviation matrices for the items.
        this.stdevQ = new DenseDoubleMatrix2D[iIndex.numItems()];
        for (int i = 0; i < numItems(); ++i)
        {
            stdevQ[i] = new DenseDoubleMatrix2D(k, k);
            for (int j = 0; j < k; j++)
            {
                stdevQ[i].setQuick(j, j, this.lambdaQ / this.stdev);
            }
        }

        // We finally apply ALS for training the algorithm.
        for (int i = 0; i < this.numIter; ++i)
        {
            set_min_P();
            set_min_Q();
        }
    }

    /**
     * Fixing the item feature vectors, train the user feature vectors using alternate
     * least squares (ALS).
     */
    private void set_min_P()
    {
        // First, find q_i * q_i^t
        DenseDoubleMatrix2D[] A2P = new DenseDoubleMatrix2D[this.numItems()];
        this.trainData.getIidxWithPreferences().parallel().forEach(iidx ->
        {
            A2P[iidx] = new DenseDoubleMatrix2D(this.k, this.k);
            DoubleMatrix1D qi = this.Q.viewRow(iidx);
            ALG.multOuter(qi, qi, A2P[iidx]);
        });

        DenseDoubleMatrix2D[] As = new DenseDoubleMatrix2D[this.numUsers()];
        DenseDoubleMatrix1D[] bs = new DenseDoubleMatrix1D[this.numUsers()];

        this.trainData.getAllUidx().parallel().forEach(uidx ->
        {
            // For user u, find the A and b matrices.
            DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(this.k, this.k);
            DenseDoubleMatrix1D b = new DenseDoubleMatrix1D(this.k);

            for (int i = 0; i < k; ++i)
            {
                A.setQuick(i, i, A.getQuick(i, i) + this.lambdaP);
            }

            // If no information is found, A = lambdaP*I, b = 0.
            // Otherwise... A = lambdaP*I + sum_i q_i q_i^T
            if (this.trainData.numItems(uidx) > 0)
            {
                this.trainData.getUidxPreferences(uidx).forEach(iv ->
                {
                    int iidx = iv.v1;
                    double rui = iv.v2;

                    A.assign(A2P[iidx], (x, y) -> x + y);
                    b.assign(Q.viewRow(iidx), (x, y) -> x + rui * y);
                });
            }

            DenseDoubleMatrix1D aux = new DenseDoubleMatrix1D(this.k);
            aux.assign(b);

            DenseDoubleMatrix2D sigmaI = new DenseDoubleMatrix2D(this.k, this.k);
            for (int i = 0; i < k; ++i)
            {
                sigmaI.setQuick(i, i, this.stdev);
            }

            // Find A^-1 b
            LUDecompositionQuick lu = new LUDecompositionQuick(0);
            lu.decompose(A);
            lu.solve(aux);
            P.viewRow(uidx).assign(aux);

            // Find A^-1 sigma
            lu.solve(sigmaI);
            this.stdevP[uidx] = sigmaI;

            As[uidx] = A;
            bs[uidx] = b;
        });

        this.As = As;
        this.bs = bs;
    }

    private void set_min_Q()

    {
        // First, find p_u * p_u^t
        DenseDoubleMatrix2D[] A2P = new DenseDoubleMatrix2D[this.numUsers()];
        prefData.getUidxWithPreferences().parallel().forEach(uidx ->
        {
            A2P[uidx] = new DenseDoubleMatrix2D(k, k);
            DoubleMatrix1D pu = P.viewRow(uidx);
            ALG.multOuter(pu, pu, A2P[uidx]);
        });

        trainData.getAllIidx().parallel().forEach(iidx ->
        {
            // For user u, find the A and b matrices.
            DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(this.k, this.k);
            DenseDoubleMatrix1D b = new DenseDoubleMatrix1D(this.k);

            for (int i = 0; i < k; ++i)
            {
                A.setQuick(i, i, A.getQuick(i, i) + lambdaQ);
            }

            // If no information is found, A = lambdaP*I, b = 0.
            // Otherwise... A = lambdaP*I + sum_i q_i q_i^T
            if (trainData.numUsers(iidx) > 0)
            {
                trainData.getIidxPreferences(iidx).forEach(iv ->
                {
                    int uidx = iv.v1;
                    double rui = iv.v2;

                    A.assign(A2P[uidx], (x, y) -> x + y);
                    b.assign(P.viewRow(uidx), (x, y) -> x + rui * y);
                });
            }

            DenseDoubleMatrix2D sigmaI = new DenseDoubleMatrix2D(this.k, this.k);
            for (int i = 0; i < k; ++i)
            {
                sigmaI.setQuick(i, i, stdev);
            }

            // Find A^-1 b
            LUDecompositionQuick lu = new LUDecompositionQuick(0);
            lu.decompose(A);
            lu.solve(b);
            Q.viewRow(iidx).assign(b);

            // Find A^-1 sigma
            lu.solve(sigmaI);
            stdevQ[iidx] = sigmaI;
        });
    }
}
