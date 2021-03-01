package es.uam.eps.ir.knnbandit.recommendation.mf.icf;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AdditiveRatingFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.FastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.stream.Stream;

/**
 * Interactive contact recommendation algorithm based on the combination of probabilistic
 * matrix factorization with multi-armed bandit algorithms for selecting items.
 * <p>
 * Zhao, X., Zhang, W., Wang, J. Interactive Collaborative filtering. CIKM 2013.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public abstract class InteractivePMFRecommender<U, I> extends AbstractInteractiveRecommender<U, I>
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
     * The current rating matrix.
     */
    protected FastUpdateablePreferenceData<U,I> retrievedData;


    /**
     * Constructor.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param hasRating True if we must ignore unknown items when updating.
     * @param k         Number of latent factors to use
     * @param stdevP    Prior standard deviation for the user factors.
     * @param stdevQ    Prior standard deviation for the item factors.
     * @param stdev     Prior standard deviation for the ratings.
     * @param numIter   Number of training iterations.
     */
    public InteractivePMFRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int k, double stdevP, double stdevQ, double stdev, int numIter)
    {
        super(uIndex, iIndex, hasRating);

        this.stdev = stdev;
        this.lambdaP = stdevP / stdev;
        this.lambdaQ = stdevQ / stdev;

        this.numIter = numIter;

        this.k = k;

        this.retrievedData = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);

        //this.P = new DenseDoubleMatrix2D(uIndex.numUsers(), k);
        //this.Q = new DenseDoubleMatrix2D(iIndex.numItems(), k);
    }

    /**
     * Constructor.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param hasRating True if we must ignore unknown items when updating.
     * @param k         Number of latent factors to use
     * @param stdevP    Prior standard deviation for the user factors.
     * @param stdevQ    Prior standard deviation for the item factors.
     * @param stdev     Prior standard deviation for the ratings.
     * @param numIter   Number of training iterations.
     */
    public InteractivePMFRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int rngSeed, int k, double stdevP, double stdevQ, double stdev, int numIter)
    {
        super(uIndex, iIndex, hasRating, rngSeed);

        this.stdev = stdev;
        this.lambdaP = stdevP / stdev;
        this.lambdaQ = stdevQ / stdev;

        this.numIter = numIter;

        this.k = k;

        this.retrievedData = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);

        //this.P = new DenseDoubleMatrix2D(uIndex.numUsers(), k);
        //this.Q = new DenseDoubleMatrix2D(iIndex.numItems(), k);
    }

    @Override
    public void init()
    {
        super.init();

        // First, we initialize the values.
        this.P = new DenseDoubleMatrix2D(uIndex.numUsers(), k);

        // Then, we initialize the Q matrix with random values.
        this.Q = new DenseDoubleMatrix2D(iIndex.numItems(), k);
        this.Q.assign(x -> Math.sqrt(1.0 / k) + Math.random());

        // Then, we declare standard deviation matrices for the users.
        this.stdevP = new DenseDoubleMatrix2D[uIndex.numUsers()];
        for (int i = 0; i < numUsers(); ++i)
        {
            stdevP[i] = new DenseDoubleMatrix2D(k, k);
            for (int j = 0; j < k; ++j)
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

        this.retrievedData.clear();
        this.set_min_P();
        this.set_min_Q();
    }


    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();
        values.forEach(t -> this.retrievedData.updateRating(t.uidx(), t.iidx(), t.value()));
        if(this.retrievedData.numPreferences() > 0)
        {
            // We finally apply ALS for training the algorithm.
            for (int i = 0; i < this.numIter; ++i)
            {
                set_min_P();
                set_min_Q();
            }
        }
    }

    /*@Override
    public void init(FastPreferenceData<U, I> prefData)
    {
        this.init();
        prefData.getUidxWithPreferences().forEach(uidx -> prefData.getUidxPreferences(uidx).forEach(i -> this.retrievedData.updateRating(uidx, i.v1, i.v2)));
        if(this.retrievedData.numPreferences() > 0)
        {
            // We finally apply ALS for training the algorithm.
            for (int i = 0; i < this.numIter; ++i)
            {
                set_min_P();
                set_min_Q();
            }
        }
    }*/

    /**
     * Fixing the item feature vectors, train the user feature vectors using alternate
     * least squares (ALS).
     */
    protected final void set_min_P()
    {
        // First, find q_i * q_i^t
        DenseDoubleMatrix2D[] A2P = new DenseDoubleMatrix2D[this.numItems()];
        this.retrievedData.getIidxWithPreferences().parallel().forEach(iidx ->
        {
            A2P[iidx] = new DenseDoubleMatrix2D(this.k, this.k);
            DoubleMatrix1D qi = this.Q.viewRow(iidx);
            ALG.multOuter(qi, qi, A2P[iidx]);
        });

        DenseDoubleMatrix2D[] As = new DenseDoubleMatrix2D[this.numUsers()];
        DenseDoubleMatrix1D[] bs = new DenseDoubleMatrix1D[this.numUsers()];

        this.retrievedData.getAllUidx().parallel().forEach(uidx ->
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
           if (this.retrievedData.numItems(uidx) > 0)
           {
               this.retrievedData.getUidxPreferences(uidx).forEach(iv ->
               {
                   int iidx = iv.v1;
                   double rui = iv.v2;

                   A.assign(A2P[iidx], Double::sum);
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

    /**
     * Fixing the user feature vectors, train the item feature vectors using alternate
     * least squares (ALS).
     */
    protected final void set_min_Q()
    {
        // First, find p_u * p_u^t
        DenseDoubleMatrix2D[] A2Q = new DenseDoubleMatrix2D[this.numUsers()];
        retrievedData.getUidxWithPreferences().parallel().forEach(uidx ->
        {
            A2Q[uidx] = new DenseDoubleMatrix2D(k, k);
            DoubleMatrix1D pu = P.viewRow(uidx);
            ALG.multOuter(pu, pu, A2Q[uidx]);
        });

        retrievedData.getAllIidx().parallel().forEach(iidx ->
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
            if (retrievedData.numUsers(iidx) > 0)
            {
                retrievedData.getIidxPreferences(iidx).forEach(iv ->
                {
                    int uidx = iv.v1;
                    double rui = iv.v2;

                    A.assign(A2Q[uidx], Double::sum);
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

    @Override
    public int next(int uidx, IntList available)
    {
        return 0;
    }
}
