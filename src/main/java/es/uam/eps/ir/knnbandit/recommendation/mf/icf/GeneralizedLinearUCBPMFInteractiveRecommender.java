/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf.icf;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;

import java.util.stream.Stream;


/**
 * Interactive contact recommendation algorithm based on the combination of probabilistic
 * matrix factorization with multi-armed bandit algorithms for selecting items.
 * <p>
 * Uses a generalized version of UCB as the method for selecting the item to recommend.
 * <p>
 * Zhao, X., Zhang, W., Wang, J. Interactive Collaborative filtering. CIKM 2013.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class GeneralizedLinearUCBPMFInteractiveRecommender<U, I> extends InteractivePMFRecommender<U, I>
{
    private final double alpha;

    private final IntList counters;

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
     * @param alpha     Parameter for indicating the importance of the UCB term.
     */
    public GeneralizedLinearUCBPMFInteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int k, double stdevP, double stdevQ, double stdev, int numIter, double alpha)
    {
        super(uIndex, iIndex, hasRating, k, stdevP, stdevQ, stdev, numIter);
        this.alpha = alpha;
        this.counters = new IntArrayList();
        for (int i = 0; i < uIndex.numUsers(); ++i)
        {
            this.counters.add(1);
        }
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
     * @param alpha     Parameter for indicating the importance of the UCB term.
     */
    public GeneralizedLinearUCBPMFInteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int rngSeed, int k, double stdevP, double stdevQ, double stdev, int numIter, double alpha)
    {
        super(uIndex, iIndex, hasRating, rngSeed, k, stdevP, stdevQ, stdev, numIter);
        this.alpha = alpha;
        this.counters = new IntArrayList();
        for (int i = 0; i < uIndex.numUsers(); ++i)
        {
            this.counters.add(1);
        }
    }

    @Override
    public void init()
    {
        super.init();
        this.counters.clear();
        uIndex.getAllUidx().forEach(uidx -> counters.add(1));
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        super.init(values);
        this.counters.clear();
        uIndex.getAllUidx().forEach(uidx -> counters.add(this.retrievedData.numItems(uidx) + 1));
    }

    /*@Override
    public void init(FastPreferenceData<U,I> trainData)
    {
        super.init(trainData);
        this.counters.clear();
        uIndex.getAllUidx().forEach(uidx -> counters.add(this.retrievedData.numItems(uidx) + 1));
    }*/
    
    @Override
    public int next(int uidx, IntList availability)
    {
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }

        DoubleMatrix1D pu = this.P.viewRow(uidx);
        DoubleMatrix2D sigmau = this.stdevP[uidx];

        double utemp = Math.log(this.counters.get(uidx));
        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();
        for (int iidx : availability)
        {
            DoubleMatrix1D qi = this.Q.viewRow(iidx);
            DoubleMatrix1D aux = new DenseDoubleMatrix1D(this.k);

            sigmau.zMult(qi, aux);

            // x_ui = \sqrt(log t)||q_i||_{2,\Sigma_{u,t}}
            double extra = Math.log(utemp) * ALG.mult(qi, aux);

            // rho(p_u^t q_i) = \frac{1}{1 + e^{- p_u^t q_i}}
            double rho = ALG.mult(pu, qi);
            rho = 1.0 / (1.0 + Math.exp(-rho));

            // score = rho(p_u^t q_i) + x_ui
            double val = rho + this.alpha * Math.sqrt(extra);

            if (Double.isNaN(val))
            {
                val = Double.NEGATIVE_INFINITY;
            }
            if (top.isEmpty() || max < val)
            {
                top = new IntArrayList();
                top.add(iidx);
                max = val;
            }
            else if (max == val)
            {
                top.add(iidx);
            }
        }

        int topSize = top.size();
        if (topSize == 1)
        {
            return top.get(0);
        }
        else
        {
            int idx = rng.nextInt(top.size());
            return top.get(idx);
        }
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        // Update the counter by 1
        this.counters.set(uidx, this.counters.get(uidx) + 1);

        // And then, update the algorithm
        DoubleMatrix1D qi = this.Q.viewRow(iidx);
        DenseDoubleMatrix2D aux = new DenseDoubleMatrix2D(this.k, this.k);
        ALG.multOuter(qi, qi, aux);

        // First, update the values for the A and b matrices for user u
        As[uidx].assign(aux, Double::sum);
        bs[uidx].assign(qi, (x, y) -> x + value * y);

        // Then, find A^-1 b and A^-1 sigma^2
        LUDecompositionQuick lu = new LUDecompositionQuick(0);
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

        this.retrievedData.updateRating(uidx, iidx, value);
    }
}
