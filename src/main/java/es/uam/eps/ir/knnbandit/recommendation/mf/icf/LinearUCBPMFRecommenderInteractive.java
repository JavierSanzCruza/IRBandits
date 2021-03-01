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
import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Comparator;
import java.util.PriorityQueue;


/**
 * Interactive contact recommendation algorithm based on the combination of probabilistic
 * matrix factorization with multi-armed bandit algorithms for selecting items.
 * <p>
 * Uses UCB as the method for selecting the item to recommend.
 * <p>
 * Zhao, X., Zhang, W., Wang, J. Interactive Collaborative filtering. CIKM 2013.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class LinearUCBPMFRecommenderInteractive<U, I> extends InteractivePMFRecommender<U, I>
{
    private final double alpha;

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
    public LinearUCBPMFRecommenderInteractive(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int k, double stdevP, double stdevQ, double stdev, int numIter, double alpha)
    {
        super(uIndex, iIndex, hasRating, k, stdevP, stdevQ, stdev, numIter);
        this.alpha = alpha;
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
    public LinearUCBPMFRecommenderInteractive(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int rngSeed, int k, double stdevP, double stdevQ, double stdev, int numIter, double alpha)
    {
        super(uIndex, iIndex, hasRating, rngSeed, k, stdevP, stdevQ, stdev, numIter);
        this.alpha = alpha;
    }

    @Override
    public int next(int uidx, IntList availability)
    {
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }

        DoubleMatrix1D pu = this.P.viewRow(uidx);
        DoubleMatrix2D sigmau = this.stdevP[uidx];

        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();
        for (int iidx : availability)
        {
            DoubleMatrix1D qi = this.Q.viewRow(iidx);
            DoubleMatrix1D aux = new DenseDoubleMatrix1D(this.k);

            sigmau.zMult(qi, aux);

            // x_ui = alpha*||q_i||_{2,\Sigma_{u,t}}
            double extra = ALG.mult(qi, aux);

            // score = p_u^t q_i + x_ui
            double val = ALG.mult(pu, qi) + this.alpha * Math.sqrt(extra);

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
    public IntList next(int uidx, IntList availability, int k)
    {
        if (availability == null || availability.isEmpty())
        {
            return new IntArrayList();
        }

        DoubleMatrix1D pu = this.P.viewRow(uidx);
        DoubleMatrix2D sigmau = this.stdevP[uidx];

        IntList top = new IntArrayList();

        int num = Math.min(k, availability.size());
        PriorityQueue<Tuple2id> queue = new PriorityQueue<>(num, Comparator.comparingDouble(x -> x.v2));
        for (int iidx : availability)
        {
            DoubleMatrix1D qi = this.Q.viewRow(iidx);
            DoubleMatrix1D aux = new DenseDoubleMatrix1D(this.k);

            sigmau.zMult(qi, aux);

            // x_ui = alpha*||q_i||_{2,\Sigma_{u,t}}
            double extra = ALG.mult(qi, aux);

            // score = p_u^t q_i + x_ui
            double val = ALG.mult(pu, qi) + this.alpha * Math.sqrt(extra);

            if(queue.size() < num)
            {
                queue.add(new Tuple2id(iidx, val));
            }
            else
            {
                Tuple2id newTuple = new Tuple2id(iidx, val);
                if(queue.comparator().compare(queue.peek(), newTuple) < 0)
                {
                    queue.poll();
                    queue.add(newTuple);
                }
            }

            while(!queue.isEmpty())
            {
                top.add(0, queue.poll().v1);
            }
        }

        return top;
    }

    @Override
    public void fastUpdate(int uidx, int iidx, double value)
    {
        double newValue;
        if(!Double.isNaN(value))
            newValue = value;
        else if(!this.ignoreNotRated)
            newValue = Constants.NOTRATEDNOTIGNORED;
        else
            return;

        DoubleMatrix1D qi = this.Q.viewRow(iidx);
        DenseDoubleMatrix2D aux = new DenseDoubleMatrix2D(this.k, this.k);
        ALG.multOuter(qi, qi, aux);

        // First, update the values for the A and b matrices for user u
        As[uidx].assign(aux, Double::sum);
        bs[uidx].assign(qi, (x, y) -> x + newValue * y);

        // Then, find A^-1 b and A^-1 sigma^2
        LUDecompositionQuick lu = new LUDecompositionQuick(0);
        DenseDoubleMatrix1D c = new DenseDoubleMatrix1D(this.k);
        c.assign(bs[uidx]);

        DoubleMatrix2D extraMatrix = new DenseDoubleMatrix2D(this.k, this.k + 1);
        for (int i = 0; i < k; ++i)
        {
            extraMatrix.setQuick(i, i, this.stdev);
        }
        extraMatrix.viewColumn(k).assign(bs[uidx]);

        lu.decompose(As[uidx]);
        lu.solve(extraMatrix);

        //lu.solve(c);

        this.P.viewRow(uidx).assign(extraMatrix.viewColumn(k));
        this.stdevP[uidx] = ALG.subMatrix(extraMatrix, 0, k - 1, 0, k - 1);

        this.retrievedData.updateRating(uidx, iidx, newValue);
        /*DenseDoubleMatrix2D sigmaI = new DenseDoubleMatrix2D(this.k, this.k);
        for (int i = 0; i < k; ++i)
        {
            sigmaI.setQuick(i, i, this.stdev);
        }
        lu.solve(sigmaI);

        this.P.viewRow(uidx).assign(c);
        this.stdevP[uidx] = sigmaI;*/
    }
}
