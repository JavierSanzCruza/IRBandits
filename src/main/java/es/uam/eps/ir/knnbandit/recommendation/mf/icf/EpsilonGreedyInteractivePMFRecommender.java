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
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;

import java.util.Random;
import java.util.stream.Stream;

/**
 * Interactive contact recommendation algorithm based on the combination of probabilistic
 * matrix factorization with multi-armed bandit algorithms for selecting items.
 * <p>
 * Uses epsilon-greedy as the method for selecting the item to recommend.
 * <p>
 * Zhao, X., Zhang, W., Wang, J. Interactive Collaborative filtering. CIKM 2013.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class EpsilonGreedyInteractivePMFRecommender<U, I> extends InteractivePMFRecommender<U, I>
{
    /**
     * Probability of selecting an item at random.
     */
    private final double epsilon;

    /**
     * Random number generator.
     */
    private final Random epsrng;

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
     * @param epsilon   Probability of recommending an item at random.
     */
    public EpsilonGreedyInteractivePMFRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int k, double stdevP, double stdevQ, double stdev, int numIter, double epsilon)
    {
        super(uIndex, iIndex, hasRating, k, stdevP, stdevQ, stdev, numIter);
        this.epsilon = epsilon;
        this.epsrng = new Random();
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
     * @param epsilon   Probability of recommending an item at random.
     */
    public EpsilonGreedyInteractivePMFRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int rngSeed, int k, double stdevP, double stdevQ, double stdev, int numIter, double epsilon)
    {
        super(uIndex, iIndex, hasRating, rngSeed, k, stdevP, stdevQ, stdev, numIter);
        this.epsilon = epsilon;
        this.epsrng = new Random();
    }

    @Override
    public int next(int uidx, IntList availability)
    {
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }

        if (epsrng.nextDouble() < this.epsilon) // Explore
        {
            int idx = epsrng.nextInt(availability.size());
            return availability.get(idx);
        }
        else // Exploit
        {
            DoubleMatrix1D pu = this.P.viewRow(uidx);

            double max = Double.NEGATIVE_INFINITY;
            IntList top = new IntArrayList();
            for (int iidx : availability)
            {
                DoubleMatrix1D qi = this.Q.viewRow(iidx);
                // score = p_u^t q_i + x_ui
                double val = ALG.mult(pu, qi);

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
    }

    @Override
    public IntList next(int uidx, IntList available, int k)
    {
        IntList avCopy = new IntArrayList();
        available.forEach(avCopy::add);

        IntList list = new IntArrayList();
        int num = Math.min(available.size(), k);
        for(int i = 0; i < num; ++i)
        {
            int elem = this.next(uidx, avCopy);
            list.add(elem);
            avCopy.remove(avCopy.indexOf(elem));
        }

        return list;
    }

    @Override
    public void fastUpdate(int uidx, int iidx, double value)
    {
        double newValue;
        if(!Double.isNaN(value) && value != Constants.NOTRATEDRATING)
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

        lu.decompose(As[uidx]);
        lu.solve(c);

        this.P.viewRow(uidx).assign(c);

        this.retrievedData.updateRating(uidx, iidx, newValue);
    }


}
