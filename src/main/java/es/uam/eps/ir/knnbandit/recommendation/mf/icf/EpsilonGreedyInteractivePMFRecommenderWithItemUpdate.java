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
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.Random;

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
public class EpsilonGreedyInteractivePMFRecommenderWithItemUpdate<U, I> extends EpsilonGreedyInteractivePMFRecommender<U, I>
{
    /**
     * Number of hits before the recommender is updated.
     */
    private final static int LIMITCOUNTER = 100;
    private int currentCounter = 0;

    /**
     * Constructor.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param ignoreNotRated True if we must ignore unknown items when updating.
     * @param k         Number of latent factors to use
     * @param stdevP    Prior standard deviation for the user factors.
     * @param stdevQ    Prior standard deviation for the item factors.
     * @param stdev     Prior standard deviation for the ratings.
     * @param numIter   Number of training iterations.
     * @param epsilon   Probability of recommending an item at random.
     */
    public EpsilonGreedyInteractivePMFRecommenderWithItemUpdate(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int k, double stdevP, double stdevQ, double stdev, int numIter, double epsilon)
    {
        super(uIndex, iIndex, ignoreNotRated, k, stdevP, stdevQ, stdev, numIter, epsilon);
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        double newValue;
        if(!Double.isNaN(value))
            newValue = value;
        else if(!this.ignoreNotRated)
            newValue = Constants.NOTRATEDNOTIGNORED;
        else
            return;

        // Update the user factors:
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

        if (value > 0.0)
        {
            this.currentCounter++;
        }
        if (currentCounter >= LIMITCOUNTER)
        {
            this.currentCounter = 0;

            // Update everything ( a single iteration ):
            this.set_min_Q();
            this.set_min_P();
        }


    }
}
