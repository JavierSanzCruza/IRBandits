/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf;

import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.mf.Factorizer;

import java.util.stream.Stream;

/**
 * Interactive version of matrix factorization algorithms. Legacy version. If a user-item pair is received
 *  * several times, it takes the first value.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class InteractiveMF<U, I> extends AbstractInteractiveMF<U, I>
{
    /**
     * Constructor.
     *
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param hasRating  True if we must ignore unknown items when updating.
     * @param k          Number of latent factors to use.
     * @param factorizer Factorizer for obtaining the factorized matrices.
     */
    public InteractiveMF(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int k, Factorizer<U, I> factorizer)
    {
        super(uIndex, iIndex, hasRating, k, factorizer, SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex), 100);
    }

    /**
     * Constructor.
     *
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param hasRating  True if we must ignore unknown items when updating.
     * @param k          Number of latent factors to use.
     * @param factorizer Factorizer for obtaining the factorized matrices.
     */
    public InteractiveMF(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int rngSeed, int k, Factorizer<U, I> factorizer)
    {
        super(uIndex, iIndex, hasRating, rngSeed, k, factorizer, SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex), 100);
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
        retrievedData.updateRating(uidx, iidx, newValue);

        if (newValue > 0.0)
        {
            this.currentCounter++;
        }
        if (currentCounter >= this.limitCounter)
        {
            this.currentCounter = 0;
            this.factorization = factorizer.factorize(k, retrievedData);
        }
    }
}
