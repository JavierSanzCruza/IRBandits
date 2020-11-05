/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf;

import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.mf.Factorizer;

import java.util.stream.Stream;

/**
 * Interactive version of matrix factorization algorithms. Legacy version.
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
        super(uIndex, iIndex, hasRating, k, factorizer, SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex));
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
        super(uIndex, iIndex, hasRating, rngSeed, k, factorizer, SimpleFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex));
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        retrievedData.updateRating(uidx, iidx, value);

        if (value > 0.0)
        {
            this.currentCounter++;
        }
        if (currentCounter >= LIMITCOUNTER)
        {
            this.currentCounter = 0;
            this.factorization = factorizer.factorize(k, retrievedData);
        }
    }
}
