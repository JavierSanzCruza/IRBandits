/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms.factorizer;

import es.uam.eps.ir.ranksys.mf.Factorizer;
import es.uam.eps.ir.ranksys.mf.plsa.PLSAFactorizer;
import org.json.JSONObject;

/**
 * Configures the factorizer for the probabilistic latent semantic analysis algorithm.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.ranksys.mf.als.HKVFactorizer
 */
public class PLSAFactorizerConfigurator<U,I> extends AbstractFactorizerConfigurator<U,I>
{
    /**
     * Number of iterations
     */
    private final static String NUMITER = "numIter";

    @Override
    public FactorizerSupplier<U, I> getFactorizer(JSONObject object)
    {
        int numIter = object.getInt(NUMITER);
        return new PLSAFactorizerSupplier(numIter);
    }

    /**
     * Class for configuring the PLSA factorizer.
     */
    private class PLSAFactorizerSupplier implements FactorizerSupplier<U,I>
    {
        /**
         * Number of iterations.
         */
        private final int numIter;

        /**
         * Constructor.
         * @param numIter number of iterations.
         */
        public PLSAFactorizerSupplier(int numIter)
        {
            this.numIter = numIter;
        }

        @Override
        public Factorizer<U, I> apply()
        {
            return new PLSAFactorizer<>(numIter);
        }

        @Override
        public String getName()
        {
            return FactorizerIdentifiers.PLSA + numIter;
        }
    }
}
