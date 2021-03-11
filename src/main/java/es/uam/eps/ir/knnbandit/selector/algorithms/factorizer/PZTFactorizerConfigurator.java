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

import es.uam.eps.ir.knnbandit.recommendation.mf.PZTFactorizer;
import es.uam.eps.ir.ranksys.mf.Factorizer;
import org.json.JSONObject;

import java.util.function.DoubleUnaryOperator;

/**
 * Configures the fast implicit matrix factorization algorithm.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.mf.PZTFactorizer
 */
public class PZTFactorizerConfigurator<U,I> extends AbstractFactorizerConfigurator<U,I>
{
    /**
     * Identifier for the confidence tuning coefficient.
     */
    private final static String ALPHA = "alpha";
    /**
     * Identifier for the regularization parameter.
     */
    private final static String LAMBDA = "lambda";
    /**
     * Number of iterations.
     */
    private final static String NUMITER = "numIter";
    /**
     * Identifier for selecting whether the algorithm must consider the use of zeroes or not.
     */
    private final static String USEZEROES = "useZeroes";

    @Override
    public FactorizerSupplier<U, I> getFactorizer(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        double lambda = object.getDouble(LAMBDA);
        int numIter = object.getInt(NUMITER);
        boolean useZeroes = object.getBoolean(USEZEROES);
        return new PZTFactorizerSupplier(alpha, lambda, numIter, useZeroes);
    }

    /**
     * Class for configuring the PZT factorizer.
     */
    private class PZTFactorizerSupplier implements FactorizerSupplier<U,I>
    {

        /**
         * Confidence tuning coefficient.
         */
        private final double alpha;
        /**
         * Regularization parameter.
         */
        private final double lambda;
        /**
         * Number of iterations.
         */
        private final int numIter;
        /**
         * True if the algorithm allows ratings equal to zero, false otherwise.
         */
        private final boolean useZeroes;


        /**
         * Constructor.
         * @param alpha     confidence tuning coefficient.
         * @param lambda    regularization parameter.
         * @param numIter   number of iterations.
         * @param useZeroes true if the algorithm allows ratings equal to zero, false otherwise.
         */
        public PZTFactorizerSupplier(double alpha, double lambda, int numIter, boolean useZeroes)
        {
            this.alpha = alpha;
            this.lambda = lambda;
            this.numIter = numIter;
            this.useZeroes = useZeroes;
        }

        @Override
        public Factorizer<U, I> apply()
        {
            DoubleUnaryOperator confidence = (double x) -> 1 + alpha * x;
            return new PZTFactorizer<>(lambda, confidence, numIter, useZeroes);
        }

        @Override
        public String getName()
        {
            return FactorizerIdentifiers.FASTIMF + "-" + alpha + "-" + lambda + "-" + numIter + "-" + (useZeroes ? "true" : "false");
        }
    }
}
