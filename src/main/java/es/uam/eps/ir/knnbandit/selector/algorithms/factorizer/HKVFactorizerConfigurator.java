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
import es.uam.eps.ir.ranksys.mf.als.HKVFactorizer;
import org.json.JSONObject;

import java.util.function.DoubleUnaryOperator;

/**
 * Configures the original implicit matrix factorization algorithm, developed by Hu, Koren and Volinski.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @see es.uam.eps.ir.ranksys.mf.als.HKVFactorizer
 */
public class HKVFactorizerConfigurator<U,I> extends AbstractFactorizerConfigurator<U,I>
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

    @Override
    public FactorizerSupplier<U, I> getFactorizer(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        double lambda = object.getDouble(LAMBDA);
        int numIter = object.getInt(NUMITER);
        return new HKVFactorizerSupplier(alpha, lambda, numIter);
    }

    /**
     * Class for configuring the HKV factorizer.
     */
    private class HKVFactorizerSupplier implements FactorizerSupplier<U,I>
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
         * Constructor.
         * @param alpha     confidence tuning coefficient.
         * @param lambda    regularization parameter.
         * @param numIter   number of iterations.
         */
        public HKVFactorizerSupplier(double alpha, double lambda, int numIter)
        {
            this.alpha = alpha;
            this.lambda = lambda;
            this.numIter = numIter;
        }

        @Override
        public Factorizer<U, I> apply()
        {
            DoubleUnaryOperator confidence = (double x) -> 1 + alpha * x;
            return new HKVFactorizer<>(lambda, confidence, numIter);
        }

        @Override
        public String getName()
        {
            return FactorizerIdentifiers.IMF + "-" + alpha + "-" + lambda + "-" + numIter;
        }
    }
}
