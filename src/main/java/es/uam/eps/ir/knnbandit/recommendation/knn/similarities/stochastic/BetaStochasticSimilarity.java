/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.knn.similarities.stochastic;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Random;
import java.util.function.IntToDoubleFunction;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Stochastic similarity that uses a Beta distribution to estimate the similarity.
 *
 * @author Javier Sanz-Cruzado Puig (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class BetaStochasticSimilarity implements StochasticUpdateableSimilarity
{
    /**
     * Current similarities (alpha values)
     */
    private final double[][] sims;
    /**
     * Norms.
     */
    private final double[] usercount;
    /**
     * Number of users.
     */
    private final int numUsers;
    /**
     * Initial alpha.
     */
    private final double alpha;
    /**
     * Initial beta
     */
    private final double beta;
    private final Random rng = new Random(0);
    /**
     * Last visited user.
     */
    private int lastu = -1;
    /**
     * Last visited item.
     */
    private int lasti = -1;


    /**
     * Constructor.
     *
     * @param numUsers Number of users.
     * @param alpha    The alpha parameter (number of successes + 1).
     * @param beta     The beta parameter (number of failures + 1).
     */
    public BetaStochasticSimilarity(int numUsers, double alpha, double beta)
    {
        this.numUsers = numUsers;
        this.sims = new double[numUsers][numUsers];
        this.usercount = new double[numUsers];
        this.alpha = alpha;
        this.beta = beta;
        for (int i = 0; i < numUsers; ++i)
        {
            this.usercount[i] = 0.0;
            for (int j = 0; j < numUsers; ++j)
            {
                this.sims[i][j] = 0.0;
            }
        }
    }

    /**
     * Constructor. Sets alpha and beta to 1.
     *
     * @param numUsers Number of users.
     */
    public BetaStochasticSimilarity(int numUsers)
    {
        this(numUsers, 1, 1);
    }

    @Override
    public IntToDoubleFunction exactSimilarity(int idx)
    {
        return (int idx2) ->
        {
            double auxalpha = this.sims[idx][idx2] + alpha;
            double auxbeta = this.usercount[idx2] + beta;
            return auxalpha / auxbeta;
        };
    }

    @Override
    public Stream<Tuple2id> exactSimilarElems(int idx)
    {
        IntToDoubleFunction sim = this.exactSimilarity(idx);
        return IntStream.range(0, numUsers).filter(i -> i != idx).mapToObj(i -> new Tuple2id(i, sim.applyAsDouble(i))).filter(x -> x.v2 > 0.0);
    }

    @Override
    public void update(int uidx, int vidx, int iidx, double uval, double vval)
    {
        if (!Double.isNaN(vval) && uval * vval > 0)
        {
            sims[uidx][vidx] += 1.0;
            sims[vidx][uidx] += 1.0;
        }

        if (lastu != uidx || lasti != iidx)
        {
            lastu = uidx;
            lasti = iidx;
            if (uval > 0)
            {
                this.usercount[uidx] += 1;
            }
        }
    }

    @Override
    public IntToDoubleFunction similarity(int idx)
    {
        return (int idx2) ->
        {
            double auxalpha = this.sims[idx][idx2];
            double auxbeta = this.usercount[idx2] - auxalpha;
            return this.betaSample(auxalpha + alpha, auxbeta + beta);
        };
    }

    @Override
    public Stream<Tuple2id> similarElems(int idx)
    {
        IntToDoubleFunction sim = this.similarity(idx);
        return IntStream.range(0, numUsers).filter(i -> i != idx).mapToObj(i -> new Tuple2id(i, sim.applyAsDouble(i))).filter(x -> x.v2 > 0.0);
    }

    /**
     * Samples from a Beta distribution.
     *
     * @param alpha The alpha value of the Beta.
     * @param beta  The beta value of the Beta.
     * @return the sampled value.
     */
    public double betaSample(double alpha, double beta)
    {
        double a = this.gammaSample(alpha);
        return a / (a + this.gammaSample(beta));
    }

    /**
     * This implementation was adapted from https://github.com/gesiscss/promoss.
     */
    public double gammaSample(double shape)
    {
        if (shape <= 0) // Not well defined, set to zero and skip
        {
            return 0;
        }
        else if (shape == 1) // Exponential
        {
            return -Math.log(rng.nextDouble());
        }
        else if (shape < 1) // Use Johnks generator
        {
            double c = 1.0 / shape;
            double d = 1.0 / (1 - shape);
            while (true)
            {
                double x = Math.pow(rng.nextDouble(), c);
                double y = x + Math.pow(rng.nextDouble(), d);
                if (y <= 1)
                {
                    return -Math.log(rng.nextDouble()) * x / y;
                }
            }
        }
        else // Bests algorithm
        {
            double b = shape - 1;
            double c = 3 * shape - 0.75;
            while (true)
            {
                double u = rng.nextDouble();
                double v = rng.nextDouble();
                double w = u * (1 - u);
                double y = Math.sqrt(c / w) * (u - 0.5);
                double x = b + y;
                if (x >= 0)
                {
                    double z = 64 * w * w * w * v * v;
                    if ((z <= (1 - 2 * y * y / x))
                            || (Math.log(z) <= 2 * (b * Math.log(x / b) - y)))
                    {
                        return x;
                    }
                }
            }
        }
    }


    @Override
    public void initialize(FastPreferenceData<?, ?> trainData)
    {
        trainData.getAllUidx().forEach(uidx -> trainData.getAllUidx().forEach(vidx -> this.sims[uidx][vidx] = 0.0));

        // First, find the norms.
        trainData.getAllUidx().forEach(uidx -> this.usercount[uidx] = trainData.getUidxPreferences(uidx).filter(iidx -> iidx.v2 > 0.0)
        .mapToDouble(iidx ->
        {
            trainData.getIidxPreferences(iidx.v1).filter(vidx -> vidx.v2 > 0.0).forEach(vidx -> this.sims[uidx][vidx.v1] += 1.0);
            return 1.0;
        }).sum());
    }
}
