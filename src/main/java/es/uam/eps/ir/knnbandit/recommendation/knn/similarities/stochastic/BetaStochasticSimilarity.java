/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.knn.similarities.stochastic;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
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
    private final Int2ObjectMap<Int2DoubleMap> sims;
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
     * Constructor.
     *
     * @param numUsers Number of users.
     * @param alpha    The alpha parameter (number of successes + 1).
     * @param beta     The beta parameter (number of failures + 1).
     */
    public BetaStochasticSimilarity(int numUsers, double alpha, double beta)
    {
        this.numUsers = numUsers;
        this.sims = new Int2ObjectOpenHashMap<>();
        this.usercount = new double[numUsers];
        this.alpha = alpha;
        this.beta = beta;
        for (int i = 0; i < numUsers; ++i)
        {
            this.usercount[i] = 0.0;
        }
    }

    @Override
    public void initialize()
    {
        IntStream.range(0, this.numUsers).forEach(uidx -> this.usercount[uidx] = 0.0);
        this.sims.clear();
    }

    @Override
    public void initialize(FastPreferenceData<?, ?> trainData)
    {
        this.sims.clear();
        trainData.getAllUidx().forEach(uidx ->
        {
            Int2DoubleOpenHashMap map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.sims.put(uidx, map);

            this.usercount[uidx] = trainData.getUidxPreferences(uidx).filter(iidx -> iidx.v2 > 0.0).mapToDouble(iidx ->
            {
                Int2DoubleOpenHashMap auxMap = new Int2DoubleOpenHashMap();
                auxMap.defaultReturnValue(0.0);

                trainData.getIidxPreferences(iidx.v1).filter(vidx -> vidx.v2 > 0.0).forEach(vidx ->
                    ((Int2DoubleOpenHashMap) this.sims.get(uidx)).addTo(vidx.v1,iidx.v2*vidx.v2));

                return iidx.v2;
            }).sum();
        });
    }

    /**
     * Constructor. Sets alpha and beta to 1.
     *
     * @param numUsers Number of users.
     */
    public BetaStochasticSimilarity(int numUsers)
    {
        this(numUsers, 1.0, 1.0);
    }

    @Override
    public IntToDoubleFunction exactSimilarity(int idx)
    {
        return (int idx2) ->
        {
            if(this.usercount[idx2] == 0.0) return 0.0;
            if(this.sims.containsKey(idx))
            {
                double auxalpha = this.sims.get(idx).getOrDefault(idx2, 0.0) + alpha;
                double auxbeta = this.usercount[idx2] + beta;
                return auxalpha / auxbeta;
            }
            return 0.0;
        };
    }

    @Override
    public Stream<Tuple2id> exactSimilarElems(int idx)
    {
        return this.sims.getOrDefault(idx, new Int2DoubleOpenHashMap()).entrySet().stream().filter(idx2 -> idx != idx2.getKey()).map(idx2 ->
        {
            double auxalpha = idx2.getValue() + alpha;
            double auxbeta = this.usercount[idx2.getKey()] + beta;
            return new Tuple2id(idx2.getKey(), auxalpha / auxbeta);
        });
    }

    @Override
    public void updateNorm(int uidx, double value)
    {
        this.usercount[uidx] += 1;
    }

    @Override
    public void updateNormDel(int uidx, double value)
    {
        this.usercount[uidx] -= 1;
    }

    @Override
    public void update(int uidx, int vidx, int iidx, double uval, double vval)
    {
        if(Double.isNaN(vval) || uval*vval == 0)
        {
            return;
        }

        if(!Double.isNaN(vval) && uval * vval > 1.0)
        {
            if(!this.sims.containsKey(uidx))
            {
                Int2DoubleMap map = new Int2DoubleOpenHashMap();
                map.defaultReturnValue(0.0);
                this.sims.put(uidx, map);
            }

            if(!this.sims.containsKey(vidx))
            {
                Int2DoubleMap map = new Int2DoubleOpenHashMap();
                map.defaultReturnValue(0.0);
                this.sims.put(vidx, map);
            }

            ((Int2DoubleOpenHashMap) this.sims.get(uidx)).addTo(vidx, uval*vval);
            ((Int2DoubleOpenHashMap) this.sims.get(vidx)).addTo(uidx, uval*vval);
        }
    }

    @Override
    public void updateDel(int uidx, int vidx, int iidx, double uval, double vval)
    {
        if(!Double.isNaN(vval) && this.sims.containsKey(uidx) && this.sims.get(uidx).containsKey(vidx))
        {
            ((Int2DoubleOpenHashMap) this.sims.get(uidx)).addTo(vidx, -uval*vval);
            ((Int2DoubleOpenHashMap) this.sims.get(vidx)).addTo(uidx, -uval*vval);

            if(this.sims.get(uidx).get(vidx) == 0.0)
            {
                this.sims.get(uidx).remove(vidx);
                this.sims.get(vidx).remove(uidx);
            }
        }
    }

    @Override
    public IntToDoubleFunction similarity(int idx)
    {
        return (int idx2) ->
        {
            double auxalpha = this.sims.getOrDefault(idx, new Int2DoubleOpenHashMap()).getOrDefault(idx2, 0.0);
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
}
