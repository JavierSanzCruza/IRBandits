/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.stats;

import java.util.Random;

/**
 * Beta distribution.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class BetaDistribution implements UnivariateStatisticalDistribution
{
    /**
     * Random number generator.
     */
    private final Random rng;
    /**
     * First parameter. In case that this distribution is used as a posterior
     * of a Bernoulli distribution, this value equals to the number of hits - 1.
     */
    private double alpha;
    /**
     * Second parameter. In case that this distribution is used as a posterior
     * of a Bernoulli distribution, this value equals to the number of hits - 1.
     */
    private double beta;

    /**
     * Constructor.
     *
     * @param alpha Initial value of alpha.
     * @param beta  Initial value of beta.
     */
    public BetaDistribution(double alpha, double beta)
    {
        this.alpha = alpha;
        this.beta = beta;
        this.rng = new Random();
    }

    @Override
    public void update(Double... values)
    {
        if (values.length == 2)
        {
            this.update(values[0], values[1]);
        }
    }

    /**
     * Updates the distribution by changing the values of alpha and beta.
     *
     * @param alpha New value for alpha.
     * @param beta  New value for beta.
     */
    public void update(double alpha, double beta)
    {
        this.alpha = alpha;
        this.beta = beta;
    }

    @Override
    public void update(double value, int i)
    {
        switch (i)
        {
            case 0:
                this.updateAlpha(value);
                break;
            case 1:
                this.updateBeta(value);
                break;
        }
    }

    /**
     * Updates the distribution by adding some values to the alpha and the beta.
     *
     * @param incrAlpha Alpha increment.
     * @param incrBeta  Beta increment.
     */
    public void updateAdd(double incrAlpha, double incrBeta)
    {
        this.alpha += incrAlpha;
        this.beta += incrBeta;
    }

    /**
     * Updates the distribution by adding some value to the alpha.
     *
     * @param incr The alpha increment.
     */
    public void updateAddAlpha(double incr)
    {
        this.alpha += incr;
    }

    /**
     * Updates the distribution by adding some value to the beta.
     *
     * @param incr The beta increment.
     */
    public void updateAddBeta(double incr)
    {
        this.beta += incr;
    }

    /**
     * Updates the distribution by changing the value of alpha.
     *
     * @param alpha The new value of alpha.
     */
    public void updateAlpha(double alpha)
    {
        this.alpha = alpha;
    }

    /**
     * Updates the distribution by changing the value of beta.
     *
     * @param beta The new value of beta.
     */
    public void updateBeta(double beta)
    {
        this.beta = beta;
    }

    @Override
    public double mean()
    {
        return alpha / (alpha + beta);
    }

    @Override
    public double getParameter(int i)
    {
        switch (i)
        {
            case 0:
                return alpha;
            case 1:
                return beta;
            default:
                return Double.NaN;
        }
    }

    /**
     * Obtain parameter alpha.
     *
     * @return parameter alpha.
     */
    public double getAlpha()
    {
        return alpha;
    }

    /**
     * Obtain parameter beta.
     *
     * @return parameter beta.
     */
    public double getBeta()
    {
        return beta;
    }

    @Override
    public double sample()
    {
        double a = gammaSample(alpha);
        return a / (a + gammaSample(beta));
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
