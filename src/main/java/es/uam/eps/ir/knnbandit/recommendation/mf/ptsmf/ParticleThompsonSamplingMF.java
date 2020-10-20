/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.mf.Particle;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles.PTSMFParticleFactory;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Particle Thompson Sampling algorithm.
 * <p>
 * Kawale et al. Efficient Thompson Sampling for Online Matrix-Factorization Recommendation (NIPS 2015)
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ParticleThompsonSamplingMF<U, I> extends InteractiveRecommender<U, I>
{
    /**
     * Number of particles.
     */
    private final int numParticles;
    /**
     * Particle factory.
     */
    private final PTSMFParticleFactory<U, I> factory;
    /**
     * List containing the different particles.
     */
    private final List<Particle<U, I>> particleList;
    /**
     * Random number generator.
     */
    private Random ptsrng;

    /**
     * Constructor.
     *
     * @param uIndex       user index.
     * @param iIndex       item index.
     * @param hasRating    true if the algorithm must not be updated when the rating is unknown, false otherwise.
     * @param numParticles the number of particles.
     * @param factory      the particle factory.
     */
    public ParticleThompsonSamplingMF(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int numParticles, PTSMFParticleFactory<U, I> factory)
    {
        super(uIndex, iIndex, hasRating);
        this.factory = factory;
        this.particleList = new ArrayList<>();
        this.ptsrng = new Random();
        this.numParticles = numParticles;
    }

    @Override
    public void init()
    {
        for(int b = 0; b < numParticles; ++b)
        {
            Particle<U,I> particle = factory.create(uIndex, iIndex);
            particleList.add(particle);
        }
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();
        values.forEach(t ->
        {
           for(Particle<U,I> particle : particleList)
           {
               particle.update(t.uidx(),t.iidx(),t.value());
           }
        });

    }

    @Override
    public int next(int uidx, IntList availability)
    {
        // First, we obtain the list of available items.
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }

        // Then, we randomly select a particle:
        int idx = ptsrng.nextInt(numParticles);
        Particle<U, I> current = particleList.get(idx);

        // Then, using that particle, for each item:
        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();
        for (int iidx : availability)
        {
            double val = current.getEstimatedReward(uidx, iidx);

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
            int iidx = rng.nextInt(top.size());
            return top.get(iidx);
        }
    }


    @Override
    public void update(int uidx, int iidx, double value)
    {
        // Reweighting: for each particle, we recalculate the weights:
        double[] weights = new double[this.numParticles];
        double sum = 0.0;
        for (int d = 0; d < this.numParticles; ++d)
        {
            double wd = this.particleList.get(d).getWeight(uidx, iidx, value);
            sum += wd;
            weights[d] = wd;
        }

        // Then, we re-sample the particles to obtain the best one.
        List<Particle<U, I>> defList = new ArrayList<>();
        for (int i = 0; i < numParticles; ++i)
        {
            double rnd = ptsrng.nextDouble();
            int idx = 0;
            double w = 0.0;
            while (rnd > w)
            {
                w += weights[idx] / sum;
                ++idx;
            }

            // The re-sampled particle:
            Particle<U, I> aux = this.particleList.get(idx - 1).clone();
            // Update the particle.
            aux.update(uidx, iidx, value);
            // Store it as the new particle.
            defList.add(aux);
        }

        this.particleList.clear();
        this.particleList.addAll(defList);
    }
}
