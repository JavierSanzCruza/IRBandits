/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf;

import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.mf.Particle;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles.PTSMFParticleFactory;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.*;
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
public class ParticleThompsonSamplingMF<U, I> extends AbstractInteractiveRecommender<U, I>
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

    /**
     * Constructor.
     *
     * @param uIndex       user index.
     * @param iIndex       item index.
     * @param hasRating    true if the algorithm must not be updated when the rating is unknown, false otherwise.
     * @param numParticles the number of particles.
     * @param factory      the particle factory.
     */
    public ParticleThompsonSamplingMF(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int rngSeed, int numParticles, PTSMFParticleFactory<U, I> factory)
    {
        super(uIndex, iIndex, hasRating, rngSeed);
        this.factory = factory;
        this.particleList = new ArrayList<>();
        this.ptsrng = new Random();
        this.numParticles = numParticles;
    }

    @Override
    public void init()
    {
        super.init();

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
    public IntList next(int uidx, IntList availability, int k)
    {
        // First, we obtain the list of available items.
        if (availability == null || availability.isEmpty())
        {
            return new IntArrayList();
        }

        // Then, we randomly select a particle:
        int idx = ptsrng.nextInt(numParticles);
        Particle<U, I> current = particleList.get(idx);

        // Then, using that particle, for each item:
        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();
        int num = Math.min(k, availability.size());
        PriorityQueue<Tuple2id> queue = new PriorityQueue<>(num, Comparator.comparingDouble(x -> x.v2));

        for (int iidx : availability)
        {
            double val = current.getEstimatedReward(uidx, iidx);

            if(queue.size() < num)
            {
                queue.add(new Tuple2id(iidx, val));
            }
            else
            {
                Tuple2id newTuple = new Tuple2id(iidx, val);
                if(queue.comparator().compare(queue.peek(), newTuple) < 0)
                {
                    queue.poll();
                    queue.add(newTuple);
                }
            }
        }

        while(!queue.isEmpty())
        {
            top.add(0, queue.poll().v1);
        }

        return top;
    }


    @Override
    public void fastUpdate(int uidx, int iidx, double value)
    {
        double newValue;
        if(!Double.isNaN(value))
            newValue = value;
        else if(!this.ignoreNotRated)
            newValue = Constants.NOTRATEDNOTIGNORED;
        else
            return;

        // Reweighting: for each particle, we recalculate the weights:
        double[] weights = new double[this.numParticles];
        double sum = 0.0;
        for (int d = 0; d < this.numParticles; ++d)
        {
            double wd = this.particleList.get(d).getWeight(uidx, iidx, newValue);
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
            aux.update(uidx, iidx, newValue);
            // Store it as the new particle.
            defList.add(aux);
        }

        this.particleList.clear();
        this.particleList.addAll(defList);
    }
}
