/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf.ictr;

import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.mf.Particle;
import es.uam.eps.ir.knnbandit.recommendation.mf.ictr.particles.ICTRParticle;
import es.uam.eps.ir.knnbandit.recommendation.mf.ictr.particles.ICTRParticleFactory;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.doubles.DoubleList;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.*;
import java.util.stream.Stream;

/**
 * Implementation of the Interactive Collaborative Topic Regression (ICTR) model.
 * <p>
 * Wang, Q. et al. Online Interactive Collaborative Filtering Using Multi Armed Bandit with Dependent Arms. IEEE TKDE 31(8)
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class ICTRRecommender<U, I> extends AbstractInteractiveRecommender<U, I>
{
    /**
     * The list of particles.
     */
    protected final List<Particle<U, I>> particles;
    /**
     * The number of latent factors for each user/item.
     */
    private final int K;
    /**
     * The number of particles.
     */
    private final int numParticles;
    /**
     * Weights of the particles.
     */
    private final DoubleList particleWeight;
    /**
     * Particle factory.
     */
    private final ICTRParticleFactory<U, I> factory;

    private final Random ictrrng;

    /**
     * Constructor.
     *
     * @param uIndex       User index.
     * @param iIndex       Item index.
     * @param hasRating    True if we must ignore unknown items when updating.
     * @param K            Number of latent factors to use.
     * @param numParticles Number of particles to use.
     * @param factory      A factory for the particles.
     */
    public ICTRRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int K, int numParticles, ICTRParticleFactory<U, I> factory)
    {
        super(uIndex, iIndex, hasRating);
        this.K = K;
        this.numParticles = numParticles;
        this.particleWeight = new DoubleArrayList();
        this.particles = new ArrayList<>();
        this.ictrrng = new Random();
        this.factory = factory;
    }

    @Override
    public void init()
    {
        super.init();

        this.particleWeight.clear();
        this.particles.clear();

        // Initialize the different particles.
        for (int i = 0; i < numParticles; ++i)
        {
            ICTRParticle<U,I> ICTRParticle = factory.create(this.uIndex, this.iIndex, this.K);
            this.particles.add(ICTRParticle);
            this.particleWeight.add(1.0 / (this.numParticles + 0.0));
        }
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();
        values.forEach(t -> particles.forEach(particle -> particle.update(t.uidx(), t.iidx(),t.value())));
    }

    @Override
    public int next(int uidx, IntList availability)
    {
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }

        // Then, for each item:
        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();
        for (int iidx : availability)
        {
            double val = this.getEstimatedReward(uidx, iidx);

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
            int idx = rng.nextInt(top.size());
            return top.get(idx);
        }
    }

    @Override
    public IntList next(int uidx, IntList availability, int k)
    {
        if (availability == null || availability.isEmpty())
        {
            return new IntArrayList();
        }

        int num = Math.min(k, availability.size());
        // Then, for each item:

        IntList top = new IntArrayList();
        PriorityQueue<Tuple2id> queue = new PriorityQueue<>(num, Comparator.comparingDouble(x -> x.v2));
        for (int iidx : availability)
        {
            double val = this.getEstimatedReward(uidx, iidx);

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

            while(!queue.isEmpty())
            {
                top.add(0, queue.poll().v1);
            }
        }

        return top;
    }

    /**
     * Obtains the estimated reward for a user-item pair.
     *
     * @param uidx the user identifier.
     * @param iidx the item identifier.
     * @return the estimated reward.
     */
    protected abstract double getEstimatedReward(int uidx, int iidx);

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


        // Update the methods.
        double sum = 0.0;
        double[] weights = new double[numParticles];

        // First, we estimate the weights using the fitness function.
        for (int i = 0; i < numParticles; ++i)
        {
            // First, we compute the weight of particle i
            double fitness = particles.get(i).getWeight(uidx, iidx, newValue);
            weights[i] = fitness;
            sum += fitness;
        }

        // Then, we re-sample the particles to obtain the best one.
        List<Particle<U, I>> defList = new ArrayList<>();
        for (int i = 0; i < numParticles; ++i)
        {
            double rnd = ictrrng.nextDouble();
            int idx = 0;
            double w = 0.0;
            while (rnd > w)
            {
                w += weights[idx] / sum;
                ++idx;
            }

            // The re-sampled particle:
            Particle<U, I> aux = this.particles.get(idx - 1).clone();
            // Update the particle.
            aux.update(uidx, iidx, newValue);
            // Store it as the new particle.
            defList.add(aux);
        }
    }
}
