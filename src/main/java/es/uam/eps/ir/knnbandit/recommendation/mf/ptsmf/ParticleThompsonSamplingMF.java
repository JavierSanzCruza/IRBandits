package es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.mf.Particle;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles.PTSMFParticleFactory;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Particle Thompson Sampling algorithm.
 *
 * Kawale et al. Efficient Thompson Sampling for Online Matrix-Factorization Recommendation
 * @param <U> Type of the users.
 * @param <I> Type of the items.
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
     * @param uIndex user index.
     * @param iIndex item index.
     * @param prefData preference data.
     * @param hasRating true if the algorithm must not be updated when the rating is unknown, false otherwise.
     * @param numParticles the number of particles.
     * @param factory the particle factory.
     */
    public ParticleThompsonSamplingMF(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRating, int numParticles, PTSMFParticleFactory<U, I> factory)
    {
        super(uIndex, iIndex, prefData, hasRating);
        this.factory = factory;
        this.particleList = new ArrayList<>();
        this.ptsrng = new Random();
        this.numParticles = numParticles;
    }

    /**
     * Constructor.
     * @param uIndex user index.
     * @param iIndex item index.
     * @param prefData preference data.
     * @param hasRating true if the algorithm must not be updated when the rating is unknown, false otherwise.
     * @param notReciprocal true if reciprocal relations are not recommended, false otherwise.
     * @param numParticles the number of particles.
     * @param factory the particle factory.
     */
    public ParticleThompsonSamplingMF(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRating, boolean notReciprocal, int numParticles, PTSMFParticleFactory<U, I> factory)
    {
        super(uIndex, iIndex, prefData, hasRating, notReciprocal);
        this.factory = factory;
        this.particleList = new ArrayList<>();
        this.ptsrng = new Random();
        this.numParticles = numParticles;
    }

    @Override
    protected void initializeMethod()
    {
        for (int b = 0; b < numParticles; ++b)
        {
            Particle<U,I> particle = factory.create(this.uIndex, this.iIndex);
            this.uIndex.getAllUidx().forEach(uidx ->
                this.trainData.getUidxPreferences(uidx).forEach(pref ->
                {
                    int iidx = pref.v1;
                    double val = pref.v2;
                    particle.update(uidx, iidx, val);
                }));
            particleList.add(particle);
        }
    }

    @Override
    public int next(int uidx)
    {
        // First, we obtain the list of available items.
        IntList list = this.availability.get(uidx);
        if (list == null || list.isEmpty())
        {
            return -1;
        }

        // Then, we randomly select a particle:
        int idx = ptsrng.nextInt(numParticles);
        Particle<U,I> current = particleList.get(idx);

        // Then, using that particle, for each item:
        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();
        for (int iidx : list)
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
    public void updateMethod(int uidx, int iidx, double value)
    {
        // First, estimate the weights.
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
