package es.uam.eps.ir.knnbandit.recommendation.mf.ictr;

import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.mf.Particle;
import es.uam.eps.ir.knnbandit.recommendation.mf.ictr.particles.ICTRParticle;
import es.uam.eps.ir.knnbandit.recommendation.mf.ictr.particles.ICTRParticleFactory;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.doubles.DoubleList;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Implementation of the Interactive Collaborative Topic Regression (ICTR) model.
 * <p>
 * Wang, Q. et al. Online Interactive Collaborative Filtering Using Multi Armed Bandit with Dependent Arms. IEEE TKDE 31(8)
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public abstract class ICTRRecommender<U, I> extends InteractiveRecommender<U, I>
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
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      Preference data.
     * @param ignoreUnknown True if we must ignore unknown items when updating.
     * @param K             Number of latent factors to use.
     * @param numParticles  Number of particles to use.
     * @param factory       A factory for the particles.
     */
    public ICTRRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreUnknown, int K, int numParticles, ICTRParticleFactory<U, I> factory)
    {
        super(uIndex, iIndex, prefData, ignoreUnknown);
        this.K = K;
        this.numParticles = numParticles;
        this.particleWeight = new DoubleArrayList();
        this.particles = new ArrayList<>();
        this.ictrrng = new Random();
        this.factory = factory;
    }

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      Preference data.
     * @param ignoreUnknown True if we must ignore unknown items when updating.
     * @param notReciprocal True if reciprocal users can be recommended, false otherwise.
     * @param K             Number of latent factors to use.
     * @param numParticles  Number of particles to use.
     * @param factory       A factory for the particles.
     */
    public ICTRRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreUnknown, boolean notReciprocal, int K, int numParticles, ICTRParticleFactory factory)
    {
        super(uIndex, iIndex, prefData, ignoreUnknown, notReciprocal);
        this.K = K;
        this.numParticles = numParticles;
        this.particleWeight = new DoubleArrayList();
        this.particles = new ArrayList<>();
        this.ictrrng = new Random();
        this.factory = factory;
    }

    @Override
    protected void initializeMethod()
    {
        this.particleWeight.clear();
        this.particles.clear();

        // Initialize the different particles.
        for (int i = 0; i < numParticles; ++i)
        {
            ICTRParticle ICTRParticle = factory.create(this.uIndex, this.iIndex, this.K);
            this.particles.add(ICTRParticle);
            this.particleWeight.add(1.0 / (this.numParticles + 0.0));
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

        // Then, for each item:
        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();
        for (int iidx : list)
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

    /**
     * Obtains the estimated reward for a user-item pair.
     *
     * @param uidx the user identifier.
     * @param iidx the item identifier.
     *
     * @return the estimated reward.
     */
    protected abstract double getEstimatedReward(int uidx, int iidx);

    @Override
    public void updateMethod(int uidx, int iidx, double value)
    {
        // Update the methods.
        double sum = 0.0;
        double[] weights = new double[numParticles];

        // First, we estimate the weights using the fitness function.
        for (int i = 0; i < numParticles; ++i)
        {
            // First, we compute the weight of particle i
            double fitness = particles.get(i).getWeight(uidx, iidx, value);
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
            aux.update(uidx, iidx, value);
            // Store it as the new particle.
            defList.add(aux);
        }
    }
}
