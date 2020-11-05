package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.ParticleThompsonSamplingMF;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles.PTSMFParticleFactory;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles.PTSMParticleFactories;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

public class BayesianParticleThompsonSamplingMFConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    private static final String SIGMAQ = "sigmaQ";
    private static final String STDEV = "stdev";
    private static final String K = "k";
    private static final String NUMP = "numParticles";
    private static final String ALPHA = "alpha";
    private static final String BETA = "beta";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        int k = object.getInt(K);
        double sigmaQ = object.getDouble(SIGMAQ);
        double stdev = object.getDouble(STDEV);
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);
        int numP = object.getInt(NUMP);

        return new BayesianParticleThompsonSamplingMFInteractiveRecommenderSupplier(k, sigmaQ, alpha, beta, stdev, numP, ignoreUnknown);
    }

    private class BayesianParticleThompsonSamplingMFInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        private final boolean ignoreUnknown;
        private final int k;
        private final double sigmaQ;
        private final double alpha;
        private final double beta;
        private final double stdev;
        private final int numP;

        public BayesianParticleThompsonSamplingMFInteractiveRecommenderSupplier(int k, double sigmaQ, double alpha, double beta, double stdev, int numP, boolean ignoreUnknown)
        {
            this.ignoreUnknown = ignoreUnknown;
            this.k = k;
            this.sigmaQ = sigmaQ;
            this.numP = numP;
            this.alpha = alpha;
            this.beta = beta;
            this.stdev = stdev;
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            PTSMFParticleFactory<U, I> factory = PTSMParticleFactories.bayesianFactory(k, stdev, sigmaQ, alpha, beta);
            return new ParticleThompsonSamplingMF<>(userIndex, itemIndex, ignoreUnknown, numP, factory);
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            PTSMFParticleFactory<U, I> factory = PTSMParticleFactories.bayesianFactory(k, stdev, sigmaQ, alpha, beta);
            return new ParticleThompsonSamplingMF<>(userIndex, itemIndex, ignoreUnknown,  rngSeed, numP, factory);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.BAYESIANPTS + "-" + k + "-" + numP + "-" + sigmaQ + "-" + alpha + "-" + beta + "-" + stdev + "-"  + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
