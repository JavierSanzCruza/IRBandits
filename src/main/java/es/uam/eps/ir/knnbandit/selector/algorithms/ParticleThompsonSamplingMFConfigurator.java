package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.knn.item.InteractiveItemBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.ParticleThompsonSamplingMF;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles.PTSMFParticleFactory;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles.PTSMParticleFactories;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import org.json.JSONObject;

import java.util.function.DoublePredicate;

public class ParticleThompsonSamplingMFConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    private static final String SIGMAP = "lambdaP";
    private static final String SIGMAQ = "lambdaQ";
    private static final String STDEV = "stdev";
    private static final String K = "k";
    private static final String NUMP = "numParticles";

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        int k = object.getInt(K);
        double sigmaP = object.getDouble(SIGMAP);
        double sigmaQ = object.getDouble(SIGMAQ);
        double stdev = object.getDouble(STDEV);
        int numP = object.getInt(NUMP);

        return new ParticleThompsonSamplingMFInteractiveRecommenderSupplier(k, sigmaP, sigmaQ, stdev, numP, ignoreUnknown);
    }

    private class ParticleThompsonSamplingMFInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        private final boolean ignoreUnknown;
        private final int k;
        private final double sigmaP;
        private final double sigmaQ;
        private final double stdev;
        private final int numP;

        public ParticleThompsonSamplingMFInteractiveRecommenderSupplier(int k, double sigmaP, double sigmaQ, double stdev, int numP, boolean ignoreUnknown)
        {
            this.ignoreUnknown = ignoreUnknown;
            this.k = k;
            this.sigmaP = sigmaP;
            this.sigmaQ = sigmaQ;
            this.numP = numP;
            this.stdev = stdev;
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            PTSMFParticleFactory<U, I> factory = PTSMParticleFactories.normalFactory(k, stdev, sigmaP, sigmaQ);
            return new ParticleThompsonSamplingMF<>(userIndex, itemIndex, ignoreUnknown, numP, factory);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.PTS + "-" + k + "-" + numP + "-" + sigmaP + "-" + sigmaQ + "-" + stdev + "-"  + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
