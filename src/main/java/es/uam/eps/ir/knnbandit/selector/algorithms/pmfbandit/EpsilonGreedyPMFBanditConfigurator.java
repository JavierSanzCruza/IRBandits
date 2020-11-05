package es.uam.eps.ir.knnbandit.selector.algorithms.pmfbandit;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.knn.item.InteractiveItemBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.mf.icf.EpsilonGreedyInteractivePMFRecommender;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.PMFBanditIdentifiers;
import es.uam.eps.ir.knnbandit.selector.algorithms.AbstractAlgorithmConfigurator;
import org.json.JSONObject;

public class EpsilonGreedyPMFBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{

    private final static String EPSILON = "epsilon";
    private final int k;
    private final double lambdaP;
    private final double lambdaQ;
    private final double stdev;
    private final int numIter;
    private final boolean ignoreUnknown;

    public EpsilonGreedyPMFBanditConfigurator(int k, double lambdaP, double lambdaQ, double stdev, int numIter, boolean ignoreUnknown)
    {
        this.k = k;
        this.lambdaP = lambdaP;
        this.lambdaQ = lambdaQ;
        this.stdev = stdev;
        this.numIter = numIter;
        this.ignoreUnknown = ignoreUnknown;
    }
    
    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        double epsilon = object.getDouble(EPSILON);
        return new EpsilonGreedyPMFBanditInteractiveRecommenderSupplier(epsilon);
    }

    private class EpsilonGreedyPMFBanditInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        private final double epsilon;

        public EpsilonGreedyPMFBanditInteractiveRecommenderSupplier(double epsilon)
        {
            this.epsilon = epsilon;
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new EpsilonGreedyInteractivePMFRecommender<>(userIndex, itemIndex, ignoreUnknown, k, lambdaP, lambdaQ, stdev, numIter, epsilon);
        }

        @Override
        public InteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new EpsilonGreedyInteractivePMFRecommender<>(userIndex, itemIndex, ignoreUnknown, rngSeed, k, lambdaP, lambdaQ, stdev, numIter, epsilon);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.PMFBANDIT + "-" + k + "-" + lambdaP + "-" + lambdaQ + "-" + stdev + "-" + numIter + "-" + PMFBanditIdentifiers.EGREEDY + "-" + epsilon + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
