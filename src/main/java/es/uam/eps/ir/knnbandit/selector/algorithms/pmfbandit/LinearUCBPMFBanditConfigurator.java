package es.uam.eps.ir.knnbandit.selector.algorithms.pmfbandit;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.mf.icf.LinearUCBPMFRecommenderInteractive;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.PMFBanditIdentifiers;
import es.uam.eps.ir.knnbandit.selector.algorithms.AbstractAlgorithmConfigurator;
import org.json.JSONObject;

public class LinearUCBPMFBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{

    private final static String ALPHA = "alpha";
    private final int k;
    private final double lambdaP;
    private final double lambdaQ;
    private final double stdev;
    private final int numIter;
    private final boolean ignoreUnknown;

    public LinearUCBPMFBanditConfigurator(int k, double lambdaP, double lambdaQ, double stdev, int numIter, boolean ignoreUnknown)
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
        double alpha = object.getDouble(ALPHA);
        return new UCBPMFBanditInteractiveRecommenderSupplier(alpha);
    }

    private class UCBPMFBanditInteractiveRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        private final double alpha;

        public UCBPMFBanditInteractiveRecommenderSupplier(double alpha)
        {
            this.alpha = alpha;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new LinearUCBPMFRecommenderInteractive<>(userIndex, itemIndex, ignoreUnknown, k, lambdaP, lambdaQ, stdev, numIter, alpha);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new LinearUCBPMFRecommenderInteractive<>(userIndex, itemIndex, ignoreUnknown, rngSeed, k, lambdaP, lambdaQ, stdev, numIter, alpha);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.PMFBANDIT + "-" + k + "-" + lambdaP + "-" + lambdaQ + "-" + stdev + "-" + numIter + "-" + PMFBanditIdentifiers.UCB + "-" + alpha + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
