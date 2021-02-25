package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.AbstractMultiArmedBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.MLECategoricalAverageItemBandit;
import org.json.JSONObject;

public class MLECategoricalAverageItemBanditConfigurator<U,I> extends AbstractBanditConfigurator<U,I>
{
    private final static String ALPHA = "alpha";
    private final static String BETA = "beta";

    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);
        return new MLECategoricalItemBanditSupplier(alpha, beta);
    }

    private static class MLECategoricalItemBanditSupplier implements BanditSupplier
    {
        private final double alpha;
        private final double beta;

        public MLECategoricalItemBanditSupplier(double alpha, double beta)
        {
            this.alpha = alpha;
            this.beta = beta;
        }

        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new MLECategoricalAverageItemBandit(numItems, alpha, beta);
        }

        @Override
        public String getName()
        {
            return ItemBanditIdentifiers.MLEPOP + "-" + alpha + "-" + beta;
        }
    }
}
