package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.item.ItemBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.item.ThompsonSamplingItemBandit;
import org.json.JSONObject;

public class ThompsonSamplingConfigurator<U,I> extends AbstractBanditConfigurator<U,I>
{
    private final static String ALPHA = "alpha";
    private final static String BETA = "beta";

    @Override
    public BanditSupplier<U, I> getBandit(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);
        return new ThompsonSamplingBanditSupplier<>(alpha, beta);
    }

    private static class ThompsonSamplingBanditSupplier<U,I> implements BanditSupplier<U,I>
    {
        private final double alpha;
        private final double beta;

        public ThompsonSamplingBanditSupplier(double alpha, double beta)
        {
            this.alpha = alpha;
            this.beta = beta;
        }

        @Override
        public ItemBandit<U, I> apply(int numItems)
        {
            return new ThompsonSamplingItemBandit<>(numItems, alpha, beta);
        }

        @Override
        public String getName()
        {
            return ItemBanditIdentifiers.THOMPSON + "-" + alpha + "-" + beta;
        }
    }
}
