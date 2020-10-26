package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.item.ItemBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.item.ThompsonSamplingItemBandit;
import org.json.JSONObject;

public class DelayThompsonSamplingConfigurator<U,I> extends AbstractBanditConfigurator<U,I>
{
    private final static String ALPHA = "alpha";
    private final static String BETA = "beta";
    private final static String DELAY = "delay";

    @Override
    public BanditSupplier<U, I> getBandit(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        double beta = object.getDouble(BETA);
        int delay = object.getInt(DELAY);
        return new DelayThompsonSamplingBanditSupplier<>(alpha, beta, delay);
    }

    private static class DelayThompsonSamplingBanditSupplier<U,I> implements BanditSupplier<U,I>
    {
        private final double alpha;
        private final double beta;
        private final int delay;

        public DelayThompsonSamplingBanditSupplier(double alpha, double beta, int delay)
        {
            this.alpha = alpha;
            this.beta = beta;
            this.delay = delay;
        }

        @Override
        public ItemBandit<U, I> apply(int numItems)
        {
            return new ThompsonSamplingItemBandit<>(numItems, alpha, beta);
        }

        @Override
        public String getName()
        {
            return ItemBanditIdentifiers.DELAYTHOMPSON + "-" + alpha + "-" + beta + "-" + delay;
        }
    }
}
