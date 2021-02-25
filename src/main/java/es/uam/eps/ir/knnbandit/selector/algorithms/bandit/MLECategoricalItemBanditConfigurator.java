package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.AbstractMultiArmedBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.MLECategoricalItemBandit;
import org.json.JSONObject;

public class MLECategoricalItemBanditConfigurator<U,I> extends AbstractBanditConfigurator<U,I>
{
    private final static String ALPHA = "alpha";

    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        return new MLECategoricalItemBanditSupplier(alpha);
    }

    private static class MLECategoricalItemBanditSupplier implements BanditSupplier
    {
        private final double alpha;

        public MLECategoricalItemBanditSupplier(double alpha)
        {
            this.alpha = alpha;
        }

        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new MLECategoricalItemBandit(numItems, alpha);
        }

        @Override
        public String getName()
        {
            return ItemBanditIdentifiers.MLEPOP + "-" + alpha;
        }
    }
}
