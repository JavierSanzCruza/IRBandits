package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.item.ItemBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.item.MLECategoricalItemBandit;
import org.json.JSONObject;

public class MLECategoricalItemBanditConfigurator<U,I> extends AbstractBanditConfigurator<U,I>
{
    private final static String ALPHA = "alpha";

    @Override
    public BanditSupplier<U, I> getBandit(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        return new MLECategoricalItemBanditSupplier<>(alpha);
    }

    private static class MLECategoricalItemBanditSupplier<U,I> implements BanditSupplier<U,I>
    {
        private final double alpha;

        public MLECategoricalItemBanditSupplier(double alpha)
        {
            this.alpha = alpha;
        }

        @Override
        public ItemBandit<U, I> apply(int numItems)
        {
            return new MLECategoricalItemBandit<>(numItems, alpha);
        }

        @Override
        public String getName()
        {
            return ItemBanditIdentifiers.MLEPOP + "-" + alpha;
        }
    }
}
