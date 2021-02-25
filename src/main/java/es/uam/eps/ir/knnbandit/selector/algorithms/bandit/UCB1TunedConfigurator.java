package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.AbstractMultiArmedBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.UCB1TunedItemBandit;
import org.json.JSONObject;

public class UCB1TunedConfigurator<U,I> extends AbstractBanditConfigurator<U,I>
{
    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        return new UCB1TunedBanditSupplier<>();
    }

    private static class UCB1TunedBanditSupplier<U,I> implements BanditSupplier
    {
        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new UCB1TunedItemBandit(numItems);
        }

        @Override
        public String getName()
        {
            return ItemBanditIdentifiers.UCB1TUNED;
        }
    }
}
