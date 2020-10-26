package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.item.ItemBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.item.UCB1TunedItemBandit;
import org.json.JSONObject;

public class UCB1TunedConfigurator<U,I> extends AbstractBanditConfigurator<U,I>
{
    @Override
    public BanditSupplier<U, I> getBandit(JSONObject object)
    {
        return new UCB1TunedBanditSupplier<>();
    }

    private static class UCB1TunedBanditSupplier<U,I> implements BanditSupplier<U,I>
    {
        @Override
        public ItemBandit<U, I> apply(int numItems)
        {
            return new UCB1TunedItemBandit<>(numItems);
        }

        @Override
        public String getName()
        {
            return ItemBanditIdentifiers.UCB1TUNED;
        }
    }
}
