package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.item.*;
import org.json.JSONObject;

public class UCB1Configurator<U,I> extends AbstractBanditConfigurator<U,I>
{
    @Override
    public BanditSupplier<U, I> getBandit(JSONObject object)
    {
        return new UCB1BanditSupplier<>();
    }

    private static class UCB1BanditSupplier<U,I> implements BanditSupplier<U,I>
    {
        @Override
        public ItemBandit<U, I> apply(int numItems)
        {
            return new UCB1ItemBandit<>(numItems);
        }

        @Override
        public String getName()
        {
            return ItemBanditIdentifiers.UCB1;
        }
    }
}
