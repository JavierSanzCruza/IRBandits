package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.*;
import org.json.JSONObject;

public class UCB1Configurator<U,I> extends AbstractBanditConfigurator<U,I>
{
    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        return new UCB1BanditSupplier<>();
    }

    private static class UCB1BanditSupplier<U,I> implements BanditSupplier
    {
        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new UCB1ItemBandit(numItems);
        }

        @Override
        public String getName()
        {
            return ItemBanditIdentifiers.UCB1;
        }
    }
}
