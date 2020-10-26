package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.item.ItemBandit;

public interface BanditSupplier<U,I>
{
    ItemBandit<U,I> apply(int numItems);

    String getName();
}
