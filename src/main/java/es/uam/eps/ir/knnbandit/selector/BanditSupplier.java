package es.uam.eps.ir.knnbandit.selector;

import es.uam.eps.ir.knnbandit.recommendation.bandits.item.ItemBandit;

@FunctionalInterface
public interface BanditSupplier<U,I>
{
    ItemBandit<U,I> apply(int numItems);
}
