package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.AbstractMultiArmedBandit;

public interface BanditSupplier
{
    AbstractMultiArmedBandit apply(int numItems);

    String getName();
}
