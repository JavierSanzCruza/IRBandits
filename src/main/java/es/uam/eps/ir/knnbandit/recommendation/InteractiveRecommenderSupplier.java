package es.uam.eps.ir.knnbandit.recommendation;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;

@FunctionalInterface
public interface InteractiveRecommenderSupplier<U,I>
{
    /**
     * Given the user and item data, builds an interactive recommendation algorithm.
     * @param userIndex user index.
     * @param itemIndex item index.
     * @return an interactive recommender.
     */
    InteractiveRecommender<U,I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex);
}
