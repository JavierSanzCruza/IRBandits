package es.uam.eps.ir.knnbandit.recommendation;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;

/**
 * Interface for obtaining interactive recommenders.
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface InteractiveRecommenderSupplier<U,I>
{
    /**
     * Given the user and item data, builds an interactive recommendation algorithm.
     * @param userIndex user index.
     * @param itemIndex item index.
     * @param rngSeed a random number generator seed.
     * @return an interactive recommender.
     */
    InteractiveRecommender<U,I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed);
    /**
     * Given the user and item data, builds an interactive recommendation algorithm.
     * @param userIndex user index.
     * @param itemIndex item index.
     * @return an interactive recommender.
     */
    InteractiveRecommender<U,I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex);
    /**
     * Obtains the name of the algorithm.
     * @return the name of the algorithm.
     */
    String getName();
}
