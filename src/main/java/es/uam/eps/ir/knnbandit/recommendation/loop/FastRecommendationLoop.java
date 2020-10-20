package es.uam.eps.ir.knnbandit.recommendation.loop;

import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import org.jooq.lambda.tuple.Tuple3;

/**
 * Interface for fast recommendation loops, relying on indexes instead of identifiers to
 * perform the different operations.
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public interface FastRecommendationLoop<U,I> extends RecommendationLoop<U,I>
{
    /**
     * Executes the complete following iteration of the recommendation loop.
     * @return a triplet indicating: the selected user index, the item index, and the payoff of the recommendation if the algorithm
     * is able to generate a recommendation, null otherwise.
     */
    Pair<Integer> fastNextIteration();
    /**
     * Obtains the result of a recommendation for the recommendation loop.
     * @return a triplet indicating: the selected user index, the item index, and the payoff of the recommendation if the algorithm
     * is able to generate a recommendation, null otherwise.
     */
    Pair<Integer> fastNextRecommendation();

    /**
     * Updates the algorithms and metrics after receiving a metric
     * @param uidx the index identifier of the user.
     * @param iidx the index identifier of the item.
     * @return NaN if the update was not succesful, the value of the recommendation otherwise.
     */
    void fastUpdate (int uidx, int iidx);

}
