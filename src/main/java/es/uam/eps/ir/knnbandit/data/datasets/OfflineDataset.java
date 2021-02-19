package es.uam.eps.ir.knnbandit.data.datasets;

import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import org.jooq.lambda.tuple.Tuple2;

import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Representation of a classical offline dataset, where we have a series of ratings
 * in an arbitrary order.
 */
public interface OfflineDataset<U,I> extends Dataset<U,I>
{
    /**
     * Get the number of relevant (user, item) pairs.
     *
     * @return the number of relevant (user, item) pairs.
     */
    int getNumRel();

    /**
     * Given a list of (uidx, iidx) pairs, finds how many relevant pairs there are.
     *
     * @param list the list of (uidx, iidx) pairs.
     * @return the count of how many relevant (uidx, iidx) pairs appear in the list.
     */
    int getNumRel(List<Tuple2<Integer, Integer>> list);

    /**
     * Obtains the set of users with ratings in the dataset.
     * @return an stream containing the users with ratings in the dataset.
     */
    Stream<U> getUsersWithPreferences();
    /**
     * Obtains the set of identifiers of users with ratings in the dataset.
     * @return an stream containing the identifiers users with ratings in the dataset.
     */
    IntStream getUidxWithPreferences();

    /**
     * Obtains the rating for a user/item pair
     * @param u the user.
     * @param i the item.
     * @return an optional value containing the preference value.
     */
    Optional<Double> getPreference(U u, I i);

    /**
     * Obtains the rating for a user/item pair
     * @param uidx the user identifier
     * @param iidx the item identifier.
     * @return an optional value containing the rating.
     */
    Optional<Double> getPreference(int uidx, int iidx);

    /**
     * Given a rating value, indicates whether it is relevant or not.
     * @param value the value.
     * @return true if it is relevant, false otherwise.
     */
    boolean isRelevant(double value);

    /**
     * Obtains the preferences of a user.
     * @param uidx the identifier of the user.
     * @return an stream containing the preferences of the user.
     */
    Stream<IdxPref> getUidxPreferences(int uidx);

    /**
     * Obtains the preferences of a user.
     * @param u the user.
     * @return an stream containing the preferences of the user.
     */
    Stream<? extends IdPref<I>> getUserPreferences(U u);

    /**
     * Obtains the preferences given to a item.
     * @param iidx the identifier of the item.
     * @return an stream containing the preferences given to the item.
     */
    Stream<IdxPref> getIidxPreferences(int iidx);

    /**
     * Obtains the preferences of a user.
     * @param i the item
     * @return an stream containing the preferences of the user.
     */
    Stream<? extends IdPref<U>> getItemPreferences(I i);
}
