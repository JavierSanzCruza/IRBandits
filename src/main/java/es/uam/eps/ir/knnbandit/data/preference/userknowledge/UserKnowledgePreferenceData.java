package es.uam.eps.ir.knnbandit.data.preference.userknowledge;

import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;

import java.util.stream.Stream;

/**
 * Interface for storing preference data which includes information about whether
 * the user knew or not the items in the system.
 * @param <U> Type of the user.
 * @param <I> Type of the item.
 */
public interface UserKnowledgePreferenceData<U,I> extends PreferenceData<U,I>
{
    /**
     * Obtains the number of the ratings previously known by the users.
     * @return the number of ratings previously known by the users.
     */
    int numKnown();

    /**
     * Obtains the number of the ratings not previously known by the users.
     * @return the number of ratings not previously known by the users.
     */
    default int numUnknown()
    {
        return this.numPreferences()-numKnown();
    }

    /**
     * Obtains the number of ratings for an item which were previously known by users.
     * @param i the item.
     * @return the number of ratings for item i which were previously known by users.
     */
    int numKnownUsers(I i);

    /**
     * Obtains the number of ratings for an item which were not previously known by users.
     * @param i the item.
     * @return the number of ratings for item i which were not previously known by users.
     */
    default int numUnknownUsers(I i)
    {
        return this.numUsers(i) - this.numKnownUsers(i);
    }

    /**
     * Obtains the number rated items previously known by a user.
     * @param u the user.
     * @return the number rated items previously known by u.
     */
    int numKnownItems(U u);

    /**
     * Obtains the number rated items not previously known by a user.
     * @param u the user.
     * @return the number rated items not previously known by u.
     */
    default int numUnknownItems(U u)
    {
        return this.numItems(u) - this.numKnownItems(u);
    }

    Stream<? extends IdPref<I>> getUserKnownPreferences(U u);
    Stream<? extends IdPref<I>> getUserUnknownPreferences(U u);
    Stream<? extends IdPref<U>> getItemKnownPreferences(I i);
    Stream<? extends IdPref<U>> getItemUnknownPreferences(I i);

    PreferenceData<U,I> getKnownPreferenceData();
    PreferenceData<U,I> getUnknownPreferenceData();
    PreferenceData<U,I> getPreferenceData();
}
