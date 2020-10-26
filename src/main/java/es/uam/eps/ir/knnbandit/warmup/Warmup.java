package es.uam.eps.ir.knnbandit.warmup;

import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.List;

/**
 * Interface for storing the warm-up data.
 */
public interface Warmup
{
    int getNumRel();
    /**
     * Gets the full list of training tuples.
     *
     * @return the full list of training tuples, null if the initializer has not been configured.
     */
    List<FastRating> getFullTraining();

    /**
     * Gets the list of training tuples without unknown ratings.
     *
     * @return the full list of training tuples, null if the initializer has not been configured.
     */
    List<FastRating> getCleanTraining();
}
