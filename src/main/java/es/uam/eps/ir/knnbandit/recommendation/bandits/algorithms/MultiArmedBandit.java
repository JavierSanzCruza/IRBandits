package es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms;

import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.utils.Pair;
import it.unimi.dsi.fastutil.ints.IntList;

/**
 * Interface for defining basic and context-less multi-armed bandits.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 */
public interface MultiArmedBandit
{
    /**
     * Selects the next arm, assuming that a selection of them is available.
     * @param available The selection of arms.
     * @param valF      The value function.
     * @return the next selected arm.
     */
    int next(int[] available, ValueFunction valF);

    /**
     * Selects the next arm, assuming that a selection of them is available.
     * @param available The selection of arms.
     * @param valF      The value function.
     * @return the next selected arm.
     */
    int next(IntList available, ValueFunction valF);

    /**
     * Selects a list of arms, given that the selection of them is available.
     * @param available     The selection of available arms.
     * @param valF          A function that determines the effective value of the arm, given a context.
     * @param k             The number of arms to select.
     * @return a list of selected arms.
     */
    IntList next(IntList available, ValueFunction valF, int k);

    /**
     * Updates the corresponding arm, given the reward.
     * @param arm       the arm to update.
     * @param reward    the reward.
     */
    void update(int arm, double reward);

    /**
     * Restarts the bandit.
     */
    void reset();

    /**
     * Obtains the number of hits/misses of an arm.
     * @param arm   the arm.
     * @return a pair, containing, in the first position, the number of hits, and, in the second, the number of misses
     * for the arm.
     */
    Pair<Integer> getStats(int arm);

    /**
     * Obtains the number of arms in the bandit.
     * @return the number of arms.
     */
    int getNumArms();

}
