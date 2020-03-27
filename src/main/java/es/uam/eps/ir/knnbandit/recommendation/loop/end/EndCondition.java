package es.uam.eps.ir.knnbandit.recommendation.loop.end;

/**
 * Interface for the classes that check whether a recommendation loop has finished or not.
 */
public interface EndCondition
{
    /**
     * Initializes the condition.
     */
    void init();

    /**
     * Checks whether the loop has ended or not.
     *
     * @return true if the end condition has been met, false otherwise.
     */
    boolean hasEnded();

    /**
     * Updates the condition
     *
     * @param uidx  last recommended user
     * @param iidx  last recommended item
     * @param value the value.
     */
    void update(int uidx, int iidx, double value);
}
