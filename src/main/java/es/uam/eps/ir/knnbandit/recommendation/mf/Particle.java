package es.uam.eps.ir.knnbandit.recommendation.mf;

/**
 * Individual particle for reinforcement learning algorithms.
 *
 * @author Javier Sanz-Cruzado
 * @author Pablo Castells
 */
public interface Particle<U, I>
{
    /**
     * Initializes the particle.
     */
    public void initialize();

    /**
     * Updates the particle.
     *
     * @param u     the user.
     * @param i     the item.
     * @param value the value of the interaction between user and item.
     */
    public void update(U u, I i, double value);

    /**
     * Updates the particle.
     *
     * @param uidx  the index of the user.
     * @param iidx  the index of the item.
     * @param value the value of the interaction between user and item.
     */
    public void update(int uidx, int iidx, double value);

    /**
     * Obtains the estimated value of the interaction between user and item.
     *
     * @param u the user.
     * @param i the item.
     *
     * @return the estimated reward
     */
    public double getEstimatedReward(U u, I i);


    public double getEstimatedReward(int uidx, int iidx);

    /**
     * Obtains the weight of the particle.
     *
     * @param u     the user.
     * @param i     the item.
     * @param value the value of the interaction between user and item.
     *
     * @return the weight of the particle.
     */
    public double getWeight(U u, I i, double value);

    public double getWeight(int uidx, int iidx, double value);

    public Particle<U, I> clone();
}
