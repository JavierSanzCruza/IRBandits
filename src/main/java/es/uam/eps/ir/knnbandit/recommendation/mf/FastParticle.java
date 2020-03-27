package es.uam.eps.ir.knnbandit.recommendation.mf;

import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;

/**
 * Fast implementation of a particle.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 * @author Javier Sanz-Cruzado
 */
public abstract class FastParticle<U, I> implements Particle<U, I>
{
    /**
     * User index.
     */
    private final FastUserIndex<U> uIndex;
    /**
     * Item index.
     */
    private final FastItemIndex<I> iIndex;


    protected int numUsers;
    protected int numItems;

    /**
     * Constructor.
     */
    public FastParticle(FastUserIndex<U> uIndex, FastItemIndex<I> iIndex)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;

        this.numUsers = uIndex.numUsers();
        this.numItems = iIndex.numItems();
    }

    @Override
    public void update(U u, I i, double value)
    {
        this.update(this.uIndex.user2uidx(u), this.iIndex.item2iidx(i), value);
    }

    @Override
    public double getEstimatedReward(U u, I i)
    {
        return this.getEstimatedReward(this.uIndex.user2uidx(u), this.iIndex.item2iidx(i));
    }

    @Override
    public abstract Particle<U, I> clone();

    @Override
    public double getWeight(U u, I i, double value)
    {
        return this.getWeight(this.uIndex.user2uidx(u), this.iIndex.item2iidx(i), value);
    }


    protected FastUserIndex<U> getUserIndex()
    {
        return this.uIndex;
    }

    protected FastItemIndex<I> getItemIndex()
    {
        return this.iIndex;
    }
}
