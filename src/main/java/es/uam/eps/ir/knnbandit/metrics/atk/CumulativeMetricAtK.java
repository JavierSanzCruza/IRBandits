package es.uam.eps.ir.knnbandit.metrics.atk;

import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.utils.Pair;

import java.util.ArrayDeque;
import java.util.Queue;

/**
 * Cumulative metric that analyzes just the last k recommended elements.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public abstract class CumulativeMetricAtK<U, I> implements CumulativeMetric<U, I>
{
    /**
     * Maximum number of recommendations to consider.
     */
    private final int k;
    /**
     * Last k recommendations.
     */
    private final Queue<Pair<Integer>> lastK;

    /**
     * Constructor.
     *
     * @param k number of recommendations to consider.
     */
    public CumulativeMetricAtK(int k)
    {
        this.k = k;
        this.lastK = new ArrayDeque<>();
    }

    @Override
    public void update(int uidx, int iidx)
    {
        if (lastK.size() >= k)
        {
            Pair<Integer> head = lastK.poll();
            this.updateDel(head.v1(), head.v2());
        }

        lastK.add(new Pair<>(uidx, iidx));
        this.updateAdd(uidx, iidx);
    }

    /**
     * Updates the value of the metric for adding the last recommended element.
     *
     * @param uidx the identifier of the target user.
     * @param iidx the identifier of the recommended candidate item.
     */
    protected abstract void updateAdd(int uidx, int iidx);

    /**
     * Updates the value of the metric for removing the oldest recommended element.
     *
     * @param uidx the identifier of the target user.
     * @param iidx the identifier of the recommended candidate item.
     */
    protected abstract void updateDel(int uidx, int iidx);

    @Override
    public void reset()
    {
        this.lastK.clear();
        this.resetMetric();
    }

    /**
     * Resets the specific structures and values of the metric.
     */
    protected abstract void resetMetric();
}
