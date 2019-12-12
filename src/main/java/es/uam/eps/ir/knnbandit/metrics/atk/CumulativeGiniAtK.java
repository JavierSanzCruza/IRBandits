package es.uam.eps.ir.knnbandit.metrics.atk;

import es.uam.eps.ir.knnbandit.utils.statistics.GiniIndex;
import org.jooq.lambda.tuple.Tuple2;

import java.util.List;

/**
 * Cumulative metric that computes the Gini index of the last k recommended items.
 *
 * @param <U> the type of the users.
 * @param <I> the type of the items.
 */
public class CumulativeGiniAtK<U, I> extends CumulativeMetricAtK<U, I>
{
    /**
     * The updateable Gini index.
     */
    private final GiniIndex gini;

    /**
     * Constructor.
     *
     * @param k        the number of recommendations to consider.
     * @param numItems the number of items in the data.
     */
    public CumulativeGiniAtK(int k, int numItems)
    {
        super(k);
        this.gini = new GiniIndex(numItems);
    }

    @Override
    public void initialize(List<Tuple2<Integer, Integer>> train, boolean notReciprocal)
    {

    }

    @Override
    public double compute()
    {
        return this.gini.getValue();
    }

    @Override
    protected void updateAdd(int uidx, int iidx)
    {
        this.gini.updateFrequency(iidx, 1);
    }

    @Override
    protected void updateDel(int uidx, int iidx)
    {
        this.gini.updateFrequency(iidx, -1);
    }

    @Override
    protected void resetMetric()
    {
        this.gini.reset();
    }
}
