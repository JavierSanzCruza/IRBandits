package es.uam.eps.ir.knnbandit.metrics;

import it.unimi.dsi.fastutil.ints.Int2LongMap;
import it.unimi.dsi.fastutil.ints.Int2LongOpenHashMap;
import org.jooq.lambda.tuple.Tuple2;

import java.util.List;

/**
 * Cumulative Expected Popularity Complement (EPC) metric. Finds how popular are the different
 * recommended items.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public class CumulativeEPC<U, I> implements CumulativeMetric<U, I>
{
    /**
     * Number of users.
     */
    private final int numUsers;
    /**
     * Number of items
     */
    private final int numItems;
    /**
     * A map containing the popularity of each item.
     */
    private final Int2LongMap popularities;
    /**
     * Current number of ratings
     */
    private double numRatings;
    /**
     * The EPC main sum.
     */
    private double sum;
    /**
     * The value for EPC for the previous iteration (which is the value that must be returned).
     */
    private double epcValue;

    /**
     * Constructor.
     *
     * @param numUsers the number of users.
     * @param numItems the number of items.
     */
    public CumulativeEPC(int numUsers, int numItems)
    {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.numRatings = 0.0;
        this.popularities = new Int2LongOpenHashMap();
        this.popularities.defaultReturnValue(0L);
        this.epcValue = Double.NaN;
        this.sum = 0.0;
    }

    @Override
    public void initialize(List<Tuple2<Integer, Integer>> train, boolean notReciprocal)
    {

    }

    @Override
    public double compute()
    {
        return epcValue;
    }

    @Override
    public void update(int uidx, int iidx)
    {
        if (numUsers > 0 && numRatings > 0.0)
        {
            this.epcValue = 1 - 1 / (numUsers * numRatings) * sum;
        }

        long pop = this.popularities.getOrDefault(iidx, this.popularities.defaultReturnValue());
        sum += 2 * pop + 1;
        numRatings++;
        this.popularities.put(iidx, pop + 1);
    }

    @Override
    public void reset()
    {
        this.numRatings = 0.0;
        this.popularities.clear();
        this.sum = 0.0;
        this.epcValue = Double.NaN;
    }
}
