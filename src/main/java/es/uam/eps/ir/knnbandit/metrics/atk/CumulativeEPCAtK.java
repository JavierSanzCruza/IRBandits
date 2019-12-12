package es.uam.eps.ir.knnbandit.metrics.atk;

import it.unimi.dsi.fastutil.ints.Int2IntMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import org.jooq.lambda.tuple.Tuple2;

import java.util.List;

/**
 * Cumulative version of Expected Popularity Complement at cutoff k
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public class CumulativeEPCAtK<U, I> extends CumulativeMetricAtK<U, I>
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
    private final Int2IntMap popularities;
    /**
     * A map containing the frequency of the items in the top K.
     */
    private final Int2IntMap frequencies;
    /**
     * Current number of ratings
     */
    private double numRatings;
    /**
     * The EPC main sum.
     */
    private double sum;

    /**
     * Constructor.
     *
     * @param k number of recommendations to consider.
     */
    public CumulativeEPCAtK(int k, int numUsers, int numItems)
    {
        super(k);
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.sum = 0.0;
        this.numRatings = k;
        this.popularities = new Int2IntOpenHashMap();
        this.frequencies = new Int2IntOpenHashMap();
        this.popularities.defaultReturnValue(0);
        this.frequencies.defaultReturnValue(0);
    }

    @Override
    public void initialize(List<Tuple2<Integer, Integer>> train, boolean notReciprocal)
    {

    }


    @Override
    protected void updateAdd(int uidx, int iidx)
    {
        int popularity = this.popularities.getOrDefault(iidx, this.popularities.defaultReturnValue());
        sum += popularity;

        this.frequencies.put(iidx, this.frequencies.getOrDefault(iidx, this.frequencies.defaultReturnValue()).intValue());
    }

    @Override
    protected void updateDel(int uidx, int iidx)
    {
        // First, we obtain the frequency:
        int frequency = this.frequencies.getOrDefault(iidx, this.frequencies.defaultReturnValue());
        int popularity = this.popularities.getOrDefault(iidx, this.popularities.defaultReturnValue());
        if (frequency < 1)
        {
            return; // An error ocurred.
        }

        // Then: update the value of the sum.
        sum -= frequency * popularity;
        sum += (frequency - 1) * (popularity + 1);
        // And update the frequency of the item in the last k elements.
        this.frequencies.put(iidx, frequency - 1);
    }

    @Override
    protected void resetMetric()
    {
        this.frequencies.clear();
        this.popularities.clear();
        this.sum = 0.0;
    }

    @Override
    public double compute()
    {
        if (this.numUsers <= 0 || this.numRatings <= 0)
        {
            return Double.NaN;
        }
        return 1 - this.sum / (this.numRatings * this.numUsers);
    }
}
