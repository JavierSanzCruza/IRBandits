package es.uam.eps.ir.knnbandit.recommendation.bandits.item;

import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import it.unimi.dsi.fastutil.PriorityQueue;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import it.unimi.dsi.fastutil.objects.ObjectHeapPriorityQueue;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Arrays;

/**
 * Bandit that selects an item proportionally to its popularity.
 * @param <U> type of the users.
 * @param <I> type of the items.
 */
public class MLECategoricalItemBandit<U,I> extends ItemBandit<U,I>
{
    /**
     * A Beta distribution for each possible item.
     */
    private final double[] values;
    /**
     * The sum of the values.
     */
    private double sum;

    /**
     * The number of items.
     */
    private final int numItems;
    /**
     * The initial number of hits for each item.
     */
    private final double[] initialAlphas;
    /**
     * The initial number of hits for all items.
     */
    private final double initialAlpha;

    /**
     * Constructor.
     *
     * @param numItems The number of items.
     */
    public MLECategoricalItemBandit(int numItems)
    {
        this.numItems = numItems;
        this.initialAlpha = 1.0;
        this.initialAlphas = null;

        this.values = new double[numItems];
        for(int i = 0; i < numItems; ++i)
        {
            this.values[i] = this.initialAlpha;
        }
        sum = this.numItems*initialAlpha;
    }

    /**
     * Constructor.
     *
     * @param numItems     Number of items.
     * @param initialAlpha The initial value for the alpha parameter of Beta distributions.
     */
    public MLECategoricalItemBandit(int numItems, double initialAlpha)
    {
        this.numItems = numItems;
        this.initialAlpha = initialAlpha;
        this.initialAlphas = null;

        this.values = new double[numItems];
        for(int i = 0; i < numItems; ++i)
        {
            this.values[i] = this.initialAlpha;
        }
        sum = this.numItems*initialAlpha;
    }

    /**
     * Constructor.
     *
     * @param numItems      Number of items.
     * @param initialAlphas The initial values for the alpha parameters of Beta distributions.
     */
    public MLECategoricalItemBandit(int numItems, double[] initialAlphas)
    {
        this.numItems = numItems;
        this.initialAlpha = 1.0;
        this.initialAlphas = initialAlphas;

        this.values = new double[numItems];
        double sum = 0.0;
        for (int i = 0; i < numItems; ++i)
        {
            values[i] = this.initialAlphas[i];
            sum += this.initialAlphas[i];
        }
    }

    @Override
    public int next(int uidx, int[] available, ValueFunction valF)
    {
        if (available == null || available.length == 0)
        {
            return -1;
        }
        else if (available.length == 1)
        {
            return available[0];
        }
        else
        {
            double availableSum = Arrays.stream(available).mapToDouble(i -> values[i]).sum();
            double val = untierng.nextDouble();

            double current = 0.0;
            for (int i : available)
            {
                double value = values[i]/availableSum;
                if ((current + value) >= val)
                {
                    return i;
                }
                else
                {
                    current += value;
                }
            }

            return available[available.length-1];
        }
    }

    @Override
    public int next(int uidx, IntList available, ValueFunction valF)
    {
        if (available == null || available.size() == 0)
        {
            return -1;
        }
        else if (available.size() == 1)
        {
            return available.get(0);
        }
        else
        {
            double availableSum = available.stream().mapToDouble(i -> values[i]).sum();
            double val = untierng.nextDouble();

            double current = 0.0;
            for (int i : available)
            {
                double value = values[i]/availableSum;
                if ((current + value) >= val)
                {
                    return i;
                }
                else
                {
                    current += value;
                }
            }

            return available.getInt(available.size()-1);
        }
    }

    @Override
    public IntList next(int uidx, IntList available, ValueFunction valFunc, int k)
    {
        IntList avCopy = new IntArrayList();
        available.forEach(avCopy::add);

        IntList list = new IntArrayList();
        int num = Math.min(available.size(), k);
        for(int i = 0; i < num; ++i)
        {
            int elem = this.next(uidx, avCopy, valFunc);
            list.add(elem);
            avCopy.remove(avCopy.indexOf(elem));
        }

        return list;
    }

    @Override
    public void update(int iidx, double value)
    {
        this.values[iidx] += value;
    }

    @Override
    public void reset()
    {
        if (initialAlphas == null)
        {
            for (int i = 0; i < numItems; ++i)
            {
                values[i] = initialAlpha;
            }
        }
        else
        {
            if (numItems >= 0) System.arraycopy(initialAlphas, 0, values, 0, numItems);
        }
    }
}
