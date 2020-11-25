package es.uam.eps.ir.knnbandit.recommendation.bandits.item;

import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.Arrays;

/**
 * Bandit that selects an item proportionally to its popularity.
 * @param <U> type of the users.
 * @param <I> type of the items.
 */
public class MLECategoricalAverageItemBandit<U,I> extends ItemBandit<U,I>
{
    /**
     * A Beta distribution for each possible item.
     */
    private final double[] hits;
    private final double[] misses;
    /**
     * The number of items.
     */
    private final int numItems;

    private final double[] initialAlphas;
    private final double[] initialBetas;
    private final double initialAlpha;
    private final double initialBeta;

    /**
     * Constructor.
     *
     * @param numItems The number of items.
     */
    public MLECategoricalAverageItemBandit(int numItems)
    {
        this.numItems = numItems;
        this.initialAlpha = 1.0;
        this.initialBeta = 1.0;
        this.initialAlphas = null;
        this.initialBetas = null;

        this.hits = new double[numItems];
        this.misses = new double[numItems];
        for(int i = 0; i < numItems; ++i)
        {
            this.hits[i] = this.initialAlpha;
            this.misses[i] = this.initialBeta;
        }
    }

    /**
     * Constructor.
     *
     * @param numItems     Number of items.
     * @param initialAlpha The initial value for the alpha parameter of Beta distributions.
     */
    public MLECategoricalAverageItemBandit(int numItems, double initialAlpha, double initialBeta)
    {
        this.numItems = numItems;
        this.initialAlpha = initialAlpha;
        this.initialBeta = initialBeta;
        this.initialAlphas = null;
        this.initialBetas = null;

        this.hits = new double[numItems];
        this.misses = new double[numItems];
        for(int i = 0; i < numItems; ++i)
        {
            this.hits[i] = this.initialAlpha;
            this.misses[i] = this.initialBeta;
        }
    }

    /**
     * Constructor.
     *
     * @param numItems      Number of items.
     * @param initialAlphas The initial values for the alpha parameters of Beta distributions.
     */
    public MLECategoricalAverageItemBandit(int numItems, double[] initialAlphas, double[] initialBetas)
    {
        this.numItems = numItems;
        this.initialAlpha = 1.0;
        this.initialAlphas = initialAlphas;
        this.initialBeta = 1.0;
        this.initialBetas = initialBetas;

        this.hits = new double[numItems];
        this.misses = new double[numItems];
        for (int i = 0; i < numItems; ++i)
        {
            hits[i] = this.initialAlphas[i];
            misses[i] = this.initialBetas[i];
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
            double availableSum = Arrays.stream(available).mapToDouble(i -> hits[i]/(hits[i]+misses[i])).sum();
            double val = untierng.nextDouble();

            double current = 0.0;
            for (int i : available)
            {
                double value = (hits[i]/(hits[i]+misses[i]))/availableSum;
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
            double availableSum = available.stream().mapToDouble(i -> hits[i]/(misses[i]+hits[i])).sum();
            double val = untierng.nextDouble();

            double current = 0.0;
            for (int i : available)
            {
                double value = (hits[i]/(hits[i]+misses[i]))/availableSum;
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
    public void update(int iidx, double value)
    {
        this.hits[iidx] += value;
        this.misses[iidx] += (1.0-value);
    }

    @Override
    public void reset()
    {
        if (initialAlphas == null || initialBetas == null)
        {
            for (int i = 0; i < numItems; ++i)
            {
                hits[i] = initialAlpha;
                misses[i] = initialBeta;
            }
        }
        else
        {
            for (int i = 0; i < numItems; ++i)
            {
                hits[i] = initialAlphas[i];
                misses[i] = initialBetas[i];
            }
        }
    }
}
