package es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms;

import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.utils.Pair;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.Arrays;

/**
 * Simple bandit-like algorithm that selects an arm proportionally to its popularity
 * i.e. by sampling from a categorical distribution.
 *
 * Bandit that selects an item proportionally to its popularity.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class PopularityMLE extends AbstractMultiArmedBandit
{
    /**
     * The number of hits for each arm.
     */
    private final double[] hits;
    /**
     * The number of misses for each arm.
     */
    private final int[] misses;

    /**
     * The sum of the values.
     */
    private double sum;

    /**
     * Initial alphas for the different items.
     */
    private final double[] initialAlphas;
    /**
     * The initial alpha.
     */
    private final double initialAlpha;

    /**
     * Constructor.
     *
     * @param numArms The number of arms.
     */
    public PopularityMLE(int numArms)
    {
        super(numArms);
        this.initialAlpha = 1.0;
        this.initialAlphas = null;

        this.hits = new double[numArms];
        this.misses = new int[numArms];
        for(int i = 0; i < numArms; ++i)
        {
            this.hits[i] = this.initialAlpha;
            this.misses[i] = 0;
        }
        sum = this.numArms*initialAlpha;
    }

    /**
     * Constructor.
     *
     * @param numArms      Number of arms.
     * @param initialAlpha The initial value for the alpha parameter of Beta distributions.
     */
    public PopularityMLE(int numArms, double initialAlpha)
    {
        super(numArms);
        this.initialAlpha = initialAlpha;
        this.initialAlphas = null;

        this.hits = new double[numArms];
        this.misses = new int[numArms];
        for(int i = 0; i < numArms; ++i)
        {
            this.hits[i] = this.initialAlpha;
            this.misses[i] = 0;

        }
        sum = this.numArms*initialAlpha;
    }

    /**
     * Constructor.
     *
     * @param numArms       Number of arms.
     * @param initialAlphas The initial values for the alpha parameters of Beta distributions.
     */
    public PopularityMLE(int numArms, double[] initialAlphas)
    {
        super(numArms);
        this.initialAlpha = 1.0;
        this.initialAlphas = initialAlphas;

        this.hits = new double[numArms];
        this.misses = new int[numArms];
        for(int i = 0; i < numArms; ++i)
        {
            this.hits[i] = this.initialAlpha;
            this.misses[i] = 0;

        }
    }

    @Override
    public int next(int[] available, ValueFunction valF)
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
            double availableSum = Arrays.stream(available).mapToDouble(i -> hits[i]).sum();
            double val = untierng.nextDouble();

            double current = 0.0;
            for (int i : available)
            {
                double value = hits[i]/availableSum;
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
    public int next(IntList available, ValueFunction valF)
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
            double availableSum = available.stream().mapToDouble(i -> hits[i]).sum();
            double val = untierng.nextDouble();

            double current = 0.0;
            for (int i : available)
            {
                double value = hits[i]/availableSum;
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
    public IntList next(IntList available, ValueFunction valFunc, int k)
    {
        IntList avCopy = new IntArrayList();
        available.forEach(avCopy::add);

        IntList list = new IntArrayList();
        int num = Math.min(available.size(), k);
        for(int i = 0; i < num; ++i)
        {
            int elem = this.next(avCopy, valFunc);
            list.add(elem);
            avCopy.remove(avCopy.indexOf(elem));
        }

        return list;
    }

    @Override
    public void update(int iidx, double value)
    {
        this.hits[iidx] += value;
        this.misses[iidx] += (value == 0.0) ? 1 : 0;
    }

    @Override
    public void reset()
    {
        if (initialAlphas == null)
        {
            for (int i = 0; i < numArms; ++i)
            {
                hits[i] = initialAlpha;
                misses[i] = 0;
            }
        }
        else
        {
            if (numArms >= 0) System.arraycopy(initialAlphas, 0, hits, 0, numArms);
        }
    }

    @Override
    public Pair<Integer> getStats(int arm)
    {
        if(arm < 0 || arm >= numArms) return null;
        int hits;
        if(initialAlphas == null)
        {
            hits = Double.valueOf(this.hits[arm]-initialAlpha).intValue();
        }
        else
        {
            hits = Double.valueOf(this.hits[arm] - initialAlphas[arm]).intValue();
        }

        return new Pair<>(hits, misses[arm]);
    }
}
