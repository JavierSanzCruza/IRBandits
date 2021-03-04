package es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms;

import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.utils.Pair;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.Arrays;

/**
 * Bandit that selects an item proportionally to its popularity.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class MLECategoricalAverageItemBandit extends AbstractMultiArmedBandit
{
    /**
     * The number of hits of the item.
     */
    private final double[] hits;
    /**
     * The number of failures of the item.
     */
    private final double[] misses;

    /**
     * The values of the initial alphas for each item.
     */
    private final double[] initialAlphas;
    /**
     * The values of the initial betas for each item.
     */
    private final double[] initialBetas;
    /**
     * A unique value for the initial alpha.
     */
    private final double initialAlpha;
    /**
     * A unique value for the initial beta.
     */
    private final double initialBeta;

    /**
     * Constructor.
     *
     * @param numArms The number of arms.
     */
    public MLECategoricalAverageItemBandit(int numArms)
    {
        super(numArms);
        this.initialAlpha = 1.0;
        this.initialBeta = 1.0;
        this.initialAlphas = null;
        this.initialBetas = null;

        this.hits = new double[numArms];
        this.misses = new double[numArms];
        for(int i = 0; i < numArms; ++i)
        {
            this.hits[i] = this.initialAlpha;
            this.misses[i] = this.initialBeta;
        }
    }

    /**
     * Constructor.
     *
     * @param numArms      Number of arms.
     * @param initialAlpha The initial value for the alpha parameter of Beta distributions.
     */
    public MLECategoricalAverageItemBandit(int numArms, double initialAlpha, double initialBeta)
    {
        super(numArms);
        this.initialAlpha = initialAlpha;
        this.initialBeta = initialBeta;
        this.initialAlphas = null;
        this.initialBetas = null;

        this.hits = new double[numArms];
        this.misses = new double[numArms];
        for(int i = 0; i < numArms; ++i)
        {
            this.hits[i] = this.initialAlpha;
            this.misses[i] = this.initialBeta;
        }
    }

    /**
     * Constructor.
     *
     * @param numArms       Number of arms.
     * @param initialAlphas The initial values for the alpha parameters of Beta distributions.
     */
    public MLECategoricalAverageItemBandit(int numArms, double[] initialAlphas, double[] initialBetas)
    {
        super(numArms);
        this.initialAlpha = 1.0;
        this.initialAlphas = initialAlphas;
        this.initialBeta = 1.0;
        this.initialBetas = initialBetas;

        this.hits = new double[numArms];
        this.misses = new double[numArms];
        for (int i = 0; i < numArms; ++i)
        {
            hits[i] = this.initialAlphas[i];
            misses[i] = this.initialBetas[i];
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
            for (int i = 0; i < numArms; ++i)
            {
                hits[i] = initialAlpha;
                misses[i] = initialBeta;
            }
        }
        else
        {
            for (int i = 0; i < numArms; ++i)
            {
                hits[i] = initialAlphas[i];
                misses[i] = initialBetas[i];
            }
        }
    }

    @Override
    public Pair<Integer> getStats(int arm)
    {
        if(arm < 0 || arm >= numArms) return null;

        if(initialAlphas == null || initialBetas == null)
        {
            int numHits = Double.valueOf(this.hits[arm] - initialAlpha).intValue();
            int numMisses = Double.valueOf(this.misses[arm] - initialBeta).intValue();
            return new Pair<>(numHits, numMisses);
        }
        else
        {
            int numHits = Double.valueOf(this.hits[arm] - initialAlphas[arm]).intValue();
            int numMisses = Double.valueOf(this.misses[arm] - initialBetas[arm]).intValue();
            return new Pair<>(numHits, numMisses);
        }
    }
}