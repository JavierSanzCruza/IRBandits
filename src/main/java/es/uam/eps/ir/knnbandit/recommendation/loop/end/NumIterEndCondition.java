package es.uam.eps.ir.knnbandit.recommendation.loop.end;

/**
 * End condition that establishes the maximum number of iterations to execute.
 */
public class NumIterEndCondition implements EndCondition
{
    /**
     * The number of iterations.
     */
    private final int numIter;
    /**
     * The current iteration.
     */
    private int actualIter;

    /**
     * Constructor.
     * @param numIter The number of iterations.
     */
    public NumIterEndCondition(int numIter)
    {
        this.numIter = numIter;
        this.init();
    }

    @Override
    public void init()
    {
        actualIter = 0;
    }

    @Override
    public boolean hasEnded()
    {
        return (actualIter >= numIter);
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        actualIter++;
    }
}
