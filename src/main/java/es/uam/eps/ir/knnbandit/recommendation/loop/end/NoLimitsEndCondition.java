package es.uam.eps.ir.knnbandit.recommendation.loop.end;

/**
 * End condition specifying that the loop has no end.
 */
public class NoLimitsEndCondition implements EndCondition
{
    @Override
    public void init()
    {

    }

    @Override
    public boolean hasEnded()
    {
        return false;
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {

    }
}
