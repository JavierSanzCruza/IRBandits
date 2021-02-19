package es.uam.eps.ir.knnbandit.metrics;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;

import java.util.List;

public class CumulativeCounter<U,I> implements CumulativeMetric<U, I>
{
    private double counter;
    @Override
    public double compute()
    {
        return counter;
    }

    @Override
    public void initialize(Dataset<U, I> dataset)
    {
        counter = 0;
    }

    @Override
    public void initialize(Dataset<U, I> dataset, List<FastRating> train)
    {
        counter = 0;
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        counter++;
    }

    @Override
    public void reset()
    {
        counter = 0;
    }
}
