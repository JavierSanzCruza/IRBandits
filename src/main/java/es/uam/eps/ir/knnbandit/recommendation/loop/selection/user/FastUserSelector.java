package es.uam.eps.ir.knnbandit.recommendation.loop.selection.user;

import java.util.Random;

public abstract class FastUserSelector implements UserSelector
{
    protected final int rngSeed;
    protected Random rng;

    public FastUserSelector(int rngSeed)
    {
        this.rngSeed = rngSeed;
        this.rng = new Random(rngSeed);
    }

    public FastUserSelector()
    {
        this.rngSeed = 0;
        this.rng = new Random(rngSeed);
    }

    @Override
    public void init()
    {
        this.rng = new Random(rngSeed);
    }
}
