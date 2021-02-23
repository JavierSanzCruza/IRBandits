package es.uam.eps.ir.knnbandit.recommendation.loop;

import it.unimi.dsi.fastutil.ints.IntList;

public class FastRecommendation
{
    private final int uidx;
    private final IntList rec;

    public FastRecommendation(int uidx, IntList rec)
    {
        this.uidx = uidx;
        this.rec = rec;
    }

    public IntList getIidxs()
    {
        return rec;
    }

    public int getUidx()
    {
        return uidx;
    }
}