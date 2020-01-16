package es.uam.eps.ir.knnbandit.recommendation.loop.end;

public class PercentagePositiveRatingsEndCondition implements EndCondition
{
    private final int numRel;
    private int currentRel;
    private final double threshold;

    public PercentagePositiveRatingsEndCondition(int numRel, double threshold)
    {
        this.numRel = numRel;
        this.threshold = threshold;
        this.init();
    }

    public PercentagePositiveRatingsEndCondition(int totalRel, double percentage, double threshold)
    {
        this.numRel = ((Double) Math.ceil(totalRel*percentage)).intValue();
        this.threshold = threshold;
        this.init();
    }

    @Override
    public void init()
    {
        this.currentRel = 0;
    }

    @Override
    public boolean hasEnded()
    {
        return (this.currentRel >= this.numRel);
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        if(value >= threshold) this.currentRel++;
    }
}
