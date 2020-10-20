package es.uam.eps.ir.knnbandit.utils;

public class FastRating
{
    private final int user;
    private final int item;
    private final double value;

    public FastRating(int user, int item, double value)
    {
        this.user = user;
        this.item = item;
        this.value = value;
    }

    public int uidx()
    {
        return user;
    }

    public int iidx()
    {
        return item;
    }

    public double value()
    {
        return value;
    }

    @Override
    public boolean equals(Object obj)
    {
        if (obj == this) {
            return true;
        }
        if (obj == null || obj.getClass() != this.getClass()) {
            return false;
        }

        FastRating rating = (FastRating) obj;
        return this.user == rating.user && this.item == rating.item;
    }
}
