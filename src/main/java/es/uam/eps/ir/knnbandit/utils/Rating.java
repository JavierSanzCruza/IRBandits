package es.uam.eps.ir.knnbandit.utils;

public class Rating<U,I>
{
    private final U user;
    private final I item;
    private final double value;

    public Rating(U user, I item, double value)
    {
        this.user = user;
        this.item = item;
        this.value = value;
    }

    public U getUser()
    {
        return user;
    }

    public I getItem()
    {
        return item;
    }

    public double getValue()
    {
        return value;
    }
}
