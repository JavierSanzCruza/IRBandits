package es.uam.eps.ir.knnbandit.recommendation.loop;

import java.util.List;

public class Recommendation<U,I>
{
    private final U user;
    private final List<I> rec;

    public Recommendation(U user, List<I> rec)
    {
        this.user = user;
        this.rec = rec;
    }

    public List<I> getItems()
    {
        return rec;
    }

    public U getUser()
    {
        return user;
    }
}