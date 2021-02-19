package es.uam.eps.ir.knnbandit.data.datasets.reader;

import java.util.Collection;

public class LogRegister<U,I>
{
    private final U user;
    private final I featuredItem;
    private final double rating;
    private final Collection<I> candidateItems;

    public LogRegister(U user, I featuredItem, double rating, Collection<I> candidateItems)
    {
        this.user = user;
        this.featuredItem = featuredItem;
        this.rating = rating;
        this.candidateItems = candidateItems;
    }

    public U getUser()
    {
        return user;
    }

    public I getFeaturedItem()
    {
        return featuredItem;
    }

    public double getRating()
    {
        return rating;
    }

    public Collection<I> getCandidateItems()
    {
        return candidateItems;
    }
}
