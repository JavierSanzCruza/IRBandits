package es.uam.eps.ir.knnbandit;

import java.util.ArrayList;
import java.util.List;

/**
 * Proxy for using the UntieRandomNumberReader.
 * Each time it is updated, selects the next random seed (in cycles).
 */
public class UntieRandomNumberReader
{
    /**
     * The list of random number seeds.
     */
    private final List<Integer> rngSeeds;
    /**
     * A counter to determine which seed to use.
     */
    private int counter;

    /**
     * Constructor.
     */
    public UntieRandomNumberReader()
    {
        this.rngSeeds = new ArrayList<>(UntieRandomNumber.rngSeeds);
        counter = -1;
    }

    /**
     * Gets the current random number generator seed.
     *
     * @return the current random number generator seed.
     */
    public int getRngSeed()
    {
        return this.rngSeeds.get(counter);
    }

    /**
     * Advances to the next random number generator seed.
     *
     * @return the next random number generator seed.
     */
    public int nextSeed()
    {
        counter = (counter + 1) % rngSeeds.size();
        return this.getRngSeed();
    }

}
