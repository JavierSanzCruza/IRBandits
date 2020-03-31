package es.uam.eps.ir.knnbandit.warmup;

import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Warm-up where we only consider that a (user, item) pair belongs to the warm-up if it is known.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 */
public class OnlyRatingsWarmup implements Warmup
{
    /**
     * The availability data.
     */
    private List<IntList> availability;
    /**
     * The training data.
     */
    private List<Tuple2<Integer, Integer>> training;

    /**
     * Initializes the values.
     *
     * @param preferenceData Validation/Test data
     * @param training       Training data
     * @param contactRec     True if we are using this for contact recommendation, false otherwise
     * @param notReciprocal  true if, in the case of contact recommendation, we cannot recommend reciprocal links to the ones already in the network.
     */
    public OnlyRatingsWarmup(SimpleFastPreferenceData<?, ?> preferenceData, List<Tuple2<Integer, Integer>> training, boolean contactRec, boolean notReciprocal)
    {
        // Find only those ratings which exist:
        this.training = new ArrayList<>();

        // Initialize the availability list:
        this.availability = new ArrayList<>();
        IntList itemList = new IntArrayList();
        IntStream.range(0, preferenceData.numItems()).forEach(itemList::add);
        if (contactRec)
        {
            IntStream.range(0, preferenceData.numUsers()).forEach(uidx ->
            {
                this.availability.add(new IntArrayList(itemList));
                this.availability.get(uidx).removeInt(itemList.indexOf(uidx));
            });
        }
        else
        {
            IntStream.range(0, preferenceData.numUsers()).forEach(uidx -> this.availability.add(new IntArrayList(itemList)));
        }

        training.forEach(tuple ->
        {
            int uidx = tuple.v1();
            int iidx = tuple.v2();
            if (preferenceData.numUsers(iidx) > 0 && preferenceData.numItems(uidx) > 0 && preferenceData.getPreference(uidx, iidx).isPresent())
            {
                this.training.add(new Tuple2<>(uidx, iidx));
                this.availability.get(uidx).removeInt(this.availability.get(uidx).indexOf(iidx));
                if (notReciprocal)
                {
                    // The only case where we might update a link that does not exist is here:
                    // If possible, remove this:
                    int index = this.availability.get(iidx).indexOf(uidx);
                    if (index > 0) // This pair has not been previously recommended:
                    {
                        this.availability.get(iidx).removeInt(this.availability.get(iidx).indexOf(uidx));
                    }
                }
            }
        });
    }

    public List<IntList> getAvailability()
    {
        return availability;
    }

    public List<Tuple2<Integer, Integer>> getFullTraining()
    {
        return training;
    }

    public List<Tuple2<Integer, Integer>> getCleanTraining()
    {
        return training;
    }
}
