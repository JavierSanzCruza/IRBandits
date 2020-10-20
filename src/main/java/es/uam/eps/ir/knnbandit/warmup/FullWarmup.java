package es.uam.eps.ir.knnbandit.warmup;

import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;

/**
 * Warm-up that stores all the possible triplets.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class FullWarmup implements Warmup
{
    /**
     * Indicates the viable candidate items for each user.
     */
    private List<IntList> availability;
    /**
     * The complete list of triplets.
     */
    private List<FastRating> fullTraining;
    /**
     * The list of triplets, but removing those not in training.
     */
    private List<FastRating> cleanTraining;

    /**
     * Initializes the values.
     *
     * @param validData     Validation data
     * @param training      Training data
     * @param contactRec    True if we are using this for contact recommendation, false otherwise
     * @param notReciprocal true if, in the case of contact recommendation, we cannot recommend reciprocal links to the ones already in the network.
     */
    public FullWarmup(SimpleFastPreferenceData<?, ?> validData, List<Tuple2<Integer, Integer>> training, boolean contactRec, boolean notReciprocal)
    {
        this.fullTraining = new ArrayList<>();
        this.cleanTraining = new ArrayList<>();

        // Initialize the availability.
        this.availability = new ArrayList<>();
        IntList itemList = new IntArrayList();
        IntStream.range(0, validData.numItems()).forEach(itemList::add);
        if (contactRec)
        {
            IntStream.range(0, validData.numUsers()).forEach(uidx ->
            {
                this.availability.add(new IntArrayList(itemList));
                this.availability.get(uidx).removeInt(itemList.indexOf(uidx));
            });
        }
        else
        {
            IntStream.range(0, validData.numUsers()).forEach(uidx -> this.availability.add(new IntArrayList(itemList)));
        }

        // Then, for each training example, update this:
        training.forEach(tuple ->
        {
            int uidx = tuple.v1;
            int iidx = tuple.v2;
            double value = 0.0;
            if(validData.numUsers(iidx) > 0 && validData.numItems(uidx) > 0)
            {
                Optional<IdxPref> opt = validData.getPreference(uidx, iidx);
                if(opt.isPresent())
                {
                    value = opt.get().v2;
                    this.cleanTraining.add(new FastRating(uidx, iidx, value));
                    if(contactRec && notReciprocal)
                    {
                        int index = this.availability.get(iidx).indexOf(uidx);
                        if(index > 0) // This pair has not been previously recommended.
                        {
                            this.availability.get(iidx).removeInt(this.availability.get(iidx).indexOf(uidx));
                        }
                    }
                }
            }

            this.availability.get(uidx).removeInt(this.availability.get(uidx).indexOf(iidx));
            this.fullTraining.add(new FastRating(uidx, iidx, value));
        });
    }

    public List<IntList> getAvailability()
    {
        return availability;
    }

    public List<FastRating> getFullTraining()
    {
        return fullTraining;
    }

    public List<FastRating> getCleanTraining()
    {
        return cleanTraining;
    }
}
