package es.uam.eps.ir.knnbandit.warmup;

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
    private List<Tuple3<Integer, Integer, Double>> training;

    /**
     * Initializes the values.
     *
     * @param validData Validation/Test data
     * @param training       Training data
     * @param contactRec     True if we are using this for contact recommendation, false otherwise
     * @param notReciprocal  true if, in the case of contact recommendation, we cannot recommend reciprocal links to the ones already in the network.
     */
    public OnlyRatingsWarmup(SimpleFastPreferenceData<?, ?> validData, List<Tuple2<Integer, Integer>> training, boolean contactRec, boolean notReciprocal)
    {
        this.training = new ArrayList<>();

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
                    this.training.add(new Tuple3<>(uidx, iidx, value));
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
        });
    }

    public List<IntList> getAvailability()
    {
        return availability;
    }

    public List<Tuple3<Integer, Integer, Double>> getFullTraining()
    {
        return training;
    }

    public List<Tuple3<Integer, Integer, Double>> getCleanTraining()
    {
        return training;
    }
}
