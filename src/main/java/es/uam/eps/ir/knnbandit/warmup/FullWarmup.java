package es.uam.eps.ir.knnbandit.warmup;

import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class FullWarmup implements Warmup
{
    private List<IntList> availability;
    private List<Tuple2<Integer, Integer>> fullTraining;
    private List<Tuple2<Integer, Integer>> cleanTraining;

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
        this.fullTraining = training;
        // Find only those ratings which exist:
        this.cleanTraining = new ArrayList<>();

        // Initialize the availability list:
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

        training.forEach(tuple ->
                         {
                             int uidx = tuple.v1();
                             int iidx = tuple.v2();
                             if (validData.numUsers(iidx) > 0 && validData.numItems(uidx) > 0 && validData.getPreference(uidx, iidx).isPresent())
                             {
                                 this.cleanTraining.add(new Tuple2<>(uidx, iidx));
                                 if (notReciprocal)
                                 {
                                     // In this case, just update the availability (if possible)
                                     int index = this.availability.get(iidx).indexOf(uidx);
                                     if (index > 0) // This pair has not been previously recommended:
                                     {
                                         this.availability.get(iidx).removeInt(this.availability.get(iidx).indexOf(uidx));
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

    public List<Tuple2<Integer, Integer>> getFullTraining()
    {
        return fullTraining;
    }

    public List<Tuple2<Integer, Integer>> getCleanTraining()
    {
        return cleanTraining;
    }
}
