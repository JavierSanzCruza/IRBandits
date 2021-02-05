package es.uam.eps.ir.knnbandit.warmup;

import es.uam.eps.ir.knnbandit.data.datasets.ContactDataset;
import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class ContactWarmup extends GeneralWarmup
{
    protected ContactWarmup(List<FastRating> cleanTraining, List<FastRating> fullTraining, List<IntList> availability, int numRel)
    {
        super(cleanTraining, fullTraining, availability, numRel);
    }

    public static ContactWarmup load(ContactDataset<?> dataset, Stream<Pair<Integer>> training, WarmupType type)
    {
        List<FastRating> fullTraining = new ArrayList<>();
        List<FastRating> cleanTraining = new ArrayList<>();
        List<IntList> availability = new ArrayList<>();

        IntList itemList = new IntArrayList();
        IntStream.range(0, dataset.numUsers()).forEach(itemList::add);
        IntStream.range(0, dataset.numUsers()).forEach(uidx ->
        {
            availability.add(new IntArrayList(itemList));
            availability.get(uidx).remove(uidx);
        });

        int numRel = training.mapToInt(t ->
        {
            int uidx = t.v1();
            int iidx = t.v2();
            double value = 0.0;
            Optional<Double> opt = dataset.getPreference(uidx, iidx);
            if(opt.isPresent())
            {
                value = opt.get();
                cleanTraining.add(new FastRating(uidx, iidx, value));
                fullTraining.add(new FastRating(uidx, iidx, value));
                availability.get(uidx).removeInt(availability.get(uidx).indexOf(iidx));

                if(!dataset.isDirected() || !dataset.useReciprocal())
                {
                    int index = availability.get(iidx).indexOf(uidx);
                    if(index > 0)
                    {
                        availability.get(iidx).removeInt(index);
                    }
                }
            }
            else if(type == WarmupType.FULL)
            {
                value = Double.NaN;
                fullTraining.add(new FastRating(uidx, iidx, value));
                availability.get(uidx).removeInt(availability.get(uidx).indexOf(iidx));
            }

            return dataset.isRelevant(value) ? 1 : 0;
        }).sum();

        return new ContactWarmup(fullTraining, cleanTraining, availability, numRel);
    }
}
