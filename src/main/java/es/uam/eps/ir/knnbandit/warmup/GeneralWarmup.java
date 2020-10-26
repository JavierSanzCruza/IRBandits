package es.uam.eps.ir.knnbandit.warmup;

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

public class GeneralWarmup implements OfflineWarmup
{

    private final List<FastRating> cleanTraining;
    private final List<FastRating> fullTraining;
    private final List<IntList> availability;
    private final int numRel;

    protected GeneralWarmup(List<FastRating> cleanTraining, List<FastRating> fullTraining, List<IntList> availability, int numRel)
    {
        this.cleanTraining = cleanTraining;
        this.fullTraining = fullTraining;
        this.availability = availability;
        this.numRel = numRel;
    }

    @Override
    public List<IntList> getAvailability()
    {
        return availability;
    }

    @Override
    public int getNumRel()
    {
        return numRel;
    }

    @Override
    public List<FastRating> getFullTraining()
    {
        return fullTraining;
    }

    @Override
    public List<FastRating> getCleanTraining()
    {
        return cleanTraining;
    }

    public static GeneralWarmup load(OfflineDataset<?,?> dataset, Stream<Pair<Integer>> training, WarmupType type)
    {
        List<FastRating> fullTraining = new ArrayList<>();
        List<FastRating> cleanTraining = new ArrayList<>();
        List<IntList> availability = new ArrayList<>();
        IntList itemList = new IntArrayList();
        dataset.getAllIidx().forEach(itemList::add);

        IntStream.range(0, dataset.numUsers()).forEach(uidx -> availability.add(new IntArrayList(itemList)));

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
            }
            else if(type == WarmupType.FULL)
            {
                value = 0.0;
                fullTraining.add(new FastRating(uidx, iidx, value));
                availability.get(uidx).removeInt(availability.get(uidx).indexOf(iidx));
            }

            return dataset.isRelevant(value) ? 1 : 0;
        }).sum();

        return new GeneralWarmup(fullTraining, cleanTraining, availability, numRel);
    }
}
