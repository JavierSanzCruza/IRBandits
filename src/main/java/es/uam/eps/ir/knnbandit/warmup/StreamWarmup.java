package es.uam.eps.ir.knnbandit.warmup;

import es.uam.eps.ir.knnbandit.data.datasets.StreamDataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class StreamWarmup implements Warmup
{
    private final List<FastRating> fullTraining;
    private final int numRel;

    protected StreamWarmup(List<FastRating> fullTraining, int numRel)
    {
        this.fullTraining = fullTraining;
        this.numRel = numRel;
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
        return fullTraining;
    }

    public StreamWarmup load(StreamDataset<?,?> dataset, List<Pair<Integer>> training) throws IOException
    {
        int numRel = 0;
        List<FastRating> fullTraining = new ArrayList<>();
        dataset.restart();
        for(Pair<Integer> t : training)
        {
            dataset.advance();
            int uidx = t.v1();
            int iidx = t.v2();

            if(dataset.getCurrentUidx() == uidx && dataset.getFeaturedIidx() == iidx)
            {
                double value = dataset.getFeaturedItemRating();
                if(dataset.getRelevanceChecker().test(value)) numRel++;
                fullTraining.add(new FastRating(uidx, iidx, value));
            }
        }

        return new StreamWarmup(fullTraining, numRel);
    }

}
