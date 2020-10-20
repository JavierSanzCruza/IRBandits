package es.uam.eps.ir.knnbandit.recommendation.loop.update;

import es.uam.eps.ir.knnbandit.data.datasets.ContactDataset;
import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.StreamDataset;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.Selection;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.Warmup;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class ReplayerUpdate<U,I> implements UpdateStrategy<U,I>
{
    private StreamDataset<U,I> dataset;

    @Override
    public void init(Dataset<U, I> dataset)
    {
        this.dataset = ((StreamDataset<U,I>) dataset);
    }

    @Override
    public Pair<List<FastRating>> selectUpdate(int uidx, int iidx, Selection<U,I> selection)
    {
        List<FastRating> list = new ArrayList<>();
        List<FastRating> metricList = new ArrayList<>();

        if(dataset.getCurrentUidx() == uidx && dataset.getFeaturedIidx() == iidx)
        {
            FastRating rating = new FastRating(uidx, iidx, dataset.getFeaturedItemRating());
            list.add(rating);
            metricList.add(rating);
        }

        return new Pair<>(list, metricList);
    }

    @Override
    public List<FastRating> getList(Warmup warmup)
    {
        return warmup.getFullTraining();
    }
}
