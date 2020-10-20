package es.uam.eps.ir.knnbandit.recommendation.loop.update;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.Selection;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.utils.Rating;
import es.uam.eps.ir.knnbandit.warmup.Warmup;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class GeneralUpdate<U,I> implements UpdateStrategy<U,I>
{
    private OfflineDataset<U,I> dataset;

    @Override
    public void init(Dataset<U, I> dataset)
    {
        this.dataset = ((OfflineDataset<U,I>) dataset);
    }

    @Override
    public Pair<List<FastRating>> selectUpdate(int uidx, int iidx, Selection<U,I> selection)
    {
        if(selection.isAvailable(uidx, iidx))
        {
            Optional<Double> value = dataset.getPreference(uidx, iidx);
            List<FastRating> list = new ArrayList<>();
            list.add(new FastRating(uidx, iidx, value.orElse(0.0)));
            return new Pair<>(list, list);
        }
        return new Pair<>(new ArrayList<>(), new ArrayList<>());
    }

    @Override
    public List<FastRating> getList(Warmup warmup)
    {
        return warmup.getFullTraining();
    }
}
