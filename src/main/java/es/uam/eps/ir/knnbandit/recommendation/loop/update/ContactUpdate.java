package es.uam.eps.ir.knnbandit.recommendation.loop.update;

import es.uam.eps.ir.knnbandit.data.datasets.ContactDataset;
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

public class ContactUpdate<U> implements UpdateStrategy<U,U>
{
    private final boolean notReciprocal;
    private ContactDataset<U> dataset;

    public ContactUpdate(boolean notReciprocal)
    {
        this.notReciprocal = notReciprocal;
    }
    
    public ContactUpdate()
    {
        this.notReciprocal = false;   
    }

    @Override
    public void init(Dataset<U, U> dataset)
    {
        this.dataset = ((ContactDataset<U>) dataset);
    }

    @Override
    public Pair<List<FastRating>> selectUpdate(int uidx, int iidx, Selection<U,U> selection)
    {
        if(selection.isAvailable(uidx, iidx))
        {
            List<FastRating> list = new ArrayList<>();
            List<FastRating> metricList = new ArrayList<>();

            Optional<Double> value = dataset.getPreference(uidx, iidx);
            if (value.isPresent())
            {
                FastRating pair = new FastRating(uidx, iidx, value.get());
                list.add(pair);
                metricList.add(pair);
                if (!dataset.isDirected())
                {
                    pair = new FastRating(iidx, uidx, value.get());
                    list.add(pair);
                }
                else if (this.notReciprocal && selection.isAvailable(iidx, uidx))
                {
                    value = dataset.getPreference(iidx, uidx);
                    pair = new FastRating(iidx, uidx, value.orElse(0.0));
                    list.add(pair);
                }
            }

            return new Pair<>(list, metricList);
        }
        return new Pair<>(new ArrayList<>(), new ArrayList<>());
    }

    @Override
    public List<FastRating> getList(Warmup warmup)
    {
        List<FastRating> list = new ArrayList<>(warmup.getFullTraining());
        for(FastRating rating : list)
        {
            if(!dataset.isDirected())
            {
                list.add(new FastRating(rating.iidx(), rating.uidx(), rating.value()));
            }
            else if(notReciprocal && rating.value() > 0.0)
            {
                list.add(new FastRating(rating.iidx(), rating.uidx(), dataset.getPreference(rating.uidx(), rating.iidx()).orElse(0.0)));
            }
        }

        return list;
    }
}
