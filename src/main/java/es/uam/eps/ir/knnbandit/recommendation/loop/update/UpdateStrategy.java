package es.uam.eps.ir.knnbandit.recommendation.loop.update;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.Selection;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.utils.Rating;
import es.uam.eps.ir.knnbandit.warmup.Warmup;

import java.util.List;

public interface UpdateStrategy<U,I>
{
    void init(Dataset<U,I> dataset);
    Pair<List<FastRating>> selectUpdate(int uidx, int iidx, Selection<U,I> selection);
    List<FastRating> getList(Warmup warmup);
}
