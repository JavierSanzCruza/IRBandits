package es.uam.eps.ir.knnbandit.recommendation.ensembles.dynamic.optimizers;

import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.basic.NDCG;
import es.uam.eps.ir.ranksys.metrics.basic.Precision;
import es.uam.eps.ir.ranksys.metrics.rel.IdealRelevanceModel;

import java.util.Set;
import java.util.function.DoublePredicate;
import java.util.stream.Collectors;

public class NDCGOptimizer<U,I> implements DynamicOptimizer<U,I>
{
    private NDCG<U,I> ndcg;

    @Override
    public void init(OfflineDataset<U, I> dataset, int cutoff)
    {
        NDCG.NDCGRelevanceModel<U,I> relModel = new NDCG.NDCGRelevanceModel<>(false,dataset.getPreferenceData(), 1.0);
        // Choose the recommender maximizing the metric.
        this.ndcg = new NDCG<>(cutoff, relModel);
    }

    @Override
    public double evaluate(Recommendation<U, I> rec)
    {
        return ndcg.evaluate(rec);
    }
}