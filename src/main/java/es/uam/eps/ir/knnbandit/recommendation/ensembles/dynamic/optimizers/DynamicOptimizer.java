package es.uam.eps.ir.knnbandit.recommendation.ensembles.dynamic.optimizers;

import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.ranksys.core.Recommendation;

public interface DynamicOptimizer<U,I>
{
    void init(OfflineDataset<U,I> dataset, int cutoff);
    double evaluate(Recommendation<U,I> rec);
}
