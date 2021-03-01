package es.uam.eps.ir.knnbandit.recommendation.ensembles.dynamic.optimizers;

import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.NonSequentialSelection;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.basic.Precision;
import es.uam.eps.ir.ranksys.metrics.rel.IdealRelevanceModel;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.function.DoublePredicate;
import java.util.stream.Collectors;

public class PrecisionOptimizer<U,I> implements DynamicOptimizer<U,I>
{
    private Precision<U,I> precision;

    @Override
    public void init(OfflineDataset<U, I> dataset, int cutoff)
    {
        IdealRelevanceModel<U,I> relModel = new IdealRelevanceModel<U, I>()
        {
            @Override
            protected UserIdealRelevanceModel<U, I> get(U u)
            {
                DoublePredicate pred = dataset.getRelevanceChecker();
                return new UserIdealRelevanceModel<U, I>()
                {
                    @Override
                    public Set<I> getRelevantItems()
                    {
                        return dataset.getUserPreferences(u).filter(i -> pred.test(i.v2)).map(i -> i.v1).collect(Collectors.toSet());
                    }

                    @Override
                    public boolean isRelevant(I i)
                    {
                        return dataset.isRelevant(dataset.getPreference(u,i).orElse(0.0));
                    }

                    @Override
                    public double gain(I i)
                    {
                        return dataset.getPreference(u,i).orElse(0.0);
                    }
                };
            }
        };

        // Choose the recommender maximizing the metric.
        this.precision = new Precision<>(cutoff, relModel);
    }

    @Override
    public double evaluate(Recommendation<U, I> rec)
    {
        return precision.evaluate(rec);
    }
}