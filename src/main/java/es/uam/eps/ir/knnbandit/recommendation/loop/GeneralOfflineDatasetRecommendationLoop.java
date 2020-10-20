package es.uam.eps.ir.knnbandit.recommendation.loop;

import es.uam.eps.ir.knnbandit.data.datasets.GeneralOfflineDataset;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.NonSequentialSelection;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.user.RandomUserSelector;
import es.uam.eps.ir.knnbandit.recommendation.loop.update.GeneralUpdate;

import java.util.Map;

/**
 * Interactive recomm
 * @param <U> type of the users
 * @param <I> type of the items.
 */
public class GeneralOfflineDatasetRecommendationLoop<U,I> extends GenericRecommendationLoop<U,I>
{
    /**
     * Constructor.
     *
     * @param dataset      the dataset containing all the information.
     * @param recommender  the interactive recommendation algorithm.
     * @param metrics      the set of metrics we want to study.
     * @param endCondition the condition that establishes whether the loop has finished or not.
     */
    public GeneralOfflineDatasetRecommendationLoop(GeneralOfflineDataset<U, I> dataset, InteractiveRecommenderSupplier<U, I> recommender, Map<String, CumulativeMetric<U, I>> metrics, EndCondition endCondition, int rngSeed)
    {
        super(dataset, new NonSequentialSelection<>(rngSeed, new RandomUserSelector(rngSeed)), recommender, new GeneralUpdate<>(), endCondition, metrics);
    }
}
