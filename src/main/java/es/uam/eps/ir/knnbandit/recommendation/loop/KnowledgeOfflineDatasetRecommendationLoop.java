package es.uam.eps.ir.knnbandit.recommendation.loop;

import es.uam.eps.ir.knnbandit.data.datasets.DatasetWithKnowledge;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;

import java.util.Map;
import java.util.Optional;

/**
 * Interactive recomm
 * @param <U> type of the users
 * @param <I> type of the items.
 */
public class KnowledgeOfflineDatasetRecommendationLoop<U,I> extends GeneralOfflineDatasetRecommendationLoop<U,I>
{
    /**
     * Constructor.
     *
     * @param dataset      the dataset containing all the information.
     * @param recommender  the interactive recommendation algorithm.
     * @param metrics      the set of metrics we want to study.
     * @param endCondition the condition that establishes whether the loop has finished or not.
     */
    public KnowledgeOfflineDatasetRecommendationLoop(DatasetWithKnowledge<U, I> dataset, InteractiveRecommender<U, I> recommender, Map<String, CumulativeMetric<U, I>> metrics, EndCondition endCondition)
    {
        super(dataset, recommender, metrics, endCondition);
    }

    @Override
    public double fastUpdate(int uidx, int iidx)
    {
        // First, we remove the availability of the item.
        int index = this.availability.get(uidx).indexOf(iidx);
        this.availability.get(uidx).removeInt(index);

        if(this.availability.get(uidx).isEmpty())
        {
            int uIndex = this.userList.indexOf(uidx);
            this.userList.remove(uIndex);
            this.availability.remove(uidx);
        }

        // Then, we find the real value.
        double value = 0.0;
        boolean hasRating = false;

        DatasetWithKnowledge<U,I> datasetWK = (DatasetWithKnowledge<U,I>) this.dataset;

        if(datasetWK.getKnownPrefData().numItems(uidx) > 0 && datasetWK.getKnownPrefData().numUsers(iidx) > 0)
        {
            Optional<IdxPref> realValue = datasetWK.getKnownPrefData().getPreference(uidx, iidx);
            if(realValue.isPresent())
            {
                value = realValue.get().v2;
                hasRating = true;
            }
        }

        // If the rating exists, or the recommender uses not existing ratings, update the recommender.
        if(hasRating || this.recommender.usesAll())
        {
            this.recommender.update(uidx, iidx, value);
        }

        double finalValue = value;
        // Update the metric values:
        this.metrics.forEach((name, metric) -> metric.update(uidx, iidx, finalValue));
        this.endCondition.update(uidx, iidx, value);
        ++this.iteration;

        return value;
    }


}
