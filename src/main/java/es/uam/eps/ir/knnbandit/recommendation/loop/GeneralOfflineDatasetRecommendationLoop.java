package es.uam.eps.ir.knnbandit.recommendation.loop;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.FastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Interactive recomm
 * @param <U> type of the users
 * @param <I> type of the items.
 */
public class GeneralOfflineDatasetRecommendationLoop<U,I> extends OfflineDatasetRecommendationLoop<U,I>
{
    /**
     * Constructor.
     *
     * @param dataset      the dataset containing all the information.
     * @param recommender  the interactive recommendation algorithm.
     * @param metrics      the set of metrics we want to study.
     * @param endCondition the condition that establishes whether the loop has finished or not.
     */
    public GeneralOfflineDatasetRecommendationLoop(Dataset<U, I> dataset, InteractiveRecommender<U, I> recommender, Map<String, CumulativeMetric<U, I>> metrics, EndCondition endCondition)
    {
        super(dataset, recommender, metrics, endCondition);
    }

    @Override
    public void init()
    {
        this.recommender.init();
        this.metrics.forEach((name, metric)->metric.reset());
        this.userList.clear();
        this.dataset.getPrefData().getUidxWithPreferences().forEach(uidx ->
        {
            userList.add(uidx);
            availability.put(uidx, this.dataset.getIidx().boxed().collect(Collectors.toCollection(IntArrayList::new)));
        });
        this.numUsers = this.userList.size();
        this.endCondition.init();
        this.rng = new Random(rngSeed);
    }

    @Override
    public void init(Warmup warmup)
    {
        // Initialize the recommender data.
        FastUpdateablePreferenceData<U,I> retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), dataset.getUserIndex(), dataset.getItemIndex());

        // Initialize the availability of the items for each user.
        this.userList.clear();
        this.availability.clear();
        List<IntList> availability = warmup.getAvailability();

        this.dataset.getPrefData().getUidxWithPreferences().forEach(uidx ->
        {
            IntList uAvailable = availability.get(uidx);
            if(uAvailable != null && !uAvailable.isEmpty())
            {
                this.userList.add(uidx);
                this.availability.put(uidx, new IntArrayList(uAvailable));
            }
        });

        // Then, the already retrieved ratings.
        warmup.getFullTraining().forEach(tuple ->
        {
            int uidx = tuple.v1;
            int iidx = tuple.v2;
            double value = tuple.v3;
            retrievedData.updateRating(uidx, iidx, value);
        });

        // Initialize the recommender
        this.recommender.init(retrievedData);

        // Initialize the metrics
        this.metrics.forEach((name, metric) -> metric.initialize(warmup.getFullTraining(), false));
        this.numUsers = this.userList.size();
        this.endCondition.init();
        this.rng = new Random(rngSeed);
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

        if(this.dataset.getPrefData().numItems(uidx) > 0 && this.dataset.getPrefData().numUsers(iidx) > 0)
        {
            Optional<IdxPref> realValue = this.dataset.getPrefData().getPreference(uidx, iidx);
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

        // Update the metric values:
        this.metrics.forEach((name, metric) -> metric.update(uidx, iidx, value));
        this.endCondition.update(uidx, iidx, value);
        ++this.iteration;

        return value;
    }


}
