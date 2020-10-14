package es.uam.eps.ir.knnbandit.recommendation.loop;

import es.uam.eps.ir.knnbandit.data.datasets.ContactDataset;
import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
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
 */
public class ContactOfflineDatasetRecommendationLoop<U> extends OfflineDatasetRecommendationLoop<U,U>
{
    /**
     * Variable that indicates, when it is true, that we cannot recommend reciprocal links, and false otherwise.
     */
    private final boolean notReciprocal;

    /**
     * Constructor.
     *
     * @param dataset      the dataset containing all the information.
     * @param recommender  the interactive recommendation algorithm.
     * @param metrics      the set of metrics we want to study.
     * @param endCondition the condition that establishes whether the loop has finished or not.
     */
    public ContactOfflineDatasetRecommendationLoop(ContactDataset<U> dataset, InteractiveRecommender<U, U> recommender, Map<String, CumulativeMetric<U, U>> metrics, EndCondition endCondition, boolean notReciprocal)
    {
        super(dataset, recommender, metrics, endCondition);
        this.notReciprocal = notReciprocal;
    }

    @Override
    public void init()
    {
        // Initialize the recommender data
        this.recommender.init(true);
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), dataset.getUserIndex(), dataset.getItemIndex());

        // Initialize the metrics
        this.metrics.forEach((name, metric)->metric.reset());
        this.userList.clear();

        // Initialize the available data.
        this.dataset.getPrefData().getUidxWithPreferences().forEach(uidx ->
        {
            userList.add(uidx);
            availability.put(uidx, this.dataset.getIidx().filter(iidx -> uidx != iidx).boxed().collect(Collectors.toCollection(IntArrayList::new)));
        });
        this.numUsers = this.userList.size();

        // Initialize other elements.
        this.endCondition.init();
        this.rng = new Random(rngSeed);
    }

    @Override
    public void init(Warmup warmup)
    {
        // Initialize the recommender data.
        this.recommender.init(warmup, false);
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), dataset.getUserIndex(), dataset.getItemIndex());

        // Initialize the availability of the items for each user.
        this.userList.clear();
        this.availability.clear();
        List<IntList> availability = warmup.getAvailability();

        // Then, initialize all the possible values associated to the warmup
        this.dataset.getPrefData().getUidxWithPreferences().forEach(uidx ->
        {
            // First, the availability.
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
            int vidx = tuple.v2;

            if(dataset.getPrefData().numItems(uidx) > 0 && dataset.getPrefData().numUsers(vidx) > 0)
            {
                Optional<IdxPref> pref = dataset.getPrefData().getPreference(uidx, vidx);
                double value = pref.map(idxPref1 -> idxPref1.v2).orElse(0.0);
                this.retrievedData.updateRating(uidx, vidx, value);
                if(!((ContactDataset<U>) this.dataset).isDirected())
                {
                    this.retrievedData.updateRating(vidx, uidx, value);
                }
                else if(pref.isPresent() && this.notReciprocal)
                {
                    if(dataset.getPrefData().numItems(vidx) > 0 && dataset.getPrefData().numItems(uidx) > 0)
                    {
                        pref = dataset.getPrefData().getPreference(vidx, uidx);
                        value = pref.map(idxPref -> idxPref.v2).orElse(0.0);
                        this.retrievedData.updateRating(vidx, uidx, value);
                    }
                }
            }
        });


        // Initialize the metrics
        this.metrics.forEach((name, metric) -> metric.initialize(warmup.getFullTraining(), false));
        this.numUsers = this.userList.size();
        this.endCondition.init();
        this.rng = new Random(rngSeed);
    }

    @Override
    public double fastUpdate(int uidx, int vidx)
    {
        // First, we check whether we have to update this:
        boolean hasRating = false;
        double value = 0.0;

        if(this.dataset.getPrefData().numItems(uidx) > 0 && this.dataset.getPrefData().numUsers(vidx) > 0)
        {
            Optional<IdxPref> realValue = this.dataset.getPrefData().getPreference(uidx, vidx);
            if(realValue.isPresent())
            {
                value = realValue.get().v2;
                hasRating = true;
            }
        }

        // If the rating exists, or the recommender uses not existing ratings, update the recommender.
        if(hasRating || this.recommender.usesAll())
        {
            this.recommender.update(uidx, vidx, value);
        }

        // Then, we remove the availability of user iidx for the user uidx.
        int index = this.availability.get(uidx).indexOf(vidx);
        this.availability.get(uidx).removeInt(index);

        if(this.availability.get(uidx).isEmpty())
        {
            int uIndex = this.userList.indexOf(uidx);
            this.userList.remove(uIndex);
            this.availability.remove(uidx);
        }

        if(!((ContactDataset<U>) dataset).isDirected() || (hasRating && this.notReciprocal))
        {
            index = this.availability.get(vidx).indexOf(uidx);
            if(index > 0) // This pair has not been previously recommended:
            {
                this.availability.get(vidx).removeInt(index);

                if(this.dataset.getPrefData().numItems(vidx) > 0 && this.dataset.getPrefData().numUsers(uidx) > 0)
                {
                    Optional<IdxPref> realValue = this.dataset.getPrefData().getPreference(vidx, uidx);
                    if(realValue.isPresent())
                    {
                        double auxValue = realValue.get().v2;
                        this.recommender.update(vidx, uidx, auxValue);
                    }
                }
            }
        }

        // Update the metric values:
        this.metrics.forEach((name, metric) -> metric.update(uidx, vidx, value));
        this.endCondition.update(uidx, vidx, value);
        ++this.iteration;

        return value;
    }
}
