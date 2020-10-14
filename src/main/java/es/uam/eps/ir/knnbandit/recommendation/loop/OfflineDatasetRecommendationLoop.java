package es.uam.eps.ir.knnbandit.recommendation.loop;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.FastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.utils.Pair;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;

import java.util.*;
import java.util.stream.Stream;

/**
 * Abstract class for determinin
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public abstract class OfflineDatasetRecommendationLoop<U,I> implements FastRecommendationLoop<U,I>
{
    /**
     * A complete dataset, containing all the necessary data.
     */
    protected final Dataset<U,I> dataset;
    /**
     * The interactive recommendation algorithm we are using in this loop
     */
    protected final InteractiveRecommender<U,I> recommender;
    /**
     * The metrics we want to compute
     */
    protected final Map<String, CumulativeMetric<U,I>> metrics;
    /**
     * The random seed for the random number generator.
     */
    protected final int rngSeed;

    /**
     * Random number generator.
     */
    protected Random rng;

    /**
     * Object determining whether the loop has ended or not.
     */
    protected final EndCondition endCondition;

    /**
     * List of users we can recommend items to.
     */
    protected final IntList userList = new IntArrayList();

    /**
     * The retrieved data.
     */
    protected FastUpdateablePreferenceData<U,I> retrievedData;

    /**
     * A list indicating the set of items we can recommend to each user in the system.
     */
    protected final Int2ObjectMap<IntList> availability;

    /**
     * The current number of users.
     */
    protected int numUsers;

    /**
     * The current iteration number.
     */
    protected int iteration;

    /**
     * Constructor.
     * @param dataset the dataset containing all the information.
     * @param recommender the interactive recommendation algorithm.
     * @param metrics the set of metrics we want to study.
     * @param endCondition the condition that establishes whether the loop has finished or not.
     */
    public OfflineDatasetRecommendationLoop(Dataset<U,I> dataset, InteractiveRecommender<U,I> recommender, Map<String, CumulativeMetric<U,I>>metrics, EndCondition endCondition)
    {
        // Initialize the dataset and the already retrieved data.
        this.dataset = dataset;
        this.retrievedData = SimpleFastUpdateablePreferenceData.load(Stream.empty(), dataset.getUserIndex(), dataset.getItemIndex());

        // Then, store the algorithm and metrics:
        this.recommender = recommender;
        this.metrics = metrics;

        this.endCondition = endCondition;


        // Initialize the iteration number.
        this.iteration = 0;

        // Initialize for the random number generator.
        this.rngSeed = 0;
        this.rng = new Random(rngSeed);

        this.availability = new Int2ObjectOpenHashMap<>();
    }

    @Override
    public Pair<Integer> fastNextRecommendation()
    {
        if(this.numUsers == 0) return null;

        boolean cont = false;
        int uidx;
        int iidx = -1;

        do
        {
            int index = rng.nextInt(numUsers);
            uidx = this.userList.get(index);
            IntList available = this.availability.get(index);
            if (available != null && !available.isEmpty())
            {
                iidx = recommender.next(uidx, available);
                if(iidx >= 0)
                {
                    cont = true;
                }
            }

            if(!cont)
            {
                this.numUsers--;
                this.userList.remove(index);
                this.availability.remove(index);
            }
        }
        while(!cont && this.numUsers > 0);

        if(this.numUsers == 0) return null;

        return new Pair<>(uidx, iidx);
    }

    @Override
    public Tuple2<U,I> nextRecommendation()
    {
        Pair<Integer> pair = this.fastNextRecommendation();
        Tuple2<U,I> tuple = null;
        if(pair != null)
        {
            tuple = new Tuple2<>(dataset.getUserIndex().uidx2user(pair.v1()), dataset.getItemIndex().iidx2item(pair.v2()));
        }
        return tuple;
    }

    @Override
    public boolean hasEnded()
    {
        return !(this.numUsers > 0) || this.endCondition.hasEnded();
    }

    @Override
    public Tuple3<Integer, Integer, Double> fastNextIteration()
    {
        Pair<Integer> rec = this.fastNextRecommendation();
        if(rec == null) return null;

        double value = this.fastUpdate(rec.v1(), rec.v2());
        if(Double.isNaN(value)) return null;
        return new Tuple3<>(rec.v1(), rec.v2(), value);
    }

    @Override
    public Tuple3<U, I, Double> nextIteration()
    {
        Tuple3<Integer, Integer, Double> next =  this.fastNextIteration();
        if(next == null) return null;
        return new Tuple3<>(dataset.getUserIndex().uidx2user(next.v1), dataset.getItemIndex().iidx2item(next.v2), next.v3);
    }

    @Override
    public double update(U u, I i)
    {
        return this.fastUpdate(this.dataset.getUserIndex().user2uidx(u), this.dataset.getItemIndex().item2iidx(i));
    }

}
