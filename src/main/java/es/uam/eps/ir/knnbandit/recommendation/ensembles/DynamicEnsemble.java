package es.uam.eps.ir.knnbandit.recommendation.ensembles;

import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.NonSequentialSelection;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.user.RoundRobinSelector;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.GeneralWarmup;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import es.uam.eps.ir.knnbandit.warmup.WarmupType;
import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.basic.Precision;
import es.uam.eps.ir.ranksys.metrics.rel.IdealRelevanceModel;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Dynamic ensemble. Each k iterations, it selects one between many recommenders.
 * @param <U> type of the users.
 * @param <I> type of the items.
 */
public class DynamicEnsemble<U,I> extends InteractiveRecommender<U,I> implements Ensemble<U,I>
{
    /**
     * Recommender suppliers.
     */
    private final List<InteractiveRecommenderSupplier<U,I>> suppliers;
    /**
     * Recommenders.
     */
    private final List<InteractiveRecommender<U,I>> recommenders;
    /**
     * Names of each recommender.
     */
    private final List<String> recNames;
    private final int[] hits;
    private final int[] misses;
    private final int numEpochs;
    private int lastRec;
    private final List<FastRating> warmup;
    private int currentEpoch;
    private Dataset<U,I> dataset;
    private final int validCutoff;
    private final double percValid;

    public DynamicEnsemble(Dataset<U,I> dataset, Map<String, InteractiveRecommenderSupplier<U,I>> recs, boolean ignoreNotRated, int numEpochs, int validCutoff, double percValid)
    {
        super(dataset, dataset, ignoreNotRated);
        this.recommenders = new ArrayList<>();
        this.recNames = new ArrayList<>();
        this.suppliers = new ArrayList<>();
        recs.forEach((name, rec) -> {
             recNames.add(name);
             recommenders.add(rec.apply(uIndex, iIndex));
             suppliers.add(rec);
        });

        this.hits = new int[recNames.size()];
        this.misses = new int[recNames.size()];

        this.numEpochs = numEpochs;
        lastRec = -1;

        this.warmup = new ArrayList<>();
        this.validCutoff = validCutoff;
        this.percValid = percValid;
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        // We first store the whole warm-up.
        values.forEach(warmup::add);
        recommenders.forEach(rec -> rec.init(warmup.stream()));
        for(int i = 0; i < recNames.size(); ++i)
        {
            this.hits[i] = 0;
            this.misses[i] = 0;
        }
        lastRec = -1;
        currentEpoch = 0;
    }

    @Override
    public int next(int uidx, IntList available)
    {
        if(lastRec == -1 || currentEpoch == 0)
        {
            this.lastRec = validate();
        }
        currentEpoch++;
        return this.recommenders.get(lastRec).next(uidx, available);
    }

    @Override
    public IntList next(int uidx, IntList available, int k)
    {
        if(lastRec == -1 || currentEpoch == 0)
        {
            this.lastRec = validate();
        }
        currentEpoch = (currentEpoch+1)%numEpochs;
        return this.recommenders.get(lastRec).next(uidx, available, k);
    }

    /**
     * Selects the next recommender to use in the ensemble:
     * @return the next recommender to use in the ensemble.
     */
    private int validate()
    {
        // We first select some tuples to use as training:
        List<Pair<Integer>> training = new ArrayList<>();

        Random rng = new Random(rngSeed);
        for(FastRating pair : this.warmup)
        {
            if(rng.nextDouble() < percValid)
            {
                training.add(new Pair<>(pair.uidx(), pair.iidx()));
            }
        }

        // Then, we build an (offline) dataset from the data we have -- in case this is not an Offline dataset,
        // an Unsupported operation exception arises.
        OfflineDataset<U,I> validationSet = (OfflineDataset<U,I>) dataset.load(warmup.stream().map(x -> new Pair<>(x.uidx(),x.iidx())).collect(Collectors.toList()));

        // Build the selection and the warmup data.
        NonSequentialSelection<U,I> nonSeqSel = new NonSequentialSelection<>(0, new RoundRobinSelector(), false);
        Warmup warmup = GeneralWarmup.load(validationSet, training.stream(), this.ignoreNotRated ? WarmupType.ONLYRATINGS : WarmupType.FULL);
        nonSeqSel.init(validationSet, warmup);

        // Optimize precision at k (where k is introduced at the constructor).
        IdealRelevanceModel<U,I> relModel = new IdealRelevanceModel<U, I>()
        {
            @Override
            protected UserIdealRelevanceModel<U, I> get(U u)
            {
                return new UserIdealRelevanceModel<U, I>()
                {
                    @Override
                    public Set<I> getRelevantItems()
                    {
                        return validationSet.getUserPreferences(u).filter(i -> i.v2 >= 1.0).map(i -> i.v1).collect(Collectors.toSet());
                    }

                    @Override
                    public boolean isRelevant(I i)
                    {
                        return validationSet.isRelevant(validationSet.getPreference(u,i).orElse(0.0));
                    }

                    @Override
                    public double gain(I i)
                    {
                        return validationSet.getPreference(u,i).orElse(0.0);
                    }
                };
            }
        };

        // Choose the recommender maximizing the metric.
        Precision<U,I> precision = new Precision<>(validCutoff, relModel);

        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();
        for(int i = 0; i < suppliers.size(); ++i)
        {
            InteractiveRecommender<U,I> rec = suppliers.get(i).apply(this.uIndex, this.iIndex);
            rec.init(this.ignoreNotRated ? warmup.getCleanTraining().stream() : warmup.getFullTraining().stream());

            double prec = validationSet.getUidxWithPreferences().mapToDouble(uidx ->
            {
                IntList available = nonSeqSel.selectCandidates(uidx);
                IntList res = rec.next(uidx, available, validCutoff);
                Recommendation<U,I> recomm = new Recommendation<>(this.uIndex.uidx2user(uidx), res.stream().map(iidx -> new Tuple2od<>(this.iIndex.iidx2item(iidx), 1.0)).collect(Collectors.toList()));
                return precision.evaluate(recomm);
            }).sum();

            if(prec > max)
            {
                max = prec;
                top.clear();
                top.add(i);
            }
            else if(prec == max)
            {
                top.add(i);
            }
        }

        if(top.size() == 1)
            return top.get(0);
        else return top.get(rng.nextInt(top.size()));
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        double newValue;
        if(!Double.isNaN(value) || !this.ignoreNotRated)
        {
            newValue = Constants.NOTRATEDNOTIGNORED;
        }
        else
        {
            return;
        }

        this.warmup.add(new FastRating(uidx, iidx, newValue));
        recommenders.forEach(rec -> rec.update(uidx, iidx, value));

        this.hits[lastRec] += (value > 0.0) ? 1 : 0;
        this.misses[lastRec] += (value > 0.0) ? 0 : 1;
    }

    @Override
    public int getCurrentAlgorithm()
    {
        return lastRec;
    }

    @Override
    public String getAlgorithmName(int idx)
    {
        return this.recNames.get(idx);
    }

    @Override
    public Pair<Integer> getAlgorithmStats(int idx)
    {
        return new Pair<>(hits[idx], misses[idx]);
    }
}