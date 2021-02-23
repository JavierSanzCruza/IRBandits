/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendation;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.Recommendation;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.Selection;
import es.uam.eps.ir.knnbandit.recommendation.loop.update.UpdateStrategy;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple2;

import java.util.*;

/**
 * Implementation of a generic recommendation loop.
 * @param <U> type of the users.
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class GenericRecommendationLoop<U,I> implements FastRecommendationLoop<U,I>
{
    /**
     * A selection mechanism for deciding the next target user and candidate item set.
     */
    protected final Selection<U,I> selection;
    /**
     * The interactive recommender for the loop.
     */
    protected final InteractiveRecommender<U,I> recommender;
    /**
     * The dataset including the necessary information.
     */
    protected final Dataset<U,I> dataset;
    /**
     * Determines whether the loop has finished or not.
     */
    protected final EndCondition endCond;
    /**
     * Establishes the values we have to use for updating the recommender/metrics/etc.
     */
    protected final UpdateStrategy<U,I> update;
    /**
     * The metrics to compute.
     */
    protected final Map<String, CumulativeMetric<U,I>> metrics;
    /**
     * The number of iterations.
     */
    protected int numIter;
    /**
     * True if the loop has ended, false otherwise.
     */
    protected boolean hasEnded;

    /**
     * The names of the metrics.
     */
    protected final List<String> metricNames;

    /**
     * The cutoff of the recommendation ranking. By default, equal to 1.
     */
    protected final int cutoff;

    /**
     * Constructor
     * @param dataset the dataset for retrieving the data and ratings.
     * @param selection selects the target user and candidate items for the recommendation.
     * @param provider a provider for the interactive recommendation approach
     * @param update the update strategy.
     * @param endcond the condition to end the loop
     * @param metrics the metrics to consider
     * @param rngSeed random number generator seed.
     */
    public GenericRecommendationLoop(Dataset<U,I> dataset, Selection<U,I> selection, InteractiveRecommenderSupplier<U,I> provider, UpdateStrategy<U,I> update, EndCondition endcond, Map<String, CumulativeMetric<U,I>> metrics, int rngSeed)
    {
        this.dataset = dataset;
        this.selection = selection;
        this.recommender = provider.apply(dataset, dataset, rngSeed);
        this.endCond = endcond;
        this.metrics = metrics;
        this.numIter = 0;
        this.metricNames = new ArrayList<>(metrics.keySet());
        this.update = update;
        Collections.sort(metricNames);
        this.hasEnded = false;
        this.cutoff = 1;
    }

    /**
     * Constructor
     * @param dataset the dataset for retrieving the data and ratings.
     * @param selection selects the target user and candidate items for the recommendation.
     * @param provider a provider for the interactive recommendation approach
     * @param update the update strategy.
     * @param endcond the condition to end the loop
     * @param metrics the metrics to consider
     * @param cutoff the cutoff of the recommendation ranking
     * @param rngSeed random number generator seed.
     *
     */
    public GenericRecommendationLoop(Dataset<U,I> dataset, Selection<U,I> selection, InteractiveRecommenderSupplier<U,I> provider, UpdateStrategy<U,I> update, EndCondition endcond, Map<String, CumulativeMetric<U,I>> metrics, int cutoff, int rngSeed)
    {
        this.dataset = dataset;
        this.selection = selection;
        this.recommender = provider.apply(dataset, dataset, rngSeed);
        this.endCond = endcond;
        this.metrics = metrics;
        this.numIter = 0;
        this.metricNames = new ArrayList<>(metrics.keySet());
        this.update = update;
        Collections.sort(metricNames);
        this.hasEnded = false;
        this.cutoff = cutoff;
    }

    @Override
    public void init()
    {
        this.selection.init(dataset);
        this.update.init(dataset);
        this.recommender.init();
        this.endCond.init(dataset);
        this.metrics.forEach((name, metric) -> metric.initialize(dataset));
        this.numIter = 0;
        this.hasEnded = false;
    }

    @Override
    public void init(Warmup warmup)
    {
        this.selection.init(dataset, warmup);
        this.update.init(dataset);
        this.recommender.init(this.update.getList(warmup).stream());
        this.endCond.init(dataset);
        this.metrics.forEach((name, metric) -> metric.initialize(dataset,warmup.getFullTraining()));
        this.numIter = 0;
        this.hasEnded = false;
    }


    @Override
    public Pair<Integer> fastNextIteration()
    {
        Pair<Integer> rec = fastNextRecommendation();
        if(rec != null)
        {
            this.fastUpdate(rec.v1(), rec.v2());
            this.increaseIteration();
        }
        return rec;
    }

    @Override
    public FastRecommendation fastNextIterationList()
    {
        FastRecommendation rec = fastNextRecommendationList();
        if(rec != null)
        {
            this.fastUpdate(rec);
            this.increaseIteration();
        }
        return rec;
    }

    @Override
    public Tuple2<U, I> nextRecommendation()
    {
        Pair<Integer> pair = this.fastNextRecommendation();
        Tuple2<U,I> tuple = null;

        if(pair != null)
        {
            tuple = new Tuple2<>(dataset.uidx2user(pair.v1()), dataset.iidx2item(pair.v1()));
        }
        return tuple;
    }

    @Override
    public Recommendation<U, I> nextRecommendationList()
    {
        FastRecommendation rec = this.fastNextRecommendationList();
        if(rec != null)
        {
            U u = dataset.uidx2user(rec.getUidx());
            List<I> list = new ArrayList<>();
            rec.getIidxs().forEach(iidx -> list.add(dataset.iidx2item(iidx)));
            return new Recommendation<>(u, list);
        }
        return null;
    }

    @Override
    public Pair<Integer> fastNextRecommendation()
    {
        Pair<Integer> pair = null;
        if(!this.hasEnded())
        {
            int uidx = selection.selectTarget();
            IntList candidates = selection.selectCandidates(uidx);
            if(uidx >= 0 && candidates != null && !candidates.isEmpty())
            {
                int iidx = this.recommender.next(uidx, candidates);
                pair =  new Pair<>(uidx, iidx);
            }
            else
            {
                this.hasEnded = true;
            }
        }
        return pair;
    }

    @Override
    public FastRecommendation fastNextRecommendationList()
    {
        IntList list = null;
        if(!this.hasEnded)
        {
            int uidx = selection.selectTarget();
            IntList candidates = selection.selectCandidates(uidx);
            if(uidx >= 0 && candidates != null && !candidates.isEmpty())
            {
                list = this.recommender.next(uidx, candidates, cutoff);
            }
            else
            {
                this.hasEnded = true;
            }

            return new FastRecommendation(uidx, list);
        }
        return null;
    }

    @Override
    public void fastUpdate(int uidx, int iidx)
    {
        Pair<List<FastRating>> updateValues = this.update.selectUpdate(uidx, iidx, this.selection);
        // First, update the recommender:
        List<FastRating> recValues = updateValues.v1();
        for(FastRating value : recValues)
        {
            recommender.update(value.uidx(), value.iidx(),value.value());
            selection.update(value.uidx(), value.iidx(), value.value());
        }

        // Then, update the metrics:
        List<FastRating> metricValues = updateValues.v2();
        for(FastRating value : metricValues)
        {
            metrics.forEach((name, metric) -> metric.update(value.uidx(), value.iidx(),value.value()));
            endCond.update(value.uidx(), value.iidx(),value.value());
        }
    }

    @Override
    public void fastUpdate(FastRecommendation rec)
    {
        int uidx = rec.getUidx();
        rec.getIidxs().forEach(iidx -> this.fastUpdate(uidx,iidx));
    }

    @Override
    public Tuple2<U,I> nextIteration()
    {
        Pair<Integer> rec = this.fastNextRecommendation();
        if(rec == null) return null;
        else return new Tuple2<>(this.dataset.uidx2user(rec.v1()), this.dataset.iidx2item(rec.v2()));
    }

    @Override
    public Recommendation<U,I> nextIterationList()
    {
        FastRecommendation rec = this.fastNextRecommendationList();
        if(rec == null) return null;
        else
        {
            U u = this.dataset.uidx2user(rec.getUidx());
            List<I> items = new ArrayList<>();
            rec.getIidxs().forEach(iidx -> items.add(dataset.iidx2item(iidx)));
            return new Recommendation<>(u, items);
        }
    }

    @Override
    public void update(Recommendation<U,I> rec)
    {
        U u = rec.getUser();
        rec.getItems().forEach(i -> this.update(u,i));
    }

    @Override
    public void update(U u, I i)
    {
        this.fastUpdate(this.dataset.user2uidx(u), this.dataset.item2iidx(i));
    }

    @Override
    public boolean hasEnded()
    {
        return this.hasEnded || this.endCond.hasEnded();
    }

    @Override
    public int getCurrentIter()
    {
        return this.numIter;
    }

    @Override
    public Map<String, Double> getMetricValues()
    {
        Map<String, Double> values = new HashMap<>();
        this.metrics.forEach((name, metric) -> values.put(name, metric.compute()));
        return values;
    }

    @Override
    public List<String> getMetrics()
    {
        return this.metricNames;
    }

    @Override
    public void increaseIteration()
    {
        ++this.numIter;
    }

    @Override
    public int getCutoff()
    {
        return this.cutoff;
    }
}
