/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.ensembles;

import es.uam.eps.ir.knnbandit.data.datasets.GeneralDataset;
import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.data.datasets.builder.BinaryGeneralDatasetBuilder;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.FastInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.ensembles.dynamic.optimizers.DynamicOptimizer;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.NonSequentialSelection;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.user.RoundRobinSelector;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.GeneralWarmup;
import es.uam.eps.ir.knnbandit.warmup.OfflineWarmup;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import es.uam.eps.ir.knnbandit.warmup.WarmupType;
import es.uam.eps.ir.ranksys.core.Recommendation;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.*;

import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Dynamic ensemble. Each k iterations, it selects one between many recommenders.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class DynamicEnsemble<U,I> extends AbstractEnsemble<U,I>
{
    /**
     * The number of epochs before changing the recommendation algorithm.
     */
    private final int numEpochs;
    /**
     * The data used as warm-up.
     */
    private final List<FastRating> warmup;
    /**
     * The number of epochs in this cycle.
     */
    private int currentEpoch;
    /**
     * The number of items to use in the recommendation cutoff.
     */
    private final int validCutoff;
    /**
     * The percentage of the warm-up data to be provided to the algorithms as input.
     */
    private final double percValid;
    /**
     * Establishes the metric we want to evaluate for the validation.
     */
    private final DynamicOptimizer<U,I> optimizer;

    /**
     * Constructor.
     * @param uIndex the user index.
     * @param iIndex the item index.
     * @param recs a map containing the recommenders.
     * @param ignoreNotRated true if we only update the ratings when they exist in the dataset, false otherwise.
     * @param numEpochs the number of epochs before changing the selected algorithm.
     * @param validCutoff the cut-off for the validation rankings.
     * @param percValid the percentage of the current data we use as training for validation.
     */
    public DynamicEnsemble(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, Map<String, InteractiveRecommenderSupplier<U,I>> recs, boolean ignoreNotRated, int numEpochs, int validCutoff, double percValid, DynamicOptimizer<U,I> optimizer)
    {
        super(uIndex, iIndex, ignoreNotRated, recs);

        this.numEpochs = numEpochs;
        this.warmup = new ArrayList<>();
        this.validCutoff = validCutoff;
        this.percValid = percValid;
        this.optimizer = optimizer;
    }


    @Override
    public void init()
    {
        super.init();
        warmup.clear();
        currentEpoch = 0;
    }
    @Override
    public void init(Stream<FastRating> values)
    {
        warmup.clear();
        values.forEach(warmup::add);
        super.init(warmup.stream());
        currentEpoch = 0;
    }

    @Override
    public int next(int uidx, IntList available)
    {
        currentEpoch = (currentEpoch+1)%numEpochs;
        return super.next(uidx, available);
    }

    @Override
    public IntList next(int uidx, IntList available, int k)
    {
        currentEpoch = (currentEpoch+1)%numEpochs;
        return super.next(uidx, available, k);
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

        // We first build the validation dataset:
        BinaryGeneralDatasetBuilder<U,I> builder = new BinaryGeneralDatasetBuilder<>();
        OfflineDataset<U,I> validation = (GeneralDataset<U,I>) builder.buildFromStream(uIndex, iIndex, warmup);

        // We load the warmup data.
        OfflineWarmup warmup = GeneralWarmup.load(validation, training.stream(), this.ignoreNotRated ? WarmupType.ONLYRATINGS : WarmupType.FULL);

        // We optimize a given ranking metric at cutoff k (k is introduced in the constructor).
        this.optimizer.init(validation, validCutoff);

        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();
        for(int i = 0; i < suppliers.size(); ++i)
        {
            // For each recommender in the ensemble, we initialize it using the warm-up
            FastInteractiveRecommender<U,I> rec = suppliers.get(i).apply(this.uIndex, this.iIndex);
            rec.init(this.ignoreNotRated ? warmup.getCleanTraining().stream() : warmup.getFullTraining().stream());

            // and we compute the value of a given ranking metric (precision, recall, nDCG, ...)
            double prec = validation.getUidxWithPreferences().mapToDouble(uidx ->
            {
                IntList available = warmup.getAvailability().get(uidx);
                IntList res = rec.next(uidx, available, validCutoff);
                Recommendation<U,I> recomm = new Recommendation<>(this.uIndex.uidx2user(uidx), res.stream().map(iidx -> new Tuple2od<>(this.iIndex.iidx2item(iidx), 1.0)).collect(Collectors.toList()));
                return this.optimizer.evaluate(recomm);
            }).sum();

            // We choose the best recommender (if there is a tie, we choose one of the best ones at random).
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
    protected void updateEnsemble(int uidx, int iidx, double value)
    {
        this.warmup.add(new FastRating(uidx, iidx, value));
    }

    @Override
    protected int selectAlgorithm(int uidx)
    {
        if(this.currentAlgorithm == -1 || this.currentEpoch == 0)
        {
            return this.validate();
        }
        return this.currentAlgorithm;
    }
}