/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.contact;

import es.uam.eps.ir.knnbandit.data.datasets.ContactDataset;
import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.selector.io.IOSelector;
import es.uam.eps.ir.knnbandit.main.WarmupRecommendation;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.ContactOfflineDatasetRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.ContactWarmup;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import es.uam.eps.ir.knnbandit.warmup.WarmupType;
import org.ranksys.formats.parsing.Parser;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Class for executing contact recommender systems in simulated interactive loops (with training)
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @param <U> type of the users.
 */
public class ContactWarmupRecommendation<U> extends WarmupRecommendation<U,U>
{
    /**
     * The contact recommendation dataset.
     */
    private final ContactDataset<U> dataset;
    /**
     * The metrics to compute.
     */
    private final Map<String, Supplier<CumulativeMetric<U,U>>> metrics;
    /**
     * Selects which warmup we use: all the links in the set or just the ones in the network.
     */
    private final WarmupType warmupType;
    /**
     * The number of items to recommend each iteration.
     */
    private final int cutoff;

    /**
     * Constructor.
     * @param input             the file containing the dataset.
     * @param separator         separator for the different dataset registers.
     * @param parser            parser for reading the users.
     * @param directed          true if the network is directed.
     * @param notReciprocal     true if we want to avoid recommending reciprocal edges to existing ones, false otherwise.
     * @param warmupType        selects which warmup we use: all the links in the set or just the ones in the network.
     * @param cutoff            the cutoff of the recommendation.
     * @param ioSelector        a selector for reading / writing files.
     * @param warmupIOSelector a selector for reading warm-up files.
     */
    public ContactWarmupRecommendation(String input, String separator, Parser<U> parser, boolean directed, boolean notReciprocal, WarmupType warmupType, int cutoff, IOSelector ioSelector, IOSelector warmupIOSelector)
    {
        super(ioSelector, warmupIOSelector);
        dataset = ContactDataset.load(input, directed, notReciprocal, parser, separator);
        this.metrics = new HashMap<>();
        metrics.put("recall", CumulativeRecall::new);
        metrics.put("gini", CumulativeGini::new);
        this.warmupType = warmupType;
        this.cutoff = cutoff;
    }

    @Override
    protected Dataset<U, U> getDataset()
    {
        return dataset;
    }

    @Override
    protected FastRecommendationLoop<U, U> getRecommendationLoop(InteractiveRecommenderSupplier<U,U> rec, EndCondition endCond, int rngSeed)
    {
        Map<String, CumulativeMetric<U,U>> localMetrics = new HashMap<>();
        metrics.forEach((name, supplier) -> localMetrics.put(name, supplier.get()));
        return new ContactOfflineDatasetRecommendationLoop<>(dataset, rec, localMetrics, endCond, rngSeed, cutoff);
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, U>>> getMetrics()
    {
        return metrics;
    }

    @Override
    protected Warmup getWarmup(List<Pair<Integer>> trainData)
    {
        return ContactWarmup.load(dataset, trainData.stream(), warmupType);
    }
}
