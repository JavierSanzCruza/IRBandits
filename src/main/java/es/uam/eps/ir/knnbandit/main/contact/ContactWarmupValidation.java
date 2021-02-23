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
import es.uam.eps.ir.knnbandit.main.WarmupValidation;
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
 */
public class ContactWarmupValidation<U> extends WarmupValidation<U,U>
{
    private final ContactDataset<U> dataset;
    private final Map<String, Supplier<CumulativeMetric<U,U>>> metrics;
    private final WarmupType warmupType;
    private final int cutoff;

    public ContactWarmupValidation(String input, String separator, Parser<U> parser, boolean directed, boolean notReciprocal, WarmupType warmupType, int cutoff)
    {
        dataset = ContactDataset.load(input, directed, notReciprocal, parser, separator);
        this.metrics = new HashMap<>();
        metrics.put("recall", CumulativeRecall::new);
        this.warmupType = warmupType;
        this.cutoff = cutoff;
    }

    @Override
    protected Dataset<U, U> getDataset()
    {
        return dataset;
    }

    @Override
    protected Dataset<U, U> getValidationDataset(List<Pair<Integer>> validationPairs)
    {
        return ContactDataset.load(dataset, validationPairs, !dataset.useReciprocal());
    }

    @Override
    protected FastRecommendationLoop<U, U> getRecommendationLoop(Dataset<U, U> validDataset, InteractiveRecommenderSupplier<U,U> rec, EndCondition endCond, int rngSeed)
    {
        Map<String, CumulativeMetric<U,U>> localMetrics = new HashMap<>();
        metrics.forEach((name, supplier) -> localMetrics.put(name, supplier.get()));
        return new ContactOfflineDatasetRecommendationLoop<>((ContactDataset<U>) validDataset, rec, localMetrics, endCond, rngSeed, cutoff);
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, U>>> getMetrics()
    {
        return metrics;
    }

    @Override
    protected Warmup getWarmup(Dataset<U,U> validDataset, List<Pair<Integer>> trainData)
    {
        return ContactWarmup.load((ContactDataset<U>) validDataset, trainData.stream(), warmupType);
    }
}
