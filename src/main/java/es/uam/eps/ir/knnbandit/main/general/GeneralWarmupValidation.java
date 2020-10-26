/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.general;

import es.uam.eps.ir.knnbandit.data.datasets.ContactDataset;
import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.GeneralDataset;
import es.uam.eps.ir.knnbandit.main.Validation;
import es.uam.eps.ir.knnbandit.main.WarmupValidation;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.GeneralOfflineDatasetRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.ContactWarmup;
import es.uam.eps.ir.knnbandit.warmup.GeneralWarmup;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import es.uam.eps.ir.knnbandit.warmup.WarmupType;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;

/**
 * Class for applying validation in general recommendation contexts (i.e. movie, music recommendation)
 * where users and items are separate sets.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class GeneralWarmupValidation<U,I> extends WarmupValidation<U,I>
{
    /**
     * The dataset containing all the offline rating information.
     */
    private final GeneralDataset<U,I> dataset;
    /**
     * The set of metrics to compute.
     */
    private final Map<String, Supplier<CumulativeMetric<U,I>>> metrics;
    private final WarmupType warmupType;

    /**
     * Constructor.
     * @param input file containing the information about the ratings.
     * @param separator a separator for reading the file.
     * @param uParser parser for reading the set of users.
     * @param iParser parser for reading the set of items.
     * @param threshold the relevance threshold.
     * @param useRatings true if we have to consider the real ratings, false to binarize them according to the threshold value.
     * @throws IOException if something fails while reading the dataset.
     */
    public GeneralWarmupValidation(String input, String separator, Parser<U> uParser, Parser<I> iParser, double threshold, boolean useRatings, WarmupType warmup) throws IOException
    {
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x : (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);

        dataset = GeneralDataset.load(input, uParser, iParser, separator, weightFunction, relevance);
        this.metrics = new HashMap<>();
        metrics.put("recall", CumulativeRecall::new);
        this.warmupType = warmup;
    }


    @Override
    protected Dataset<U, I> getDataset()
    {
        return dataset;
    }

    @Override
    protected Dataset<U, I> getValidationDataset(List<Pair<Integer>> validationPairs)
    {
        return GeneralDataset.load(dataset, validationPairs);
    }


    @Override
    protected FastRecommendationLoop<U, I> getRecommendationLoop(InteractiveRecommenderSupplier<U,I> rec, EndCondition endCond, int rngSeed)
    {
        Map<String, CumulativeMetric<U,I>> localMetrics = new HashMap<>();
        metrics.forEach((name, supplier) -> localMetrics.put(name, supplier.get()));
        return new GeneralOfflineDatasetRecommendationLoop<>(dataset, rec, localMetrics, endCond, rngSeed);
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, I>>> getMetrics()
    {
        return metrics;
    }

    @Override
    protected Warmup getWarmup(List<Pair<Integer>> trainData)
    {
        return GeneralWarmup.load(dataset, trainData.stream(), warmupType);
    }
}