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

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.GeneralDataset;
import es.uam.eps.ir.knnbandit.selector.io.IOSelector;
import es.uam.eps.ir.knnbandit.selector.io.IOType;
import es.uam.eps.ir.knnbandit.main.Validation;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.GeneralOfflineDatasetRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;

/**
 * Class for choosing optimal recommendation algorithms (in general domains, like movies, songs...) in simulated interactive loops.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class GeneralValidation<U,I> extends Validation<U,I>
{
    /**
     * The dataset containing all the offline rating information.
     */
    private final GeneralDataset<U,I> dataset;
    /**
     * The set of metrics to compute.
     */
    private final Map<String, Supplier<CumulativeMetric<U,I>>> metrics;
    /**
     * The number of items to recommend each iteration.
     */
    private final int cutoff;

    /**
     * Constructor.
     * @param input         file containing the information about the ratings.
     * @param separator     a separator for reading the file.
     * @param uParser       parser for reading the set of users.
     * @param iParser       parser for reading the set of items.
     * @param threshold     the relevance threshold.
     * @param useRatings    true if we have to consider the real ratings, false to binarize them according to the threshold value.
     * @param cutoff        the number of items to recommend each iteration.
     * @param ioSelector    a selector for reading / writing files.
     * @throws IOException if something fails while reading the dataset.
     */
    public GeneralValidation(String input, String separator, Parser<U> uParser, Parser<I> iParser, double threshold, boolean useRatings, int cutoff, IOSelector ioSelector) throws IOException
    {
        super(ioSelector);
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x : (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);

        dataset = GeneralDataset.load(input, uParser, iParser, separator, weightFunction, relevance);
        this.metrics = new HashMap<>();
        metrics.put("recall", CumulativeRecall::new);
        this.cutoff = cutoff;
    }

    @Override
    protected Dataset<U, I> getDataset()
    {
        return dataset;
    }

    @Override
    protected FastRecommendationLoop<U, I> getRecommendationLoop(InteractiveRecommenderSupplier<U,I> rec, EndCondition endCond, int rngSeed)
    {
        Map<String, CumulativeMetric<U,I>> localMetrics = new HashMap<>();
        metrics.forEach((name, supplier) -> localMetrics.put(name, supplier.get()));
        return new GeneralOfflineDatasetRecommendationLoop<>(dataset, rec, localMetrics, endCond, rngSeed, cutoff);
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, I>>> getMetrics()
    {
        return metrics;
    }
}
