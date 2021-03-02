/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.withknowledge;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.DatasetWithKnowledge;
import es.uam.eps.ir.knnbandit.selector.io.IOSelector;
import es.uam.eps.ir.knnbandit.main.Validation;
import es.uam.eps.ir.knnbandit.metrics.CumulativeKnowledgeRecall;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.OfflineDatasetWithKnowledgeRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;

/**
 * Class for selecting the optimal recommendation algorithms in general recommendation contexts (i.e. movie, music recommendation)
 * where users and items are separate sets. The data contains information about whether the user knew about the
 * recommended items or not.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class WithKnowledgeValidation<U,I> extends Validation<U,I>
{
    /**
     * The dataset containing all the offline rating information.
     */
    private final DatasetWithKnowledge<U,I> dataset;
    /**
     * The set of metrics to compute.
     */
    private final Map<String, Supplier<CumulativeMetric<U,I>>> metrics;

    /**
     * Data use.
     */
    private final KnowledgeDataUse dataUse;
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
     * @param use           the type of data we are using (according to the whether the user knows or not about the items).
     * @param cutoff        the number of items to recommend each iteration.
     * @param ioSelector    a selector for reading / writing files.
     * @throws IOException if something fails while reading the dataset.
     */
    public WithKnowledgeValidation(String input, String separator, Parser<U> uParser, Parser<I> iParser, double threshold, boolean useRatings, KnowledgeDataUse use, int cutoff, IOSelector ioSelector) throws IOException
    {
        super(ioSelector);
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x : (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);

        dataset = DatasetWithKnowledge.load(input, uParser, iParser, separator, weightFunction, relevance);
        this.metrics = new HashMap<>();
        metrics.put("recall", CumulativeRecall::new);
        metrics.put("known-recall", () -> new CumulativeKnowledgeRecall<>(KnowledgeDataUse.ONLYKNOWN));
        metrics.put("unknown-recall", () -> new CumulativeKnowledgeRecall<>(KnowledgeDataUse.ONLYUNKNOWN));
        this.dataUse = use;
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
        return new OfflineDatasetWithKnowledgeRecommendationLoop<>(dataset, rec, localMetrics, endCond, this.dataUse, rngSeed, cutoff);
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, I>>> getMetrics()
    {
        return metrics;
    }
}
