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
import es.uam.eps.ir.knnbandit.main.AdvancedOutputResumer;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
import es.uam.eps.ir.knnbandit.metrics.CumulativeKnowledgeRecall;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;


/**
 * Class that summarizes the metrics for different recommendation executions in a general domain (movies, music...).
 * The dataset includes information about whether the recommended items are known or not.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class WithKnowledgeAdvancedOutputResumer<U,I> extends AdvancedOutputResumer<U,I>
{
    /**
     * The dataset.
     */
    private final DatasetWithKnowledge<U,I> dataset;
    /**
     * The metrics to compute.
     */
    private final Map<String, Supplier<CumulativeMetric<U,I>>> metrics;

    /**
     * Constructor.
     * @param input         file containing the information about the ratings.
     * @param separator     a separator for reading the file.
     * @param uParser       parser for reading the set of users.
     * @param iParser       parser for reading the set of items.
     * @param threshold     the relevance threshold.
     * @param useRatings    true if we have to consider the real ratings, false to binarize them according to the threshold value.
     * @throws IOException if something fails while reading the dataset.
     */
    public WithKnowledgeAdvancedOutputResumer(String input, String separator, Parser<U> uParser, Parser<I> iParser, double threshold, boolean useRatings, IOSelector ioSelector) throws IOException
    {
        super(ioSelector);
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x : (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
        dataset = DatasetWithKnowledge.load(input, uParser, iParser, separator, weightFunction, relevance);
        this.metrics = new HashMap<>();
        metrics.put("recall", CumulativeRecall::new);
        metrics.put("known-recall", () -> new CumulativeKnowledgeRecall<>(KnowledgeDataUse.ONLYKNOWN));
        metrics.put("unknown-recall", () -> new CumulativeKnowledgeRecall<>(KnowledgeDataUse.ONLYUNKNOWN));
        metrics.put("gini", CumulativeGini::new);
    }

    @Override
    protected Dataset<U, I> getDataset()
    {
        return dataset;
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, I>>> getMetrics()
    {
        return metrics;
    }
}
