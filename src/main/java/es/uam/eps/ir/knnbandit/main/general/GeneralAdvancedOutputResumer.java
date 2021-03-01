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
import es.uam.eps.ir.knnbandit.io.IOType;
import es.uam.eps.ir.knnbandit.main.AdvancedOutputResumer;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;

public class GeneralAdvancedOutputResumer<U,I> extends AdvancedOutputResumer<U,I>
{
    private final GeneralDataset<U,I> dataset;
    private final Map<String, Supplier<CumulativeMetric<U, I>>> metrics;
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
    public GeneralAdvancedOutputResumer(String input, String separator, Parser<U> uParser, Parser<I> iParser, double threshold, boolean useRatings, IOType ioType, boolean gzipped) throws IOException
    {
        super(ioType, gzipped);
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x : (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
        dataset = GeneralDataset.load(input, uParser, iParser, separator, weightFunction, relevance);
        this.metrics = new HashMap<>();
        metrics.put("recall", CumulativeRecall::new);
        metrics.put("gini", CumulativeGini::new);
    }

    @Override
    protected Dataset<U, I> getDataset()
    {
        return this.dataset;
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, I>>> getMetrics()
    {
        return metrics;
    }
}
