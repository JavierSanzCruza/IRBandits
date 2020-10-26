/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.stream;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.ReplayerStreamDataset;
import es.uam.eps.ir.knnbandit.data.datasets.StreamDataset;
import es.uam.eps.ir.knnbandit.main.Recommendation;
import es.uam.eps.ir.knnbandit.metrics.ClickthroughRate;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.ReplayerRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Class for applying validation on a dataset using the replayer strategy.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ReplayerRecommendation<U,I> extends Recommendation<U,I>
{
    /**
     * The streaming dataset
     */
    private final StreamDataset<U,I> dataset;
    /**
     * The set of cumulative metrics.
     */
    private final Map<String, Supplier<CumulativeMetric<U,I>>> metrics;

    /**
     * Constructor.
     * @param input the file containing the log information to use in the replayer strategy.
     * @param separator a file separator.
     * @param userIndex the user index.
     * @param itemIndex the item index.
     * @param threshold the relevance threshold.
     * @param uParser a parser for reading the users.
     * @param iParser a parser for reading the items.
     * @throws IOException if something fails while reading the dataset.
     */
    public ReplayerRecommendation(String input, String separator, String userIndex, String itemIndex, double threshold, Parser<U> uParser, Parser<I> iParser) throws IOException
    {
        dataset = ReplayerStreamDataset.load(input, userIndex, itemIndex, separator, uParser, iParser, threshold);
        this.metrics = new HashMap<>();
        metrics.put("ctr", ClickthroughRate::new);
        metrics.put("gini", CumulativeGini::new);
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
        return new ReplayerRecommendationLoop<>(dataset, rec, localMetrics, endCond);
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, I>>> getMetrics()
    {
        return metrics;
    }
}
