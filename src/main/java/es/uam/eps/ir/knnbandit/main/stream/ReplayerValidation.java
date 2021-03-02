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
import es.uam.eps.ir.knnbandit.selector.io.IOSelector;
import es.uam.eps.ir.knnbandit.main.Validation;
import es.uam.eps.ir.knnbandit.metrics.ClickthroughRate;
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
 * Class for selecting the optimal recommendation algorithms using the replayer procedure.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ReplayerValidation<U,I> extends Validation<U,I>
{
    /**
     * The set of cumulative metrics.
     */
    private final Map<String, Supplier<CumulativeMetric<U,I>>> metrics;

    /**
     * File containing the log information to use in the replayer strategy.
     */
    private final String input;
    /**
     * Separator for the elements on each register.
     */
    private final String separator;
    /**
     * The user index.
     */
    private final String userIndex;
    /**
     * The item index.
     */
    private final String itemIndex;
    /**
     * The relevance threshold.
     */
    private final double threshold;
    /**
     * A parser for reading the users.
     */
    private final Parser<U> uParser;
    /**
     * A parser for reading the items.
     */
    private final Parser<I> iParser;

    /**
     * Constructor.
     * @param input         the file containing the log information to use in the replayer strategy.
     * @param separator     a file separator.
     * @param userIndex     the user index.
     * @param itemIndex     the item index.
     * @param threshold     the relevance threshold.
     * @param uParser       a parser for reading the users.
     * @param iParser       a parser for reading the items.
     * @param ioSelector    a selector for reading / writing files.
     */
    public ReplayerValidation(String input, String separator, String userIndex, String itemIndex, double threshold, Parser<U> uParser, Parser<I> iParser, IOSelector ioSelector)
    {
        super(ioSelector);
        this.input = input;
        this.separator = separator;
        this.userIndex = userIndex;
        this.itemIndex = itemIndex;
        this.threshold = threshold;
        this.uParser = uParser;
        this.iParser = iParser;


        this.metrics = new HashMap<>();
        metrics.put("ctr", ClickthroughRate::new);
    }


    @Override
    protected Dataset<U, I> getDataset()
    {
        try
        {
            return ReplayerStreamDataset.load(input, userIndex, itemIndex, separator, uParser, iParser, threshold);
        }
        catch (IOException e)
        {
            return null;
        }
    }

    @Override
    protected FastRecommendationLoop<U, I> getRecommendationLoop(InteractiveRecommenderSupplier<U,I> rec, EndCondition endCond, int rngSeed)
    {
        Map<String, CumulativeMetric<U,I>> localMetrics = new HashMap<>();
        metrics.forEach((name, supplier) -> localMetrics.put(name, supplier.get()));
        StreamDataset<U,I> dataset = (StreamDataset<U,I>) this.getDataset();

        return new ReplayerRecommendationLoop<>(dataset, rec, localMetrics, endCond, rngSeed);
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, I>>> getMetrics()
    {
        return metrics;
    }
}
