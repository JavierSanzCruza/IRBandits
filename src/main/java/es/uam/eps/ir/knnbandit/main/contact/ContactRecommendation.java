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
import es.uam.eps.ir.knnbandit.main.Recommendation;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.loop.ContactOfflineDatasetRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import org.ranksys.formats.parsing.Parser;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Class for executing contact recommender systems in simulated interactive loops (without training)
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @param <U> type of the users.
 */
public class ContactRecommendation<U> extends Recommendation<U,U>
{
    /**
     * The contact recommendation dataset.
     */
    private final ContactDataset<U> dataset;
    /**
     * The map of metric suppliers.
     */
    private final Map<String, Supplier<CumulativeMetric<U,U>>> metrics;
    /**
     * The cut-off of the recommendation.
     */
    private final int cutoff;

    /**
     * Constructor.
     * @param input         the file containing the dataset.
     * @param separator     separator for the different dataset registers.
     * @param parser        parser for reading the users.
     * @param directed      true if the network is directed.
     * @param notReciprocal true if we want to avoid recommending reciprocal edges to existing ones, false otherwise.
     * @param cutoff        the cutoff of the recommendation.
     * @param ioSelector    a selector for reading / writing files.
     */
    public ContactRecommendation(String input, String separator, Parser<U> parser, boolean directed, boolean notReciprocal, int cutoff, IOSelector ioSelector)
    {
        super(ioSelector);
        dataset = ContactDataset.load(input, directed, notReciprocal, parser, separator);
        this.metrics = new HashMap<>();
        metrics.put("recall", CumulativeRecall::new);
        metrics.put("gini", CumulativeGini::new);
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
}
