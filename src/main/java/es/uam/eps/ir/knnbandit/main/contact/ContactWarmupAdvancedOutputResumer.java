/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
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
import es.uam.eps.ir.knnbandit.selector.io.IOType;
import es.uam.eps.ir.knnbandit.main.WarmupAdvancedOutputResumer;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
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
 * Class that summarizes the metrics for different contact recommendation executions.
 * Uses warm-up data.
 *
 * @param <U> type of the users.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ContactWarmupAdvancedOutputResumer<U> extends WarmupAdvancedOutputResumer<U,U>
{
    /**
     * Contact recommendation dataset.
     */
    private final ContactDataset<U> dataset;
    /**
     * Metrics to compute.
     */
    private final Map<String, Supplier<CumulativeMetric<U,U>>> metrics;

    /**
     * Constructor.
     * @param input             file containing the information about the ratings.
     * @param separator         a separator for reading the file.
     * @param parser            parser for reading users from file.
     * @param directed          if the network is directed.
     * @param notReciprocal     true if we want to avoid recommending reciprocal edges to existing ones, false otherwise.
     * @param ioSelector        a selector for reading / writing files.
     * @param warmupIoSelector  a selector for reading the warm-up file.
     */
    public ContactWarmupAdvancedOutputResumer(String input, String separator, Parser<U> parser, boolean directed, boolean notReciprocal, IOSelector ioSelector, IOSelector warmupIoSelector)
    {
        super(ioSelector, warmupIoSelector);
        dataset = ContactDataset.load(input, directed, notReciprocal, parser, separator);
        this.metrics = new HashMap<>();
        metrics.put("recall", CumulativeRecall::new);
        metrics.put("gini", CumulativeGini::new);
    }

    @Override
    protected Dataset<U, U> getDataset()
    {
        return dataset;
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, U>>> getMetrics()
    {
        return metrics;
    }

    @Override
    protected Warmup getWarmup(List<Pair<Integer>> trainData)
    {
        return ContactWarmup.load(dataset, trainData.stream(), WarmupType.ONLYRATINGS);
    }
}
