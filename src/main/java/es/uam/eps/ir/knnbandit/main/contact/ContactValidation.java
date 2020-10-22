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
import es.uam.eps.ir.knnbandit.main.auxiliar.Validation;
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
 * Class for executing contact recommender systems in simulated interactive loops (with training)
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ContactValidation<U> extends Validation<U,U>
{
    private final ContactDataset<U> dataset;
    private final Map<String, Supplier<CumulativeMetric<U,U>>> metrics;
    private final boolean notReciprocal;

    public ContactValidation(String input, String separator, Parser<U> parser, boolean directed, boolean notReciprocal)
    {
        dataset = ContactDataset.load(input, directed, parser, separator);
        this.metrics = new HashMap<>();
        metrics.put("recall", () -> new CumulativeRecall<>(dataset.getNumRel(!directed || notReciprocal), 0.5));
        this.notReciprocal = notReciprocal;
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
        return new ContactOfflineDatasetRecommendationLoop<>(dataset, rec, localMetrics, endCond, notReciprocal, rngSeed);
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, U>>> getMetrics()
    {
        return metrics;
    }
}
