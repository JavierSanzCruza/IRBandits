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
import es.uam.eps.ir.knnbandit.main.TrainingStatistics;
import es.uam.eps.ir.knnbandit.selector.io.IOSelector;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

/**
 * Class for computing the statistics for training data. It uses general-domain
 * recommendation data.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class GeneralTrainingStatistics<U,I> extends TrainingStatistics<U,I>
{
    /**
     * The dataset.
     */
    private final GeneralDataset<U,I> dataset;

    /**
     * Constructor.
     * @param input             file containing the information about the ratings.
     * @param separator         a separator for reading the file.
     * @param uParser           parser for reading the set of users.
     * @param iParser           parser for reading the set of items.
     * @param threshold         the relevance threshold.
     * @param useRatings        true if we have to consider the real ratings, false to binarize them according to the threshold value.
     * @param warmupIoSelector  selects the reader for the warm-up data.
     * @throws IOException if something fails while reading the dataset.
     */
    public GeneralTrainingStatistics(String input, String separator, Parser<U> uParser, Parser<I> iParser, double threshold, boolean useRatings, IOSelector warmupIoSelector) throws IOException
    {
        super(warmupIoSelector);
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x : (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
        dataset = GeneralDataset.load(input, uParser, iParser, separator, weightFunction, relevance);
    }

    @Override
    protected Dataset<U, I> getDataset()
    {
        return this.dataset;
    }
}
