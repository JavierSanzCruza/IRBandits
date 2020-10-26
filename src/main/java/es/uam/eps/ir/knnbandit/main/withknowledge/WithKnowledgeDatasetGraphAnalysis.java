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

import es.uam.eps.ir.knnbandit.data.datasets.DatasetWithKnowledge;
import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.main.DatasetGraphAnalysis;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

/**
 * Obtains the common items graph for domain-independent datasets with information
 * about whether the user knew about the items, and obtains some statistics.
 *
 * @see es.uam.eps.ir.knnbandit.main.DatasetGraphAnalysis
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class WithKnowledgeDatasetGraphAnalysis<U,I> extends DatasetGraphAnalysis<U,I>
{
    /**
     * The dataset with knowledge information
     */
    private final DatasetWithKnowledge<U,I> dataset;

    /**
     * Indicates which subset of the ratings to use for finding the graph.
     */
    private final KnowledgeDataUse dataUse;

    /**
     * Constructor.
     * @param input file containing the information about the ratings.
     * @param separator a separator for reading the file.
     * @param uParser parser for reading the set of users.
     * @param iParser parser for reading the set of items.
     * @param threshold the relevance threshold.
     * @param useRatings true if we have to consider the real ratings, false to binarize them according to the threshold value.
     * @param dataUse indicates which subset of the ratings to use for finding the graph.
     * @throws IOException if something fails while reading the dataset.
     */
    public WithKnowledgeDatasetGraphAnalysis(String input, String separator, Parser<U> uParser, Parser<I> iParser, double threshold, boolean useRatings, KnowledgeDataUse dataUse) throws IOException
    {
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x : (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
        this.dataset = DatasetWithKnowledge.load(input, uParser, iParser, separator, weightFunction, relevance);
        this.dataUse = dataUse;
    }

    @Override
    protected OfflineDataset<U, I> getDataset()
    {
        return dataset.getDataset(dataUse);
    }
}
