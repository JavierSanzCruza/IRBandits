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
import es.uam.eps.ir.knnbandit.main.TrainingStatistics;
import es.uam.eps.ir.knnbandit.selector.io.IOSelector;
import org.ranksys.formats.parsing.Parser;

/**
 * Class for computing the statistics for training data. It uses contact recommendation data.
 *
 * @param <U> type of the users.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ContactTrainingStatistics<U> extends TrainingStatistics<U,U>
{
    /**
     * The dataset.
     */
    private final ContactDataset<U> dataset;

    /**
     * Constructor.
     * @param input             file containing the information about the ratings.
     * @param separator         a separator for reading the file.
     * @param parser            parser for reading the set of users.
     * @param directed          true if the network is directed, false otherwise.
     * @param notReciprocal     true if we want to avoid recommending reciprocal edges to existing ones, false otherwise.
     * @param warmupIOSelector  selects the reader for the warm-up data.
     */
    public ContactTrainingStatistics(String input, String separator, Parser<U> parser, boolean directed, boolean notReciprocal, IOSelector warmupIOSelector)
    {
        super(warmupIOSelector);
        dataset = ContactDataset.load(input, directed, notReciprocal, parser, separator);
    }

    @Override
    protected Dataset<U, U> getDataset()
    {
        return dataset;
    }
}
