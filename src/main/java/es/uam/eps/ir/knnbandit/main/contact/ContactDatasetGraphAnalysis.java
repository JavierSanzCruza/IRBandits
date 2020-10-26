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
import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.main.DatasetGraphAnalysis;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;

/**
 * Obtains the common items graph for contact recommendation datasets and finds some statistics.
 * @see es.uam.eps.ir.knnbandit.main.DatasetGraphAnalysis
 *
 * @param <U> type of the users.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ContactDatasetGraphAnalysis<U> extends DatasetGraphAnalysis<U,U>
{
    /**
     * The contact recommendation dataset.
     */
    private final ContactDataset<U> dataset;
    /**
     * Constructor.
     * @param input file containing the information about the ratings.
     * @param separator a separator for reading the file.
     */
    public ContactDatasetGraphAnalysis(String input, String separator, Parser<U> parser, boolean directed, boolean notReciprocal)
    {
        dataset = ContactDataset.load(input, directed, notReciprocal, parser, separator);
    }

    @Override
    protected OfflineDataset<U, U> getDataset()
    {
        return dataset;
    }
}
