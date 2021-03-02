/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.data.datasets.reader;

import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.parsing.Parsers;

import java.util.Collection;
import java.util.HashSet;

/**
 * Simple reader for a stream dataset. It assumes that the candidate item set is
 * available.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class StreamCandidateSelectionDatasetReader<U,I> extends StreamDatasetReader<U,I>
{
    /**
     * Constructor.
     * @param file      the route to the dataset file.
     * @param uParser   a parser for reading the users.
     * @param iParser   a parser for reading the items.
     * @param separator the separator between the fields in a register in the dataset.
     */
    public StreamCandidateSelectionDatasetReader(String file, Parser<U> uParser, Parser<I> iParser, String separator)
    {
        super(file, uParser, iParser, separator);
    }

    @Override
    protected LogRegister<U,I> processRegister(String line)
    {
        LogRegister<U,I> register;

        // Process the register:
        String[] split = line.split(separator);
        if(split.length < 3)
        {
            return null;
        }

        U u = uParser.parse(split[0]);
        I i = iParser.parse(split[1]);
        double value = Parsers.dp.parse(split[2]);
        Collection<I> candidates = new HashSet<>();
        for (int j = 3; j < split.length; ++j)
        {
            candidates.add(iParser.parse(split[j]));
        }
        if (!candidates.contains(i)) candidates.add(i);

        return new LogRegister<>(u,i,value, candidates);
    }
}
