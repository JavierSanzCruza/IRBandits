/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.io;

import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import org.jooq.lambda.tuple.Tuple3;
import org.jooq.lambda.tuple.Tuple4;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

/**
 * Interface for reading the outcome of an ensemble algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface EnsembleReader extends Reader
{
    /**
     * Reads an individual iteration for an ensemble.
     * @return a tuple containing (iteration, recommendation, time, algorithm).
     * @throws IOException if something fails while reading the file.
     */
    Tuple4<Integer, FastRecommendation, Long, String> readIterationEnsemble() throws IOException;

    /**
     * Reads a whole file, and obtains the different user-item pairs.
     * @param filename the name of the file.
     * @return a list of user-item-algorithm triplets.
     * @throws IOException if something fails while reading the file.
     */
    List<Tuple3<Integer, Integer, String>> readEnsembleFile(String filename) throws IOException;

    /**
     * Reads a whole file, and obtains the different user-item pairs.
     * @param stream an input stream.
     * @return a list of user-item-algorithm triplets.
     * @throws IOException if something fails while reading the stream.
     */
    List<Tuple3<Integer, Integer, String>> readEnsembleFile(InputStream stream) throws IOException;
}
