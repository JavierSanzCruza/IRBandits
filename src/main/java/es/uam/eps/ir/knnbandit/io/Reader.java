/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.io;

import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import org.jooq.lambda.tuple.Tuple3;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

/**
 * Interface for reading recommendation files.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface Reader
{
    /**
     * Initializes the reader.
     * @param filename name of the file.
     * @throws IOException if something fails while initializing.
     */
    void initialize(String filename) throws IOException;

    /**
     * Initializes the reader.
     * @param inputStream an input stream.
     * @throws IOException if something fails while initializing.
     */
    void initialize(InputStream inputStream) throws IOException;

    /**
     * Reads a single iteration.
     * @return a triplet containing a) the iteration number, b) the recommendation, c) the time needed for the recommendation.
     * @throws IOException if something fails while reading the file.
     */
    Tuple3<Integer, FastRecommendation, Long> readIteration() throws IOException;

    /**
     * Closes the reader.
     * @throws IOException if something fails while closing the reader.
     */
    void close() throws IOException;

    /**
     * Reads the header of the file.
     * @return a list containing the elements in the header of the file.
     * @throws IOException if something fails while reading the header.
     */
    List<String> readHeader() throws IOException;

    /**
     * Reads a whole file, and obtains the different user-item pairs.
     * @param filename the name of the file.
     * @return a list of user-item pairs.
     * @throws IOException if something fails while reading the file.
     */
    List<Pair<Integer>> readFile(String filename) throws IOException;


    /**
     * Reads a whole file, and obtains the different user-item pairs.
     * @param stream an input stream.
     * @return a list of user-item pairs.
     * @throws IOException if something fails while reading the stream.
     */
    List<Pair<Integer>> readFile(InputStream stream) throws IOException;

}
