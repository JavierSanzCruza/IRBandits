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

import java.io.IOException;

/**
 * Interface for writing the outcome of an ensemble algorithm.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface EnsembleWriter extends Writer
{
    /**
     * Writes a line into the file.
     * @param numIter       the iteration number.
     * @param uidx          the user identifier.
     * @param iidx          the item identifier.
     * @param time          the execution time.
     * @param algorithm     the algorithm executed in this iteration.
     * @throws IOException if something fails while writing.
     */
    void writeEnsembleLine(int numIter, int uidx, int iidx, long time, String algorithm) throws IOException;

    /**
     * Writes a ranking into the file.
     * @param numIter   the iteration number.
     * @param rec       the recommendation ranking.
     * @param time      the execution time.
     * @param algorithm the algorithm executed in this iteration.
     * @throws IOException if something fails while writing.
     */
    void writeEnsembleRanking(int numIter, FastRecommendation rec, long time, String algorithm) throws IOException;
    /**
     * Writes the header of the file (if any)
     * @throws IOException if something fails while writing.
     */
    void writeEnsembleHeader() throws IOException;
}
