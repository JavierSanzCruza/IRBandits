/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.io;

import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendation;

import java.io.BufferedWriter;
import java.io.*;
import java.util.List;
import java.util.Map;

/**
 * Class for writing a recommendation loop.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class Writer
{
    /**
     * A writer, for printing the results into a file.
     */
    private BufferedWriter bw;
    /**
     * The set of metrics to use.
     */
    private final List<String> metricNames;

    /**
     * Constructor.
     *
     * @param filename    the name of the file in which to store the output.
     * @param metricNames the names of the metrics.
     * @throws IOException if something fails while creating the writer.
     */
    public Writer(String filename, List<String> metricNames) throws IOException
    {
        this.bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)));
        this.metricNames = metricNames;
    }

    /**
     * Writes the header of the output file.
     */
    public void writeHeader() throws IOException
    {
        StringBuilder builder = new StringBuilder();
        builder.append("numIter");
        builder.append("\tuidx");
        builder.append("\tiidx");
        for (String metric : metricNames)
        {
            builder.append("\t");
            builder.append(metric);
        }
        builder.append("\ttime");
        bw.write(builder.toString());
    }

    /**
     * Writes a line of the output file
     *
     * @param numIter current iteration number.
     * @param uidx    user identifier.
     * @param iidx    item identifier.
     * @param metrics metric values.
     * @param time    time needed to execute this iteration
     * @throws IOException if something fails while writing.
     */
    public void writeLine(int numIter, int uidx, int iidx, Map<String, Double> metrics, long time) throws IOException
    {
        StringBuilder builder = new StringBuilder();
        builder.append("\n");
        builder.append(numIter);
        builder.append("\t");
        builder.append(uidx);
        builder.append("\t");
        builder.append(iidx);
        for (String metric : metricNames)
        {
            builder.append("\t");
            builder.append(metrics.get(metric));
        }
        builder.append("\t");
        builder.append(time);
        bw.write(builder.toString());
    }

    /**
     * Writes a recommendation ranking.
     *
     * @param numIter   iteration number
     * @param rec       the recommendation
     * @param metrics   the metrics
     * @param time      the time needed to execute this iteration.
     * @throws IOException if something fails while writing.
     */
    public void writeRanking(int numIter, FastRecommendation rec, Map<String, Double> metrics, long time) throws IOException
    {
        int uidx = rec.getUidx();
        for(int iidx : rec.getIidxs())
        {
            this.writeLine(numIter, uidx, iidx, metrics, time);
        }
    }

    /**
     * Closes the writer.
     *
     * @throws IOException if something fails while closing.
     */
    public void close() throws IOException
    {
        if (this.bw != null)
        {
            this.bw.close();
        }
        this.bw = null;
    }
}
