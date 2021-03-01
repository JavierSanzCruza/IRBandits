/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main;

import es.uam.eps.ir.knnbandit.io.ReaderInterface;
import es.uam.eps.ir.knnbandit.io.WriterInterface;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2id;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Executes a recommendation loop, and writes its values into a file:
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class Executor<U,I>
{
    private final WriterInterface writer;
    private final ReaderInterface reader;
    private final boolean gzipped;

    public Executor(WriterInterface writer, ReaderInterface reader, boolean gzipped)
    {
        this.writer = writer;
        this.reader = reader;
        this.gzipped = gzipped;
    }

    /**
     * Executes the full recommendation loop for a single algorithm.
     * @param loop the recommendation loop.
     * @param file the file in which we want to store everything.
     * @param resume true if we want to resume previous executions, false otherwise.
     * @param interval the pace at which we want to write the previous execution values.
     * @return the final number of iterations.
     */
    public Map<String, List<Double>> executeWithoutWarmup(FastRecommendationLoop<U,I> loop, String file, boolean resume, int interval)
    {
        // Initialize it:
        loop.init();
        return execute(loop, file, resume, interval);
    }

    /**
     * Executes the full recommendation loop for a single algorithm.
     * @param loop the recommendation loop.
     * @param file the file in which we want to store everything.
     * @param resume true if we want to resume previous executions, false otherwise.
     * @param interval the pace at which we want to write the previous execution values.
     * @return the final number of iterations.
     */
    public Map<String, List<Double>> executeWithWarmup(FastRecommendationLoop<U,I> loop, String file, boolean resume, int interval, Warmup warmup)
    {
        // Initialize it:
        loop.init(warmup);
        return execute(loop, file, resume, interval);
    }

    /**
     * Executes the full recommendation loop for a single algorithm.
     * @param loop the recommendation loop.
     * @param file the file in which we want to store everything.
     * @param resume true if we want to resume previous executions, false otherwise.
     * @param interval the pace at which we want to write the previous execution values.
     * @return the final number of iterations.
     */
    private Map<String, List<Double>> execute(FastRecommendationLoop<U, I> loop, String file, boolean resume, int interval)
    {
        Map<String, List<Double>> metricValues = new HashMap<>();
        loop.getMetrics().forEach(metricName -> metricValues.put(metricName, new ArrayList<>()));

        try
        {
            if(loop.getCutoff() == 1)
            {
                // Step 1: retrieve the previously computed iterations for this algorithm:
                List<Tuple3<Integer, Integer, Long>> list = new ArrayList<>();
                if (resume)
                {
                    list = this.retrievePreviousIterations(file);
                }

                OutputStream stream = gzipped ? new GZIPOutputStream(new FileOutputStream(file)) : new FileOutputStream(file);
                writer.initialize(stream);
                writer.writeHeader();

                // Step 2: if there are any, we update the loop with such values.
                if (resume && !list.isEmpty())
                {
                    metricValues.putAll(this.updateWithPrevious(loop, list, interval));
                }
            }
            else
            {
                List<Tuple2<FastRecommendation, Long>> list = new ArrayList<>();
                if(resume)
                {
                    list = this.retrievePreviousIterationsRankings(file);
                }

                writer.initialize(file);
                writer.writeHeader();

                if(resume && !list.isEmpty())
                {
                    metricValues.putAll(this.updateWithPreviousRankings(loop, list, interval));
                }
            }

            // Step 3: until the loop ends, we
            int currentIter = this.executeRemaining(loop, interval, metricValues);
            writer.close();
            return metricValues;
        }
        catch (IOException ioe)
        {
            System.err.println("ERROR: Some error occurred when executing algorithm " + file);
            return null;
        }
    }

    /**
     * Retrieves previous iterations of an execution, when the execution comes in the form of rankings.
     *
     * @param filename the name of the file.
     * @return a list containing the retrieved (recommendation ranking, time) tuples.
     * @throws IOException if something fails while reading the file.
     */
    public List<Tuple2<FastRecommendation, Long>> retrievePreviousIterationsRankings(String filename) throws IOException
    {
        List<Tuple2<FastRecommendation, Long>> recovered = new ArrayList<>();

        File f = new File(filename);
        if(f.exists() && !f.isDirectory())
        {
            InputStream stream = gzipped ? new GZIPInputStream(new FileInputStream(filename)) : new FileInputStream(filename);

            reader.initialize(stream);
            reader.readHeader();

            Tuple3<Integer, FastRecommendation, Long> line;

            // Read each line
            while((line = reader.readIteration()) != null)
            {
                recovered.add(new Tuple2<>(line.v2, line.v3));
            }

            reader.close();
        }

        return recovered;
    }

    /**
     * Retrieves previous iterations of an execution.
     *
     * @param filename the name of the file.
     * @return a list containing the retrieved (uidx, iidx, time) triplets.
     * @throws IOException if something fails while reading the file.
     */
    public List<Tuple3<Integer,Integer,Long>> retrievePreviousIterations(String filename) throws IOException
    {
        // Initialize the list
        List<Tuple3<Integer,Integer,Long>> recovered = new ArrayList<>();

        File f = new File(filename);
        if(f.exists() && !f.isDirectory())
        {
            InputStream stream = gzipped ? new GZIPInputStream(new FileInputStream(filename)) : new FileInputStream(filename);

            reader.initialize(stream);
            reader.readHeader();

            Tuple3<Integer, FastRecommendation, Long> line;

            // Read each line
            while((line = reader.readIteration()) != null)
            {
                int uidx = line.v2.getUidx();
                long time = line.v3;
                for(Tuple2id id : line.v2.getIidxs())
                    recovered.add(new Tuple3<>(uidx, id.v1, time));
            }

            reader.close();
        }

        return recovered;
    }

    /**
     * Given the list of recovered ranking-time tuples, updates the recommendation loop.
     *
     * @param loop      the recommendation loop.
     * @param recovered the list of recovered (ranking, time) tuples.
     * @param interval  the interval between different data points.
     * @return a map containing the values of the metrics in certain time points.
     * @throws IOException if something fails while writing.
     */
    public Map<String, List<Double>> updateWithPreviousRankings(FastRecommendationLoop<U,I> loop, List<Tuple2<FastRecommendation, Long>> recovered, int interval) throws IOException
    {
        List<String> metricNames = loop.getMetrics();
        Map<String, List<Double>> metricValues = new HashMap<>();

        for(String name : metricNames)
        {
            metricValues.put(name, new ArrayList<>());
        }

        for(Tuple2<FastRecommendation, Long> tuple : recovered)
        {
            FastRecommendation rec = tuple.v1;
            long time = tuple.v2;

            loop.fastUpdate(rec);
            loop.increaseIteration();

            int iter = loop.getCurrentIter();

            Map<String, Double> metricVals = loop.getMetricValues();
            writer.writeRanking(iter, rec, time);

            if(iter % interval == 0)
            {
                for(String name : metricNames)
                {
                    double value = metricVals.get(name);
                    metricValues.get(name).add(value);
                }
            }
        }

        return metricValues;
    }


    /**
     * Given the list of recovered triplets, updates the recommendation loop.
     *
     * @param loop      the recommendation loop.
     * @param recovered the list of recovered (uidx, iidx, time) triplets.
     * @param interval  the interval between different data points.
     * @return a map containing the values of the metrics in certain time points.
     * @throws IOException if something fails while writing.
     */
    public Map<String, List<Double>> updateWithPrevious(FastRecommendationLoop<U, I> loop, List<Tuple3<Integer,Integer,Long>> recovered, int interval) throws IOException
    {
        List<String> metricNames = loop.getMetrics();
        Map<String, List<Double>> metricValues = new HashMap<>();

        for(String name : metricNames)
        {
            metricValues.put(name, new ArrayList<>());
        }

        for(Tuple3<Integer,Integer,Long> triplet : recovered)
        {
            int uidx = triplet.v1();
            int iidx = triplet.v2();
            long time = triplet.v3();

            loop.fastUpdate(uidx, iidx);
            int iter = loop.getCurrentIter();

            Map<String, Double> metricVals = loop.getMetricValues();
            writer.writeLine(iter, uidx, iidx, time);

            if(iter % interval == 0)
            {
                for(String name : metricNames)
                {
                    double value = metricVals.get(name);
                    metricValues.get(name).add(value);
                }
            }
        }

        return metricValues;
    }

    /**
     * Execute the remaining loop
     *
     * @param loop         the recommendation loop.
     * @param interval     the interval.
     * @param metricValues the list of metric values.
     * @return the number of iterations for finishing the loop.
     */
    public int executeRemaining(FastRecommendationLoop<U, I> loop, int interval, Map<String, List<Double>> metricValues) throws IOException
    {
        List<String> metricNames = loop.getMetrics();
        boolean ranking = loop.getCutoff() > 1;

        // Apply it until the end.
        while (!loop.hasEnded())
        {
            Map<String, Double> metrics;
            int numIter;
            long time;
            if(!ranking)
            {
                long aa = System.currentTimeMillis();
                Pair<Integer> rating = loop.fastNextIteration();
                long bb = System.currentTimeMillis();

                if (rating == null)
                    break; // Everything has finished

                int uidx = rating.v1();
                int iidx = rating.v2();
                time = bb - aa;
                numIter = loop.getCurrentIter();
                metrics = loop.getMetricValues();

                writer.writeLine(numIter, uidx, iidx, time);
            }
            else
            {
                long aa = System.currentTimeMillis();
                FastRecommendation rec = loop.fastNextIterationList();
                long bb = System.currentTimeMillis();

                if(rec == null)
                    break; // Everything has finished

                time = bb-aa;
                numIter = loop.getCurrentIter();
                metrics = loop.getMetricValues();
                writer.writeRanking(numIter, rec, time);
            }

            if (numIter % interval == 0)
            {
                for (String name : metricNames)
                {
                    double value = metrics.get(name);
                    metricValues.get(name).add(value);
                }
            }
        }

        // Store the value of the last iteration.
        int numIter = loop.getCurrentIter();
        if (numIter % interval != 0)
        {
            Map<String, Double> metrics = loop.getMetricValues();
            for (String name : metricNames)
            {
                double value = metrics.get(name);
                metricValues.get(name).add(value);
            }
        }

        return numIter;
    }
}
