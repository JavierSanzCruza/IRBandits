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

import es.uam.eps.ir.knnbandit.io.Writer;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
            // Step 1: retrieve the previously computed iterations for this algorithm:
            List<Tuple3<Integer, Integer, Long>> list = new ArrayList<>();
            if (resume)
            {
                list = this.retrievePreviousIterations(file +  ".txt");
            }

            Writer writer = new Writer(file, loop.getMetrics());
            writer.writeHeader();

            // Step 2: if there are any, we update the loop with such values.
            if (resume && !list.isEmpty())
            {
                metricValues.putAll(this.updateWithPrevious(loop, list, writer, interval));
            }

            // Step 3: until the loop ends, we
            int currentIter = this.executeRemaining(loop, writer, interval, metricValues);
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
        if (f.exists() && !f.isDirectory()) // if the file exists, then recover the triplets:
        {
            // Once we know that the file exists, we open it.
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filename))))
            {
                String line = br.readLine();
                int len;
                if (line != null)
                {
                    String[] split = line.split("\t");
                    len = split.length;

                    // Read each line
                    while ((line = br.readLine()) != null)
                    {
                        split = line.split("\t");
                        if (split.length < len)
                        {
                            break;
                        }

                        // Obtain the triplet
                        int uidx = Parsers.ip.parse(split[1]);
                        int iidx = Parsers.ip.parse(split[2]);
                        long time = Parsers.lp.parse(split[len - 1]);

                        // Add it to the recovered list.
                        recovered.add(new Tuple3<>(uidx, iidx, time));
                    }
                }
            }
        }

        return recovered;
    }

    /**
     * Given the list of recovered triplets, updates the recommendation loop.
     *
     * @param loop      the recommendation loop.
     * @param recovered the list of recovered (uidx, iidx, time) triplets.
     * @param writer    a writer for storing the recommendation loop in a file.
     * @param interval  the interval between different data points.
     * @return a map containing the values of the metrics in certain time points.
     * @throws IOException if something fails while writing.
     */
    public Map<String, List<Double>> updateWithPrevious(FastRecommendationLoop<U, I> loop, List<Tuple3<Integer,Integer,Long>> recovered, Writer writer, int interval) throws IOException
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
            writer.writeLine(iter, uidx, iidx, metricVals, time);

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
     * @param writer       the writer.
     * @param interval     the interval.
     * @param metricValues the list of metric values.
     * @return the number of iterations for finishing the loop.
     */
    public int executeRemaining(FastRecommendationLoop<U, I> loop, Writer writer, int interval, Map<String, List<Double>> metricValues) throws IOException
    {
        List<String> metricNames = loop.getMetrics();

        // Apply it until the end.
        while (!loop.hasEnded())
        {
            long aa = System.currentTimeMillis();
            Pair<Integer> rating = loop.fastNextIteration();
            long bb = System.currentTimeMillis();

            if(rating == null)
                break; // Everything has finished

            int uidx = rating.v1();
            int iidx = rating.v2();
            long time = bb - aa;
            int numIter = loop.getCurrentIter();
            Map<String, Double> metrics = loop.getMetricValues();

            writer.writeLine(numIter, uidx, iidx, metrics, time);

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
