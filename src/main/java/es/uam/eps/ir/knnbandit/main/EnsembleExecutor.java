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

import es.uam.eps.ir.knnbandit.io.EnsembleReader;
import es.uam.eps.ir.knnbandit.io.EnsembleWriter;
import es.uam.eps.ir.knnbandit.io.Writer;
import es.uam.eps.ir.knnbandit.recommendation.ensembles.FastEnsemble;
import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.selector.io.IOSelector;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2LongMap;
import it.unimi.dsi.fastutil.objects.Object2LongOpenHashMap;
import org.jooq.lambda.tuple.Tuple3;
import org.jooq.lambda.tuple.Tuple4;
import org.ranksys.core.util.tuples.Tuple2id;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Executes a recommendation loop which takes an ensemble as recommendation algorithm.
 * In addition to the metric results, this class obtains the proportion of use of each
 * recommendation algorithm in the ensemble.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class EnsembleExecutor<U,I> implements Executor<U,I>
{
    /**
     * Selects the format for the files.
     */
    private final IOSelector selector;

    /**
     * Stores the different metrics along time.
     */
    private final Map<String, List<Double>> metrics;

    /**
     * Stores the number of times each algorithm in the ensemble has been used.
     */
    private final Map<String, List<Long>> ensembleUsage;

    /**
     * Stores the curernt number of usages of each algorithm in the ensemble.
     */
    private final Object2LongMap<String> currentCounter;

    /**
     * Algorithm to index.
     */
    private final Object2IntMap<String> alg2index;
    /**
     * The ensemble.
     */
    private FastEnsemble<U,I> ensemble;

    /**
     * Constructor.
     * @param selector the format of the files.
     */
    public EnsembleExecutor(IOSelector selector)
    {
        this.selector = selector;
        this.metrics = new HashMap<>();
        this.ensembleUsage = new HashMap<>();
        this.ensemble = null;
        this.alg2index = new Object2IntOpenHashMap<>();
        this.currentCounter = new Object2LongOpenHashMap<>();
    }

    /**
     * Executes the full recommendation loop for a single algorithm.
     * @param loop the recommendation loop.
     * @param file the file in which we want to store everything.
     * @param resume true if we want to resume previous executions, false otherwise.
     * @param interval the pace at which we want to write the previous execution values.
     * @return the final number of iterations.
     */
    public int executeWithoutWarmup(FastRecommendationLoop<U,I> loop, String file, boolean resume, int interval)
    {
        // Initialize it:
        loop.init();
        this.reset(loop);
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
    public int executeWithWarmup(FastRecommendationLoop<U,I> loop, String file, boolean resume, int interval, Warmup warmup)
    {
        // Initialize it:
        loop.init(warmup);
        this.reset(loop);
        return execute(loop, file, resume, interval);
    }

    /**
     * Resets the configuration of the executor.
     */
    private void reset(FastRecommendationLoop<U,I> loop)
    {
        this.metrics.clear();
        this.ensembleUsage.clear();
        this.alg2index.clear();
        this.currentCounter.clear();

        loop.getMetrics().forEach(metric -> this.metrics.put(metric, new ArrayList<>()));
        this.ensemble = (FastEnsemble<U,I>) loop.getRecommender();
        for(int i = 0; i < ensemble.getAlgorithmCount(); ++i)
        {
            String alg = ensemble.getAlgorithmName(i);
            ensembleUsage.put(alg, new ArrayList<>());
            alg2index.put(alg, i);
            currentCounter.put(alg, 0L);
        }
    }

    /**
     * Executes the full recommendation loop for a single algorithm.
     * @param loop the recommendation loop.
     * @param file the file in which we want to store everything.
     * @param resume true if we want to resume previous executions, false otherwise.
     * @param interval the pace at which we want to write the previous execution values.
     * @return the final number of iterations, -1 if something goes wrong.
     */
    private int execute(FastRecommendationLoop<U, I> loop, String file, boolean resume, int interval)
    {
        loop.getMetrics().forEach(metricName -> metrics.put(metricName, new ArrayList<>()));
        EnsembleWriter writer = (EnsembleWriter) selector.getWriter();
        try
        {
            if(loop.getCutoff() == 1)
            {
                // Step 1: retrieve the previously computed iterations for this algorithm:
                List<Tuple4<Integer, Integer, Long, String>> list = new ArrayList<>();
                if (resume)
                {
                    list = this.retrievePreviousIterations(file);
                }

                writer.initialize(selector.getOutputStream(file));
                writer.writeEnsembleHeader();

                // Step 2: if there are any, we update the loop with such values.
                if (resume && !list.isEmpty())
                {
                    int currentIter = this.updateWithPrevious(loop, list, interval, writer);
                    System.out.println("Finished recovering " + currentIter + " iterations for " + file);
                }
            }
            else
            {
                List<Tuple3<FastRecommendation, Long, String>> list = new ArrayList<>();
                if(resume)
                {
                    list = this.retrievePreviousIterationsRankings(file);
                }

                writer.initialize(selector.getOutputStream(file));
                writer.writeHeader();

                if(resume && !list.isEmpty())
                {
                    int currentIter = this.updateWithPreviousRankings(loop, list, interval, writer);
                    System.out.println("Finished recovering " + currentIter + " iterations for " + file);
                }
            }

            // Step 3: until the loop ends, we
            int currentIter = this.executeRemaining(loop, interval, writer);
            writer.close();
            return currentIter;
        }
        catch (IOException ioe)
        {
            System.err.println("ERROR: Some error occurred when executing algorithm " + file);
            return -1;
        }
    }

    /**
     * Retrieves previous iterations of an execution, when the execution comes in the form of rankings.
     *
     * @param filename the name of the file.
     * @return a list containing the retrieved (recommendation ranking, time) tuples.
     * @throws IOException if something fails while reading the file.
     */
    private List<Tuple3<FastRecommendation, Long, String>> retrievePreviousIterationsRankings(String filename) throws IOException
    {
        List<Tuple3<FastRecommendation, Long, String>> recovered = new ArrayList<>();

        File f = new File(filename);
        if(f.exists() && !f.isDirectory())
        {
            EnsembleReader reader = (EnsembleReader) selector.getReader();
            reader.initialize(selector.getInputStream(filename));
            reader.readHeader();

            Tuple4<Integer, FastRecommendation, Long, String> line;

            // Read each line
            while((line = reader.readIterationEnsemble()) != null)
            {
                recovered.add(new Tuple3<>(line.v2, line.v3, line.v4));
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
    private List<Tuple4<Integer,Integer,Long, String>> retrievePreviousIterations(String filename) throws IOException
    {
        // Initialize the list
        List<Tuple4<Integer,Integer,Long, String>> recovered = new ArrayList<>();

        File f = new File(filename);
        if(f.exists() && !f.isDirectory())
        {
            EnsembleReader reader = (EnsembleReader) selector.getReader();
            reader.initialize(selector.getInputStream(filename));
            reader.readHeader();

            Tuple4<Integer, FastRecommendation, Long, String> line;

            // Read each line
            while((line = reader.readIterationEnsemble()) != null)
            {
                int uidx = line.v2.getUidx();
                long time = line.v3;
                for(Tuple2id id : line.v2.getIidxs())
                    recovered.add(new Tuple4<>(uidx, id.v1, time, line.v4));
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
    private int updateWithPreviousRankings(FastRecommendationLoop<U,I> loop, List<Tuple3<FastRecommendation, Long, String>> recovered, int interval, EnsembleWriter writer) throws IOException
    {
        for(Tuple3<FastRecommendation, Long, String> tuple : recovered)
        {
            FastRecommendation rec = tuple.v1;
            long time = tuple.v2;
            String algorithm = tuple.v3;

            ((Object2LongOpenHashMap<String>) this.currentCounter).addTo(algorithm, 1L);

            loop.fastUpdateNotRec(rec);
            loop.increaseIteration();

            int iter = loop.getCurrentIter();

            Map<String, Double> metricVals = loop.getMetricValues();
            writer.writeEnsembleRanking(iter, rec, time, algorithm);

            if(iter % interval == 0)
            {
                for(String name : metrics.keySet())
                {
                    double value = metricVals.get(name);
                    metrics.get(name).add(value);
                }

                for(String name : ensembleUsage.keySet())
                {
                    long value = this.currentCounter.get(name);
                    ensembleUsage.get(name).add(value);
                }
            }
        }

        if(!loop.hasEnded())
        {
            for(Tuple3<FastRecommendation, Long, String> tuple : recovered)
            {
                ensemble.setCurrentAlgorithm(this.alg2index.get(tuple.v3));
                loop.fastUpdateRec(tuple.v1);
            }
        }

        return loop.getCurrentIter();
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
    private int updateWithPrevious(FastRecommendationLoop<U, I> loop, List<Tuple4<Integer,Integer,Long, String>> recovered, int interval, Writer writer) throws IOException
    {
        for(Tuple4<Integer,Integer,Long, String> triplet : recovered)
        {
            int uidx = triplet.v1();
            int iidx = triplet.v2();
            long time = triplet.v3();
            String algorithm = triplet.v4();

            loop.fastUpdateNotRec(uidx, iidx);
            loop.increaseIteration();
            int iter = loop.getCurrentIter();
            ((Object2LongOpenHashMap<String>) this.currentCounter).addTo(algorithm, 1L);

            Map<String, Double> metricVals = loop.getMetricValues();
            writer.writeLine(iter, uidx, iidx, time);

            if(iter % interval == 0)
            {
                for(String name : metricVals.keySet())
                {
                    double value = metricVals.get(name);
                    metrics.get(name).add(value);
                }

                for(String name : ensembleUsage.keySet())
                {
                    long value = currentCounter.get(name);
                    ensembleUsage.get(name).add(value);
                }
            }
        }

        if(!loop.hasEnded())
        {
            for(Tuple4<Integer, Integer, Long, String> tuple : recovered)
            {
                ensemble.setCurrentAlgorithm(this.alg2index.get(tuple.v3));
                loop.fastUpdateRec(tuple.v1, tuple.v2);
            }
        }
        return loop.getCurrentIter();
    }

    /**
     * Execute the remaining loop
     *
     * @param loop         the recommendation loop.
     * @param interval     the interval.
     * @return the number of iterations for finishing the loop.
     */
    private int executeRemaining(FastRecommendationLoop<U, I> loop, int interval, EnsembleWriter writer) throws IOException
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
                int currentAlg = ensemble.getCurrentAlgorithm();
                String algName = ensemble.getAlgorithmName(currentAlg);
                ((Object2LongOpenHashMap<String>) currentCounter).addTo(algName, 1);

                writer.writeEnsembleLine(numIter, uidx, iidx, time, algName);
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
                int currentAlg = ensemble.getCurrentAlgorithm();
                String algName = ensemble.getAlgorithmName(currentAlg);
                ((Object2LongOpenHashMap<String>) currentCounter).addTo(algName, 1);

                writer.writeEnsembleRanking(numIter, rec, time, algName);
            }

            if (numIter % interval == 0)
            {
                for (String name : metricNames)
                {
                    double value = metrics.get(name);
                    this.metrics.get(name).add(value);
                }

                for(String name : currentCounter.keySet())
                {
                    long value = currentCounter.get(name);
                    ensembleUsage.get(name).add(value);
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
                this.metrics.get(name).add(value);
            }

            for(String name : currentCounter.keySet())
            {
                long value = currentCounter.get(name);
                ensembleUsage.get(name).add(value);
            }
        }

        return numIter;
    }

    /**
     * Obtains the metrics from the execution.
     * @return the metrics.
     */
    public Map<String, List<Double>> getMetrics()
    {
        return this.metrics;
    }

    /**
     * The usage of ensembles.
     * @return the usage of ensembles.
     */
    public Map<String, List<Long>> getEnsembleUsage()
    {
        return this.ensembleUsage;
    }
}
