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

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.selector.io.IOSelector;
import es.uam.eps.ir.knnbandit.io.Reader;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;

import java.io.*;
import java.util.*;
import java.util.function.Supplier;

/**
 * Class for summarizing the outcomes of recommenders.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class AdvancedOutputResumer<U,I>
{
    /**
     * Identifier name for the time needed to execute the recommendations.
     */
    private final String TIMENAME = "time";
    /**
     * The input-output selector.
     */
    private final IOSelector ioSelector;


    /**
     * Constructor.
     * @param ioSelector input-output type selector for the reader / writer.
     */
    public AdvancedOutputResumer(IOSelector ioSelector)
    {
        this.ioSelector = ioSelector;
    }

    /**
     * Summarizes the contents of a directory.
     * @param input     the directory containing the recommendation executions.
     * @param points    the iterations we select from the summary.
     * @param recursive true if we want to extend our summary to internal directories of the selected one.
     * @throws IOException if something fails while reading or writing the files.
     */
    public void summarize(String input, IntList points, boolean recursive) throws IOException
    {
        Dataset<U,I> dataset = this.getDataset();
        System.out.println(dataset.toString());
        points.sort(Comparator.naturalOrder());

        File directory = new File(input);
        if(!directory.exists())
        {
            System.err.println("ERROR: Directory " + input + " does not exist.");
        }
        else if(directory.isDirectory())
        {
            this.readDirectory(directory, points, recursive);
        }
        else // if it is a file
        {
            Map<String, Map<String, Map<Integer, Double>>> res = new HashMap<>();
            res.put(directory.getName(), this.readFile(directory, points));
            this.printSummary(res, points, directory.getParentFile());
        }
    }

    /**
     * Reads a directory, resuming everything there.
     *
     * @param directory the name of the directory.
     * @param list      the number of iterations to measure.
     * @param recursive the recursive values.
     */
    private void readDirectory(File directory, IntList list, boolean recursive) throws IOException
    {
        long a = System.currentTimeMillis();
        System.out.println("Entered directory" + directory);

        File[] files = directory.listFiles();
        if(files == null)
        {
            System.err.println("Nothing found in directory " + directory);
            return;
        }

        // Differentiate between files and directories
        List<File> indivFiles = new ArrayList<>();
        List<File> directories = new ArrayList<>();

        for(File file : files)
        {
            if(file.isDirectory())
            {
                directories.add(file);
            }
            else if(!file.getName().contains("rngSeed") && !file.getName().contains("summary"))
            {
                indivFiles.add(file);
            }
        }

        // If there are individual files to summarize:
        if(!indivFiles.isEmpty())
        {
            // We first read the file and obtain the values.
            Map<String, Map<String, Map<Integer, Double>>> results = new HashMap<>();
            for (File f : indivFiles)
            {
                Map<String, Map<Integer, Double>> map = readFile(f, list);
                results.put(f.getName(), map);
            }

            this.printSummary(results, list, directory);
        }

        // If we set this algorithm as recursive, we do repeat for each subdirectory.
        if(recursive)
        {
            for (File dir : directories)
            {
                readDirectory(dir, list, true);
            }
        }
        else if(indivFiles.isEmpty())
        {
            System.err.println("Nothing found in directory " + directory);
        }

        long b = System.currentTimeMillis();
        System.out.println("Exited directory " + directory + " (" + (b - a) + " ms.)");
    }

    /**
     * Prints the summary of a directory.
     * @param results       a map, indexed by algorithm, containing the metric values at the given points.
     * @param list          the iteration numbers included in the summary.
     * @param directory     the directory we have analyzed.
     * @throws IOException  if something fails while writing the files.
     */
    private void printSummary(Map<String, Map<String, Map<Integer, Double>>> results, IntList list, File directory) throws IOException
    {
        // As a first step, we obtain the corresponding metric names.
        Set<String> metrics = new HashSet<>();
        results.forEach((alg, map) -> metrics.addAll(map.keySet()));
        List<String> metricNames = new ArrayList<>(metrics);

        // Second, we create the directory where we want to print the metric values.
        String outputDir = directory.getAbsolutePath() + File.separator + "metrics" + File.separator;
        File outputFolder = new File(outputDir);
        if(!outputFolder.exists())
        {
            boolean check = outputFolder.mkdir();
            if(!check)
            {
                System.err.println("ERROR: Program could not create directory " + outputDir);
                return;
            }
        }

        // And, finally, we do the printing. One file for each metric:
        for (String metricName : metricNames)
        {
            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDir + metricName + ".txt"))))
            {
                // Write the header: algorithm\tmetric at point 1\tmetric at point 2\t....
                bw.write("algorithm");
                for (int point : list)
                {
                    bw.write("\t" + point);
                }
                bw.write("\n");

                // For each algorithm:
                for (String alg : results.keySet())
                {
                    bw.write(alg);
                    Map<String, Map<Integer, Double>> map = results.get(alg);
                    if (!map.containsKey(metricName))
                    {
                        for (int ignored : list)
                        {
                            bw.write("\t-");
                        }
                    }
                    else
                    {
                        for (int point : list)
                        {
                            if (map.get(metricName).containsKey(point))
                            {
                                bw.write("\t" + map.get(metricName).get(point));
                            }
                            else
                            {
                                bw.write("\t-");
                            }
                        }
                    }

                    bw.write("\n");
                }
            }
        }
    }

    /**
     * Reads a file, and obtains, for some given iteration numbers, the values of the different metrics.
     * @param f the file to read.
     * @param list the list of iteration numbers to consider.
     * @return a map, indexed by metric, containing the values of the metric for the given point.
     * @throws IOException if something fails while reading the file.
     */
    private Map<String, Map<Integer, Double>> readFile(File f, IntList list) throws IOException
    {
        Map<String, Map<Integer, Double>> res = new HashMap<>();

        Map<String, Supplier<CumulativeMetric<U,I>>> metricsSupps = this.getMetrics();
        Map<String, CumulativeMetric<U,I>> metrics = new HashMap<>();
        metricsSupps.forEach((name, metric) ->
        {
            CumulativeMetric<U,I> m = metric.get();
            m.initialize(this.getDataset());
            metrics.put(name, m);
            res.put(name, new HashMap<>());
        });
        res.put(TIMENAME, new HashMap<>());

        Reader reader = ioSelector.getReader();
        reader.initialize(ioSelector.getInputStream(f.getAbsolutePath()));

        int i = 0;
        Tuple3<Integer, FastRecommendation, Long> triplet;
        long cumTime = 0;
        while((triplet = reader.readIteration()) != null)
        {
            int numIter = triplet.v1;
            FastRecommendation fastRec = triplet.v2;
            long time = triplet.v3;

            metrics.values().forEach(metric -> metric.update(fastRec));

            int finalI = i;
            if (numIter == list.get(i))
            {
                metrics.forEach((name, metric) -> res.get(name).put(list.get(finalI), metric.compute()));
                res.get(TIMENAME).put(list.get(finalI), cumTime+0.0);
                cumTime = 0;
                ++i;
            }

            cumTime += time;
        }

        reader.close();
        System.out.println("Finished reading file " + f.getAbsolutePath());
        return res;
    }

    /**
     * Obtains the dataset.
     * @return the dataset used during the validation.
     */
    protected abstract Dataset<U,I> getDataset();

    /**
     * Obtains the metrics.
     * @return a map with supplier for the metrics.
     */
    protected abstract Map<String, Supplier<CumulativeMetric<U,I>>> getMetrics();
}
