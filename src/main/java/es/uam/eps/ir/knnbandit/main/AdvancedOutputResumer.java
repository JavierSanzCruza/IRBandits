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
import es.uam.eps.ir.knnbandit.utils.statistics.GiniIndex;
import es.uam.eps.ir.knnbandit.utils.statistics.GiniIndex2;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2LongOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.*;

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
     * Summarizes all the executions in a file.
     * @param input directory containing the recommendation files.
     * @param points a list of time points to consider.
     * @param recursive true if we want to make this recursive.
     */
    public void summarize(String input, IntList points, boolean recursive) throws IOException
    {
        Dataset<U,I> dataset = this.getDataset();
        System.out.println(dataset.toString());
        points.sort(Comparator.naturalOrder());

        File directory = new File(input);
        if(!directory.isDirectory())
        {
            System.err.println("ERROR: " + input + " is not a directory");
            return;
        }

        // Read the directory:
        readDirectory(directory, points, recursive);
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

        // Then, we first read the files:
        Map<String, Map<String, Map<Integer, Double>>> results = new HashMap<>();
        for(File f : indivFiles)
        {
            Map<String, Map<Integer, Double>> map = readFile(f, list);
            if(map != null)
                results.put(f.getName(), map);
        }

        Set<String> metrics = new HashSet<>();
        results.forEach((alg, map) -> metrics.addAll(map.keySet()));
        List<String> metricNames = new ArrayList<>(metrics);

        String outputDir = directory.getAbsolutePath() + File.separator + "metrics" + File.separator;
        File outputFolder = new File(outputDir);
        outputFolder.mkdir();

        for (String metricName : metricNames)
        {
            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDir + metricName + ".txt"))))
            {
                bw.write("algorithm");
                for (int point : list)
                {
                    bw.write("\t" + point);
                }
                bw.write("\n");
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

        if(recursive)
        {
            for (File dir : directories)
            {
                readDirectory(dir, list, true);
            }
        }
        long b = System.currentTimeMillis();
        System.out.println("Exited directory " + directory + " (" + (b - a) + " ms.)");
    }

    private Map<String, Map<Integer, Double>> readFile(File f, IntList list) throws IOException
    {
        Dataset<U,I> dataset = this.getDataset();
        int numItems = dataset.numItems();
        Map<String, Map<Integer, Double>> res = new HashMap<>();
        Map<Integer, String> mapping = new HashMap<>();

        try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f))))
        {
            String header = br.readLine();
            String[] split = header.split("\t");
            if(split.length < 5)
            {
                System.err.println("Failure while reading file " + f.getAbsolutePath());
                return null;
            }

            int len = split.length;
            for(int i = 3; i < len; ++i)
            {
                String metricName = split[i];
                res.put(metricName, new Int2DoubleOpenHashMap());
                mapping.put(i, metricName);

            }

            res.put("auxIncrGini", new Int2DoubleOpenHashMap());
            mapping.put(len, "auxIncrGini");
            res.put("auxGini", new Int2DoubleOpenHashMap());
            mapping.put(len+1, "auxGini");

            Int2LongOpenHashMap freqs = new Int2LongOpenHashMap();
            freqs.defaultReturnValue(0);
            GiniIndex2 giniIndex = new GiniIndex2(dataset.numItems());

            for (int i = 3; i < len - 1; ++i)
            {
                String metricName = split[i];
                res.put(metricName, new Int2DoubleOpenHashMap());
                mapping.put(i, metricName);
            }

            String line;
            int numIter = 0;
            int i = 0;
            int listSize = list.size();

            while((line = br.readLine()) != null && i < listSize)
            {
                String[] lineSplit = line.split("\t");
                int iidx = Parsers.ip.parse(lineSplit[2]);

                freqs.addTo(iidx, 1);
                giniIndex.updateFrequency(iidx);

                numIter++;
                if (numIter == list.get(i))
                {
                    for (int j = 3; j < len - 1; ++j)
                    {
                        String metricName = mapping.get(j);
                        res.get(metricName).put(list.get(i), Double.parseDouble(lineSplit[j]));
                    }

                    String metricName = mapping.get(len);
                    res.get(metricName).put(list.get(i), giniIndex.getValue());
                    GiniIndex gi = new GiniIndex(numItems,freqs);
                    metricName = mapping.get(len+1);
                    res.get(metricName).put(list.get(i), gi.getValue());
                    ++i;
                }
            }
            System.err.println("Finished reading file " + f.getAbsolutePath());

            return res;
        }
    }

    /**
     * Obtains the dataset.
     * @return the dataset used during the validation.
     */
    protected abstract Dataset<U,I> getDataset();
}
