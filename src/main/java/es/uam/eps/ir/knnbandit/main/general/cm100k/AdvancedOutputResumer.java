/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.general.cm100k;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.utils.statistics.GiniIndex;
import es.uam.eps.ir.knnbandit.utils.statistics.GiniIndex2;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2LongOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.*;

/**
 * Class for summarizing the outcomes of recommenders.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class AdvancedOutputResumer
{
    /**
     * Program that summarizes the outcomes of recommenders.
     *
     * @param args Execution arguments:
     *             <ol>
     *                  <li><b>Input directory:</b> Directory containing the recommendation outputs</li>
     *                  <li><b>Points: </b> A list of comma separated time points to consider</li>
     *                  <li><b>Recursive</b> (optional) <b>:</b> If present (as flag -r), checks internal folders.</li>
     *             </ol>
     */
    public static void main(String[] args) throws IOException
    {
        if(args.length < 3)
        {
            System.err.println("Dataset: file containing the whole set of ratings");
            System.err.println("Threshold: relevance threshold");
            System.err.println("Input directory: directory containing the recommendation files");
            System.err.println("Points: comma separated time points to consider");
            return;
        }

        String data = args[0];
        double threshold = Parsers.dp.parse(args[1]);
        String inputDir = args[2];
        String timePoints = args[3];

        // First, read the dataset.
        Dataset<Long, Long> dataset = Dataset.load(data, Parsers.lp, Parsers.lp, "\t", (double x) -> x, (double x) -> x >= threshold);
        System.out.println("Read the whole data");
        System.out.println(dataset.toString());

        // Then, obtain the time points:
        IntList list = new IntArrayList();
        String[] split = timePoints.split(",");
        for (String num : split)
        {
            list.add(Parsers.ip.parse(num));
        }
        list.sort(Comparator.naturalOrder());

        boolean recursive = args.length > 4 && (args[4].equalsIgnoreCase("-r"));
        File directory = new File(inputDir);
        if(!directory.isDirectory())
        {
            System.err.println("ERROR: " + inputDir + " is not a directory");
            return;
        }

        readDirectory(directory, list, recursive, dataset);
    }

    /**
     * Reads a directory, resuming everything there.
     *
     * @param directory the name of the directory.
     * @param list      the number of iterations to measure.
     * @param recursive the recursive values.
     */
    private static void readDirectory(File directory, IntList list, boolean recursive, Dataset<Long, Long> dataset) throws IOException
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
            Map<String, Map<Integer, Double>> map = readFile(f, list, dataset);
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
                        for (int point : list)
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
                readDirectory(dir, list, true, dataset);
            }
        }

        long b = System.currentTimeMillis();
        System.out.println("Exited directory " + directory + " (" + (b - a) + " ms.)");
    }

    private static Map<String, Map<Integer, Double>> readFile(File f, IntList list, Dataset<Long,Long> dataset) throws IOException
    {
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
}
