/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main;

import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
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
public class OutputResumer
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
        if (args.length < 2)
        {
            System.err.println("ERROR: Invalid arguments");
            System.err.println("Input directory: directory containing the recommendation files");
            System.err.println("Points: comma separated time points to consider");
        }

        String inputDirectory = args[0];
        String timePoints = args[1];

        // Obtain the number of elements in the list.
        IntList list = new IntArrayList();
        String[] split = timePoints.split(",");
        for (String num : split)
        {
            list.add(Parsers.ip.parse(num));
        }
        list.sort(Comparator.naturalOrder());

        boolean recursive;
        if (args.length > 2)
        {
            recursive = args[2].equalsIgnoreCase("-r");
        }
        else
        {
            recursive = false;
        }

        File directory = new File(inputDirectory);
        if (!directory.isDirectory())
        {
            System.err.println("ERROR: " + inputDirectory + " is not a directory");
            return;
        }

        readDirectory(directory, list, recursive);
    }

    /**
     * Reads a directory, resuming everything there.
     *
     * @param directory the name of the directory.
     * @param list      the number of iterations to measure.
     * @param recursive the recursive values.
     */
    private static void readDirectory(File directory, IntList list, boolean recursive) throws IOException
    {
        long a = System.currentTimeMillis();
        System.out.println("Entered directory " + directory);

        File[] files = directory.listFiles();
        if (files == null)
        {
            System.err.println("Nothing found in directory " + directory);
            return;
        }

        List<File> indivFiles = new ArrayList<>();
        List<File> directories = new ArrayList<>();

        for (File file : files)
        {
            if (file.isDirectory() && recursive)
            {
                directories.add(file);
            }
            else if (!file.getName().equals("rngSeed"))
            {
                indivFiles.add(file);
            }
        }

        // First, read the files.
        Map<String, Map<String, Map<Integer, Double>>> results = new HashMap<>();
        for (File f : indivFiles)
        {
            Map<String, Map<Integer, Double>> map = readFile(f, list);
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

        for (File dir : directories)
        {
            readDirectory(dir, list, recursive);
        }

        long b = System.currentTimeMillis();
        System.out.println("Exited directory " + directory + " (" + (b - a) + " ms.)");
    }

    private static Map<String, Map<Integer, Double>> readFile(File f, IntList list) throws IOException
    {
        Map<String, Map<Integer, Double>> res = new HashMap<>();
        Map<Integer, String> mapping = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f))))
        {
            String header = br.readLine();
            String[] split = header.split("\t");
            int len = split.length;

            for (int i = 3; i < len - 1; ++i)
            {
                String metricName = split[i];
                res.put(metricName, new Int2DoubleOpenHashMap());
                mapping.put(i, metricName);
            }

            String line;
            int listSize = list.size();
            int i = 0;
            int numIter = 0;
            while ((line = br.readLine()) != null && i < listSize)
            {
                String[] lineSplit = line.split("\t");
                numIter++;
                if (numIter == list.get(i))
                {
                    for (int j = 3; j < len - 1; ++j)
                    {
                        String metricName = mapping.get(j);
                        res.get(metricName).put(list.get(i), Double.parseDouble(lineSplit[j]));
                    }
                    ++i;
                }
            }
        }

        return res;
    }

}
