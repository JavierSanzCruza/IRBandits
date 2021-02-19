/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.others;

import es.uam.eps.ir.knnbandit.data.datasets.reader.LogRegister;
import es.uam.eps.ir.knnbandit.data.datasets.reader.SimpleStreamDatasetReader;
import es.uam.eps.ir.knnbandit.data.datasets.reader.StreamDatasetReader;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AdditiveRatingFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.ranksys.core.preference.IdPref;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

/**
 * Given the Yahoo! R6B dataset, this obtains a subsample which might be used
 * for evaluating recommendation approaches.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class YahooR6BSimplifier
{
    /**
     * The program for obtaining a subsample of the Yahoo! R6B dataset.
     * @param args Execution arguments:
     * <ol>
     *  <li><b>Input:</b> the .tgz file containing the dataset.</li>
     *  <li><b>Output:</b> a directory for storing the reduced dataset, indexes and statistics.</li>
     *  <li><b>Threshold: </b> the minimum number of times a user has to appear in the original log to be considered</li>
     * </ol>
     * Optional arguments:
     * <ul>
     *  <li><b>--files file1,file2,...</b> a comma-separated list of the files in the dataset to consider for this sampling</li>
     *  <li><b>--userdatafile name</b> a file containing the statistics for each user. If not available, it is computed.</li>
     * </ul>
     * @throws IOException if something fails while reading/writing.
     */
    public static void main(String[] args) throws IOException
    {
        // Read the arguments of the program.
        if(args.length < 2)
        {
            System.err.println("ERROR: Invalid arguments");
            System.err.println("Arguments");
            System.err.println("\tInput: the .tgz file containing the dataset.");
            System.err.println("\tOutput: the directory in which we want to store the reduced dataset and statistics.");
            System.err.println("\tThreshold: the minimum number of times that a user has to appear to be considered.");
            System.err.println("Optional arguments:");
            System.err.println("--files file1,file2,... : the subset of the files to consider for the reduction");
            System.err.println("--userdataFile : a file containing the statistics for each user. Otherwise, it is computed");
        }

        String input = args[0];
        String outputDir = args[1];
        long threshold = Parsers.lp.parse(args[2]);
        Set<String> files = new HashSet<>();
        String userDataFile = null;
        String fileList = "";

        for(int i = 3; i < args.length; ++i)
        {
            if(args[i].equalsIgnoreCase("--files"))
            {
                ++i;
                fileList = args[i];
                String[] split = args[i].split(",");
                files.addAll(Arrays.asList(split));
            }
            else if(args[i].equalsIgnoreCase("--userdatafile"))
            {
                ++i;
                userDataFile = args[i];
            }
        }

        // Step 1: if userDataFile == null, then, we have to compute the statistics:
        System.out.println("Obtaining the whole dataset statistics");
        long a = System.currentTimeMillis();
        if(userDataFile == null)
        {
            String[] auxArgs;
            if(files.isEmpty())
            {
                auxArgs = new String[]{input, outputDir + "full"};
            }
            else
            {
                auxArgs = new String[]{input, outputDir + "full", "--files", fileList};
            }

            YahooR6BAnalyzer.main(auxArgs);
            userDataFile = outputDir + "full" + YahooR6BAnalyzer.USERDATAFILE;
        }
        long b = System.currentTimeMillis();
        System.out.println("Obtained the whole dataset statistics (" + (b-a) + " ms.)");

        // Step 2: determine the valid set of users:
        System.out.println("Determining the valid set of users");
        FastUpdateableUserIndex<String> uIndex = new SimpleFastUpdateableUserIndex<>();
        try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(userDataFile))))
        {
            String line = br.readLine();
            while((line = br.readLine()) != null)
            {
                String[] split = line.split("\\s+");
                String user = split[0];
                double numRatings = Parsers.dp.parse(split[1]);

                if(numRatings >= threshold)
                {
                    uIndex.addUser(user);
                }
            }
        }
        b = System.currentTimeMillis();
        System.out.println("Determined the valid set of " + uIndex.numUsers() + " users (" + (b-a) + "ms.)");


        // Initialize the variables:
        FastUpdateableItemIndex<String> iIndex = new SimpleFastUpdateableItemIndex<>();

        // Now, we want to generate an individual file containing all the reduced log
        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDir + "log.txt"))))
        {
            // Open the .tgz file:
            GzipCompressorInputStream gzipIn = new GzipCompressorInputStream(new FileInputStream(input));
            TarArchiveInputStream tarIn = new TarArchiveInputStream(gzipIn);

            TarArchiveEntry entry;

            System.out.println("Starting writing the log file");
            while((entry = (TarArchiveEntry) tarIn.getNextEntry()) != null)
            {
                String name = entry.getName();
                if(!name.equals("README.txt") && (files.isEmpty() || files.contains(name)))
                {
                    b = System.currentTimeMillis();
                    System.out.println("Processing file " + name + " (" + (b-a) + " ms.)");
                    BufferedReader br = new BufferedReader(new InputStreamReader(new GZIPInputStream(tarIn)));
                    {
                        String line;
                        while((line = br.readLine()) != null)
                        {
                            String[] split = line.split("\\s+");
                            String item = split[1];
                            List<String> items = new ArrayList<>();
                            double rating = Parsers.dp.parse(split[2]);

                            boolean isCurrentUser = false;
                            boolean isEmptyUser = true;
                            char[] user = new char[135];

                            for(int i = 0; i < 134; ++i) user[i] = '0';

                            for (int i = 3; i < split.length; ++i)
                            {
                                if (split[i].equals("|user"))
                                {
                                    isCurrentUser = true;
                                }
                                else if (split[i].startsWith("|"))
                                {
                                    isCurrentUser = false;
                                    String itemId = split[i].substring(1);
                                    items.add(itemId);
                                }
                                else if (isCurrentUser)
                                {
                                    int index = Parsers.ip.parse(split[i]);
                                    if (index > 1)
                                    {
                                        isEmptyUser = false;
                                        user[index - 2] = '1';
                                    }
                                }
                            }

                            if (!isEmptyUser)
                            {
                                String userId = new String(user);
                                if(uIndex.containsUser(userId))
                                {
                                    int uidx = uIndex.user2uidx(userId);
                                    int iidx = iIndex.addItem(item);

                                    bw.write(uidx + "\t" + iidx + "\t" + rating);
                                    for(String itemId : items)
                                    {
                                        bw.write("\t" + iIndex.addItem(itemId));
                                    }
                                    bw.write("\n");
                                }
                            }
                       }
                    }

                    b = System.currentTimeMillis();
                    System.out.println("Finished processing file " + name + " (" + (b-a) + " ms.)");
                }
            }
            b = System.currentTimeMillis();
            System.out.println("Finished writing the log file (" + (b-a) + " ms.)");
        }
        catch(IOException ioe)
        {
            System.err.println("ERROR: something happened while reading/writing the log file.");
        }

        // Now, we print the user index:
        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDir + YahooR6BAnalyzer.USERFILE))))
        {
            int numUsers = uIndex.numUsers();
            for(int i = 0; i < numUsers; ++i)
            {
                bw.write(i + "\n");
            }
        }

        // Now, we print the item index:
        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDir + YahooR6BAnalyzer.ITEMFILE))))
        {
            int numUsers = iIndex.numItems();
            for(int i = 0; i < numUsers; ++i)
            {
                bw.write(i + "\n");
            }
        }

        FastUpdateableUserIndex<Integer> userIndex = SimpleFastUpdateableUserIndex.load(IntStream.range(0, uIndex.numUsers()).boxed());
        FastUpdateableItemIndex<Integer> itemIndex = SimpleFastUpdateableItemIndex.load(IntStream.range(0, iIndex.numItems()).boxed());
        AdditiveRatingFastUpdateablePreferenceData<Integer, Integer> numTimes = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), userIndex, itemIndex);
        AdditiveRatingFastUpdateablePreferenceData<Integer, Integer> numPos = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), userIndex, itemIndex);
        StreamDatasetReader<Integer, Integer> streamReader = new SimpleStreamDatasetReader<>(outputDir + "log.txt", Parsers.ip, Parsers.ip, "\t");
        streamReader.initialize();

        while(!streamReader.hasEnded())
        {
            LogRegister<Integer, Integer> register = streamReader.readRegister();
            if(register != null)
            {
                int uidx = register.getUser();
                int iidx = register.getFeaturedItem();
                double reward = register.getRating();

                numTimes.update(uidx, iidx, 1.0);
                numPos.update(uidx, iidx, reward);
            }
        }

        // Now, we print the whole statistics file.
        // STEP 1: the user index file
        try(BufferedWriter bwUserData = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDir + YahooR6BAnalyzer.USERDATAFILE))))
        {
            bwUserData.write("userId\tnumRatings\tnumRel\tnumRep\tnumRepRel\n");
            userIndex.getAllUsers().forEach(user ->
            {
                try
                {
                    bwUserData.write(user + "");
                    if(numTimes.numItems(user) > 0)
                    {
                        Pair<Double> pair1 = numTimes.getUserPreferences(user).map(pref -> new Pair<>(pref.v2, pref.v2 - 1)).reduce(new Pair<>(0.0,0.0), (x,y) -> new Pair<>(x.v1()+y.v1(), x.v2()+y.v2()));
                        Pair<Double> pair2 = numPos.getUserPreferences(user).map(pref -> new Pair<>(pref.v2, pref.v2 > 1 ? pref.v2 - 1 : 0)).reduce(new Pair<>(0.0,0.0), (x,y) -> new Pair<>(x.v1()+y.v1(), x.v2()+y.v2()));
                        bwUserData.write("\t" + pair1.v1() + "\t" + pair2.v1() + "\t" + pair1.v2() + "\t" + pair2.v2() + "\n");
                    }
                    else
                    {
                        bwUserData.write("\t0\t0\t0\n");
                    }
                }
                catch(IOException ioe)
                {
                    System.err.println("Something wrong occurred");
                }
            });
        }

        // STEP 2: the item index files
        try(BufferedWriter bwItemData = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDir + YahooR6BAnalyzer.ITEMDATAFILE))))
        {
            bwItemData.write("itemId\tnumRatings\tnumRel\tnumRep\tnumRepRel\n");
            itemIndex.getAllItems().forEach(item ->
            {
                try
                {
                    bwItemData.write(item + "");
                    if(numTimes.numUsers(item) > 0)
                    {
                        Pair<Double> pair1 = numTimes.getItemPreferences(item).map(pref -> new Pair<>(pref.v2, pref.v2 - 1)).reduce(new Pair<>(0.0,0.0), (x,y) -> new Pair<>(x.v1()+y.v1(), x.v2()+y.v2()));
                        Pair<Double> pair2 = numPos.getItemPreferences(item).map(pref -> new Pair<>(pref.v2, pref.v2 > 1 ? pref.v2 - 1 : 0)).reduce(new Pair<>(0.0,0.0), (x,y) -> new Pair<>(x.v1()+y.v1(), x.v2()+y.v2()));
                        bwItemData.write("\t" + pair1.v1() + "\t" + pair2.v1() + "\t" + pair1.v2() + "\t" + pair2.v2() + "\n");
                    }
                    else
                    {
                        bwItemData.write("\t0\t0\t0\n");
                    }
                 }
                 catch(IOException ioe)
                 {
                     System.err.println("Something wrong occurred");
                 }
             });
        }

        // STEP 3: the ratings file
        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDir + YahooR6BAnalyzer.RATINGSFILE))))
        {
            bw.write("userId\titemId\tnumPos\tnumTimes\tCTR\n");
            numTimes.getUsersWithPreferences().forEach(user -> numTimes.getUserPreferences(user).forEach(item ->
            {
                try
                {
                    Integer i = item.v1();
                    Optional<? extends IdPref<Integer>> opt = numPos.getPreference(user, i);
                    if (opt.isPresent() && opt.get().v2 > 0.0)
                    {
                        bw.write(user + "\t" + i + "\t" + opt.get().v2() + "\t" + item.v2() + "\t" + (opt.get().v2/item.v2()) + "\n");
                    }
                    else
                    {
                        bw.write(user + "\t" + i + "\t" + 0.0 + "\t" + item.v2() + "\t" + 0 + "\n");
                    }
                }
                catch(IOException ioe)
                {
                    System.err.println("Something wrong occurred");
                }
            })
            );
        }
    }
}
