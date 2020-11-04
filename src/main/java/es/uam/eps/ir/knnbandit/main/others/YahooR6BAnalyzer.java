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
import java.util.Arrays;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

/**
 * Given the Yahoo! R6B dataset, this class is used for analyzing the different
 * properties of the dataset.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class YahooR6BAnalyzer
{
    /**
     * The name of the file containing a relation of the users.
     * Format of this file:
     *
     * <p>user1 \n</p>
     * <p>user2 \n</p>
     * <p>...</p>
     * <p>userN</p>
     */
    public final static String USERFILE = "user.txt";
    /**
     * The name of the file containing information about the different users.
     * Format of this file (without header - the definitive file contains a header):
     *
     * <p>user1 \t numRatings1 \t numRel1 \t numRepeated1 \t numRelevantRepeated1 \n</p>
     * <p>user2 \t numRatings2 \t numRel2 \t numRepeated2 \t numRelevantRepeated2 \n</p>
     * <p>...</p>
     * <p>userN \t numRatingsN \t numRelN \t numRepeatedN \t numRelevantRepeatedN \n</p>
     */
    public final static String USERDATAFILE = "userdata.txt";
    /**
     * The name of the file containing a relation of the items.
     * Format of this file:
     *
     * <p>item1 \n</p>
     * <p>item2 \n</p>
     * <p>...</p>
     * <p>itemN</p>
     */
    public final static String ITEMFILE = "item.txt";
    /**
     * The name of the file containing information about the different items.
     * Format of this file (without header - the definitive file contains a header):
     *
     * <p>item1 \t numRatings1 \t numRel1 \t numRepeated1 \t numRelevantRepeated1 \n</p>
     * <p>item2 \t numRatings2 \t numRel2 \t numRepeated2 \t numRelevantRepeated2 \n</p>
     * <p>...</p>
     * <p>itemN \t numRatingsN \t numRelN \t numRepeatedN \t numRelevantRepeatedN \n</p>
     */
    public final static String ITEMDATAFILE = "itemdata.txt";
    /**
     * The name of the file containing the ratings for each explored user / item pair.
     * Format of this file (without header - the definitive file contains a header):
     *
     * <p>user1 \t item1 \t numPositiveRatings1 \t numRatings1 \t Clicktrough rate1 \n</p>
     * <p>user2 \t item2 \t numPositiveRatings2 \t numRatings2 \t Clicktrough rate2 \n</p>
     * <p>...</p>
     * <p>userN \t itemN \t numPositiveRatingsN \t numRatingsN \t Clicktrough rateN \n</p>
     */
    public final static String RATINGSFILE = "ratings.txt";

    /**
     * Program that analyzes the information.
     * @param args Execution arguments:
     * <ol>
     *  <li><b>Input:</b> the .tgz file containing the dataset.</li>
     *  <li><b>Output:</b> a directory for storing the reduced dataset, indexes and statistics.</li>
     * </ol>
     * Optional arguments:
     * <ul>
     *  <li><b>--files file1,file2,...</b> a comma-separated list of the files in the dataset to consider for this sampling</li>
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
            System.err.println("Optional arguments:");
            System.err.println("--files file1,file2,... : the subset of the files to consider for the reduction");
        }

        String input = args[0];
        String outputDir = args[1];
        Set<String> files = new HashSet<>();
        for(int i = 2; i < args.length; ++i)
        {
            if(args[i].equalsIgnoreCase("--files"))
            {
                ++i;
                String[] split = args[i].split(",");
                files.addAll(Arrays.asList(split));
            }
        }

        GzipCompressorInputStream gzipIn = new GzipCompressorInputStream(new FileInputStream(input));
        TarArchiveInputStream tarIn = new TarArchiveInputStream(gzipIn);

        FastUpdateableUserIndex<String> uIndex = new SimpleFastUpdateableUserIndex<>();
        FastUpdateableItemIndex<String> iIndex = new SimpleFastUpdateableItemIndex<>();
        AdditiveRatingFastUpdateablePreferenceData<String, String> numTimes = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        AdditiveRatingFastUpdateablePreferenceData<String, String> numPos = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);

        System.out.println("Reading the log");
        long a = System.currentTimeMillis();
        TarArchiveEntry entry;
        while((entry = (TarArchiveEntry) tarIn.getNextEntry()) != null)
        {
            String name = entry.getName();
            if(!name.equals("README.txt") && (files.isEmpty() || files.contains(name)))
            {
                System.out.println("Processing the file " + name);
                BufferedReader br = new BufferedReader(new InputStreamReader(new GZIPInputStream(tarIn)));
                {
                    br.lines().forEach(line ->
                    {
                        String[] split = line.split("\\s+");
                        String item = split[1];
                        numTimes.addItem(item);
                        numPos.addItem(item);
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
                                numTimes.addItem(itemId);
                                numPos.addItem(itemId);
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
                            numTimes.addUser(userId);
                            numPos.addUser(userId);

                            numTimes.update(userId, item, 1.0);
                            numPos.update(userId, item, rating);
                        }

                    });
                }
            }
        }
        long b = System.currentTimeMillis();
        System.out.println("Finished reading the log (" + (b-a) + " ms.)");

        // now, print everything:
        // STEP 1: the user index file
        System.out.println("Writing the user files");

        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[1] + USERFILE)));
            BufferedWriter bwUserData = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[1] + USERDATAFILE))))
        {
            bwUserData.write("userId\tnumRatings\tnumRel\tnumRep\tnumRepRel\n");
            uIndex.getAllUsers().forEach(user ->
            {
                try
                {
                    bw.write(user + "\n");
                    bwUserData.write(user);
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
        b = System.currentTimeMillis();
        System.out.println("Finished writing the user files (" + (b-a) + " ms.)");

        // STEP 2: the item index files
        System.out.println("Writing the item files");
        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[1] + ITEMFILE)));
            BufferedWriter bwUserData = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[1] + ITEMDATAFILE))))
        {
            bwUserData.write("itemId\tnumRatings\tnumRel\tnumRep\tnumRepRel\n");
            iIndex.getAllItems().forEach(item ->
            {
                try
                {
                    bw.write(item + "\n");
                    bwUserData.write(item);
                    if(numTimes.numUsers(item) > 0)
                    {
                        Pair<Double> pair1 = numTimes.getItemPreferences(item).map(pref -> new Pair<>(pref.v2, pref.v2 - 1)).reduce(new Pair<>(0.0,0.0), (x,y) -> new Pair<>(x.v1()+y.v1(), x.v2()+y.v2()));
                        Pair<Double> pair2 = numPos.getItemPreferences(item).map(pref -> new Pair<>(pref.v2, pref.v2 > 1 ? pref.v2 - 1 : 0)).reduce(new Pair<>(0.0,0.0), (x,y) -> new Pair<>(x.v1()+y.v1(), x.v2()+y.v2()));
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
        b = System.currentTimeMillis();
        System.out.println("Finished writing the item files (" + (b-a) + " ms.)");

        // STEP 3: the ratings file
        System.out.println("Writing the ratings file");
        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[1] + RATINGSFILE))))
        {
            bw.write("userId\titemId\tnumPos\tnumTimes\tCTR\n");
            numTimes.getUsersWithPreferences().forEach(user ->
                numTimes.getUserPreferences(user).forEach(item ->
                {
                    try
                    {
                        String i = item.v1();
                        Optional<? extends IdPref<String>> opt = numPos.getPreference(user, i);
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
        b = System.currentTimeMillis();
        System.out.println("Finished writing the rating file (" + (b-a) + " ms.)");
    }

}
