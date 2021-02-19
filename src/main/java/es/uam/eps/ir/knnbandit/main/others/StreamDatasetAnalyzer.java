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
import es.uam.eps.ir.knnbandit.data.datasets.reader.StreamCandidateSelectionDatasetReader;
import es.uam.eps.ir.knnbandit.data.datasets.reader.StreamDatasetReader;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AdditiveRatingFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.ranksys.core.preference.IdPref;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Given the Yahoo! R6B dataset, this obtains a subsample which might be used
 * for evaluating recommendation approaches.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class StreamDatasetAnalyzer
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
        String log = args[2];
        String userIndex = args[3];
        String itemIndex = args[4];


        // Step 2: determine the valid set of users:
        System.out.println("Determining the valid set of users");
        FastUpdateableUserIndex<Integer> uIndex = new SimpleFastUpdateableUserIndex<>();
        try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(userIndex))))
        {
            String line = br.readLine();
            while((line = br.readLine()) != null)
            {
                uIndex.addUser(Parsers.ip.parse(line));
            }
        }

        FastUpdateableItemIndex<Integer> iIndex = new SimpleFastUpdateableItemIndex<>();
        try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(itemIndex))))
        {
            String line = br.readLine();
            while((line = br.readLine()) != null)
            {
                iIndex.addItem(Parsers.ip.parse(line));
            }
        }

        AdditiveRatingFastUpdateablePreferenceData<Integer, Integer> numTimes = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        AdditiveRatingFastUpdateablePreferenceData<Integer, Integer> numPos = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        StreamDatasetReader<Integer, Integer> streamReader = new StreamCandidateSelectionDatasetReader<>(log, Parsers.ip, Parsers.ip, "\t");
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
            uIndex.getAllUsers().forEach(user ->
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
            iIndex.getAllItems().forEach(item ->
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
