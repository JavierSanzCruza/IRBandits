/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.selector;


import es.uam.eps.ir.knnbandit.main.contact.ContactTrainingStatistics;
import es.uam.eps.ir.knnbandit.main.general.GeneralTrainingStatistics;
import es.uam.eps.ir.knnbandit.main.withknowledge.WithKnowledgeTrainingStatistics;
import es.uam.eps.ir.knnbandit.partition.Partition;
import es.uam.eps.ir.knnbandit.partition.RelevantPartition;
import es.uam.eps.ir.knnbandit.partition.UniformPartition;
import es.uam.eps.ir.knnbandit.selector.io.IOSelector;
import es.uam.eps.ir.knnbandit.selector.io.IOType;
import es.uam.eps.ir.knnbandit.warmup.WarmupType;
import org.ranksys.formats.parsing.Parsers;

import java.io.IOException;
import java.util.Arrays;

import static es.uam.eps.ir.knnbandit.main.selector.DatasetType.*;

/**
 * Main class for summarizing the content of recommendation files.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class TrainingStatisticsSelector
{
    /**
     * Analyzes some statistics of the warmup data.
     * @param type the type of the dataset we are using.
     * @param args the execution arguments.
     *
     * @throws IOException if something fails while reading/writing
     */
    public void statistics(String type, String[] args) throws IOException
    {
        String[] execArgs;
        String errorString = this.getErrorMessage(type);

        int firstIndex;
        int length;
        int lastIndex;

        switch(type)
        {
            case GENERAL:
                length = 6;
                firstIndex = 1;
                lastIndex = 6;
                break;
            case CONTACT:
            case KNOWLEDGE:
                length = 5;
                firstIndex = 0;
                lastIndex = 7;
                break;
            case STREAM:
            default:
                length = args.length;
                firstIndex = 0;
                lastIndex = 0;
        }

        if(args.length < length)
        {
            System.err.println(errorString);
            return;
        }

        execArgs = Arrays.copyOfRange(args, firstIndex, args.length);
        String input = execArgs[0];
        String training = execArgs[1];

        int auxNumParts = Parsers.ip.parse(execArgs[2]);
        int numParts = Math.abs(auxNumParts);
        Partition partition = (auxNumParts > 0) ? new UniformPartition() : new RelevantPartition();
        double percTrain = Double.NaN;
        IOType warmupIotype = IOType.TEXT;
        boolean warmupGzipped = false;
        for (int i = lastIndex; i < execArgs.length; ++i)
        {
            if("-perctrain".equals(args[i]))
            {
                ++i;
                percTrain = Parsers.dp.parse(args[i]);
            }
            else if("-warmup-io-type".equals(args[i]))
            {
                ++i;
                warmupIotype = IOType.fromString(args[i]);
            }
            else if("--warmup-gzipped".equals(args[i]))
            {
                warmupGzipped = true;
            }
        }

        IOSelector ioSelector = new IOSelector(warmupIotype, warmupGzipped);
        switch(type)
        {
            case GENERAL:
            {
                double threshold = Parsers.dp.parse(execArgs[3]);
                boolean useRatings = execArgs[4].equalsIgnoreCase("true");

                if(args[0].equalsIgnoreCase("movielens"))
                {
                    GeneralTrainingStatistics<Long, Long> stats = new GeneralTrainingStatistics<>(input, "::", Parsers.lp, Parsers.lp, threshold, useRatings, ioSelector);
                    stats.statistics(training, partition, numParts, percTrain);
                }
                else if(args[0].equalsIgnoreCase("foursquare"))
                {
                    GeneralTrainingStatistics<Long, String> stats = new GeneralTrainingStatistics<>(input, "::", Parsers.lp, Parsers.sp, threshold, useRatings, ioSelector);
                    stats.statistics(training, partition, numParts, percTrain);
                }
                break;
            }
            case CONTACT:
            {
                boolean directed = execArgs[3].equalsIgnoreCase("true");
                boolean notReciprocal = execArgs[4].equalsIgnoreCase("true");

                ContactTrainingStatistics<Long> stats = new ContactTrainingStatistics<>(input, "\t", Parsers.lp, directed, notReciprocal, ioSelector);
                stats.statistics(training, partition, numParts, percTrain);

                break;
            }
            case KNOWLEDGE:
            {
                double threshold = Parsers.dp.parse(execArgs[3]);
                boolean useRatings = execArgs[4].equalsIgnoreCase("true");

                WithKnowledgeTrainingStatistics<Long, Long> stats = new WithKnowledgeTrainingStatistics<>(input, "::", Parsers.lp, Parsers.lp, threshold, useRatings, ioSelector);
                stats.statistics(training, partition, numParts, percTrain);
                break;

            }
            case STREAM:
            {
                break;
            }
            default:
        }
    }

    /**
     * Obtains the error message in case the parameters are wrong.
     * @param type the dataset type.
     * @return the error message.
     */
    private String getErrorMessage(String type)
    {
        StringBuilder builder = new StringBuilder();
        builder.append("ERROR: Invalid arguments\n");
        builder.append("Usage:\n");
        builder.append("\tInput: preference data input\n");
        builder.append("\tTraining: file containing the warmup recommendation data\n");
        builder.append("\tNum. parts: the number of parts. If negative, this is divided by positive ratings.\n");

        switch (type)
        {
            case GENERAL:
            case KNOWLEDGE:
                builder.append("\tThreshold: true if the graph is directed, false otherwise\n");
                builder.append("\tUse ratings: true if we want to recommend reciprocal edges, false otherwise\n");
                break;
            case CONTACT:
                builder.append("\tDirected: true if the graph is directed, false otherwise\n");
                builder.append("\tNot Reciprocal: true if we want to recommend reciprocal edges, false otherwise\n");
                break;
            case STREAM:
            default:
        }

        builder.append("Optional arguments:\n");
        builder.append("\t-perctrain perc : The percentage of the warm-up data to use as training (by default, it is splitted in equal parts");
        builder.append("\t-warmup-io-type : establishes the format of the warm-up files. Possible values:\n");
        builder.append("\t\tbinary : for binary files\n");
        builder.append("\t\ttext : for text files (default value)\n");
        builder.append("\t--warmup-gzipped : if the warm-up files are compressed (by default, they are not compressed)");
        return builder.toString();
    }

}
