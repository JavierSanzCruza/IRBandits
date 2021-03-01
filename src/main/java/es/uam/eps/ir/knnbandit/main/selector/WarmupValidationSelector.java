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

import es.uam.eps.ir.knnbandit.io.IOType;
import es.uam.eps.ir.knnbandit.main.WarmupValidation;
import es.uam.eps.ir.knnbandit.main.contact.ContactWarmupValidation;
import es.uam.eps.ir.knnbandit.main.general.GeneralWarmupValidation;
import es.uam.eps.ir.knnbandit.main.withknowledge.WithKnowledgeWarmupValidation;
import es.uam.eps.ir.knnbandit.partition.Partition;
import es.uam.eps.ir.knnbandit.partition.RelevantPartition;
import es.uam.eps.ir.knnbandit.partition.UniformPartition;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import es.uam.eps.ir.knnbandit.warmup.WarmupType;
import org.ranksys.formats.parsing.Parsers;
import static es.uam.eps.ir.knnbandit.main.selector.DatasetType.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.function.Supplier;

/**
 * Main class for determining the type of dataset / execution we are using
 * for applying validation when some warmup is available.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class WarmupValidationSelector
{
    /**
     * Executes the validation.
     * @param type the type of the dataset we are using.
     * @param args the execution arguments.
     * @throws IOException if something fails while reading/writing
     * @throws UnconfiguredException if the algorithm selector is badly configured.
     */
    public void validate(String type, String[] args) throws IOException, UnconfiguredException
    {
        String[] execArgs;
        String errorString = this.getErrorMessage(type);

        int firstIndex;
        int length;
        int lastIndex;

        switch(type)
        {
            case GENERAL:
                length = 12;
                firstIndex = 1;
                lastIndex = 11;
                break;
            case CONTACT:
                length = 11;
                firstIndex = 0;
                lastIndex = 11;
                break;
            case KNOWLEDGE:
                length = 12;
                firstIndex = 0;
                lastIndex = 12;
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
        String algorithms = execArgs[0];
        String input = execArgs[1];
        String output = execArgs[2];
        Supplier<EndCondition> endCond = EndConditionSelector.select(Parsers.dp.parse(execArgs[3]));
        boolean resume = execArgs[4].equalsIgnoreCase("true");
        int k = 1;
        int cutoff = 1;
        WarmupType warmup = WarmupType.FULL;
        IOType iotype = IOType.TEXT;
        boolean gzipped = false;
        for (int i = lastIndex; i < execArgs.length; ++i)
        {
            if ("-k".equals(args[i]))
            {
                ++i;
                k = Parsers.ip.parse(args[i]);
            }
            else if("-type".equals(args[i]))
            {
                ++i;
                warmup = WarmupType.fromString(args[i]);
            }
            else if("-cutoff".equals(args[i]))
            {
                ++i;
                cutoff = Parsers.ip.parse(args[i]);
            }
            else if("-io-type".equals(args[i]))
            {
                ++i;
                iotype = IOType.fromString(args[i]);
            }
            else if("--gzipped".equals(args[i]))
            {
                gzipped = true;
            }
        }

        String training = execArgs[5];
        String testType = execArgs[6];
        int auxNumParts = Parsers.ip.parse(execArgs[7]);
        double percTrain = Parsers.dp.parse(execArgs[8]);
        int numParts = Math.abs(auxNumParts);
        Partition partition = (auxNumParts > 0) ? new UniformPartition() : new RelevantPartition();

        switch(type)
        {
            case GENERAL:
            {
                double threshold = Parsers.dp.parse(execArgs[9]);
                boolean useRatings = execArgs[10].equalsIgnoreCase("true");
                if(args[0].equalsIgnoreCase("movielens"))
                {
                    WarmupValidation<Long, Long> valid = new GeneralWarmupValidation<>(input, "::", Parsers.lp, Parsers.lp, threshold, useRatings, warmup, cutoff, iotype, gzipped);
                    valid.validate(algorithms, output, endCond, resume, training, partition, testType, numParts, percTrain, k);
                }
                else if(args[0].equalsIgnoreCase("foursquare"))
                {
                    WarmupValidation<Long, String> valid = new GeneralWarmupValidation<>(input, "::", Parsers.lp, Parsers.sp, threshold, useRatings, warmup, cutoff, iotype, gzipped);
                    valid.validate(algorithms, output, endCond, resume, training, partition, testType, numParts, percTrain, k);
                }
                break;
            }
            case CONTACT:
            {
                boolean directed = execArgs[9].equalsIgnoreCase("true");
                boolean notReciprocal = execArgs[10].equalsIgnoreCase("true");

                WarmupValidation<Long, Long> valid = new ContactWarmupValidation<>(input, "\t", Parsers.lp, directed, notReciprocal, warmup, cutoff, iotype, gzipped);
                valid.validate(algorithms, output, endCond, resume, training, partition, testType, numParts, percTrain, k);

                break;
            }
            case KNOWLEDGE:
            {
                double threshold = Parsers.dp.parse(execArgs[9]);
                boolean useRatings = execArgs[10].equalsIgnoreCase("true");
                KnowledgeDataUse dataUse = KnowledgeDataUse.fromString(execArgs[11]);

                WarmupValidation<Long, Long> valid = new WithKnowledgeWarmupValidation<>(input, "::", Parsers.lp, Parsers.lp, threshold, useRatings, dataUse, warmup, cutoff, iotype, gzipped);
                valid.validate(algorithms, output, endCond, resume, training, partition, testType, numParts, percTrain, k);
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
        builder.append("\tAlgorithms: JSON file containing the recommender list\n");
        builder.append("\tInput: preference data input\n");
        builder.append("\tOutput: folder in which to store the output\n");
        builder.append("\tNum. Iter.: number of iterations.\n");
        builder.append("\t\t0 : run until no recommendable item is left\n");
        builder.append("\t\t[0,1] : percentage of the positive ratings to retrieve\n");
        builder.append("\t\t[1,infty] : maximum number of iterations\n");
        builder.append("\tResume: true if we want to resume previous executions, false if we want to overwrite\n");
        builder.append("\tWarmup: file containing the warmup information\n");
        builder.append("\tNum. parts: the number of parts in which we want to divide the warmup. If negative, the partition will divide considering only relevant ratings");
        builder.append("\tTrain percentage: the fraction of the warmup which shall be used as training");
        switch (type)
        {
            case GENERAL:
                builder.append("\tThreshold: true if the graph is directed, false otherwise\n");
                builder.append("\tUse ratings: true if we want to recommend reciprocal edges, false otherwise\n");
                break;
            case CONTACT:
                builder.append("\tDirected: true if the graph is directed, false otherwise\n");
                builder.append("\tNot Reciprocal: true if we want to recommend reciprocal edges, false otherwise\n");
                break;
            case KNOWLEDGE:
                builder.append("\tThreshold: true if the graph is directed, false otherwise\n");
                builder.append("\tUse ratings: true if we want to recommend reciprocal edges, false otherwise\n");
                builder.append("\tKnowledge: ALL if we want to use all the ratings, KNOWN if we want to use only the known ones, UNKNOWN otherwise\n");
                break;
            case STREAM:
            default:
        }

        builder.append("Optional arguments:\n");
        builder.append("\t-k value : The number of times each individual approach has to be executed (by default: 1)");
        builder.append("\t-cutoff value : The number of items to recommend on each iteration (by default: 1)");

        return builder.toString();
    }
}
