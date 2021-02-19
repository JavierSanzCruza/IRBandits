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


import es.uam.eps.ir.knnbandit.main.contact.ContactDatasetGraphAnalysis;
import es.uam.eps.ir.knnbandit.main.general.GeneralDatasetGraphAnalysis;
import es.uam.eps.ir.knnbandit.main.withknowledge.WithKnowledgeDatasetGraphAnalysis;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.doubles.DoubleList;
import org.ranksys.formats.parsing.Parsers;

import java.io.IOException;
import java.util.Arrays;

import static es.uam.eps.ir.knnbandit.main.selector.DatasetType.*;

/**
 * Main class for finding a graph relating users if they have rated the same
 * item.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class DatasetGraphAnalysisSelector
{
    /**
     * Analyzes some statistics of the warmup data.
     * @param type the type of the dataset we are using.
     * @param args the execution arguments.
     *
     * @throws IOException if something fails while reading/writing
     */
    public void analyze(String type, String[] args) throws IOException
    {
        String[] execArgs;
        String errorString = this.getErrorMessage(type);

        int firstIndex;
        int length;

        switch(type)
        {
            case GENERAL:
                length = 7;
                firstIndex = 1;
                break;
            case CONTACT:
                length = 6;
                firstIndex = 0;
                break;
            case KNOWLEDGE:
                length = 7;
                firstIndex = 0;
                break;
            case STREAM:
            default:
                System.err.println(errorString);
                return;
        }

        if(args.length < length)
        {
            System.err.println(errorString);
            return;
        }

        execArgs = Arrays.copyOfRange(args, firstIndex, args.length);
        String input = execArgs[0];
        String output = execArgs[1];
        int limit = Parsers.ip.parse(execArgs[2]);
        DoubleList list = new DoubleArrayList();
        String[] split = execArgs[3].split(",");
        for(String weight : split)
        {
            list.add(Parsers.dp.parse(weight));
        }


        switch(type)
        {
            case GENERAL:
            {
                double threshold = Parsers.dp.parse(execArgs[2]);
                boolean useRatings = execArgs[3].equalsIgnoreCase("true");

                if(args[0].equalsIgnoreCase("movielens"))
                {
                    GeneralDatasetGraphAnalysis<Long, Long> stats = new GeneralDatasetGraphAnalysis<>(input, "::", Parsers.lp, Parsers.lp, threshold, useRatings);
                    stats.analyze(output, limit, list);
                }
                else if(args[0].equalsIgnoreCase("foursquare"))
                {
                    GeneralDatasetGraphAnalysis<Long, String> stats = new GeneralDatasetGraphAnalysis<>(input, "::", Parsers.lp, Parsers.sp, threshold, useRatings);
                    stats.analyze(output, limit, list);
                }
                break;
            }
            case CONTACT:
            {
                boolean directed = execArgs[2].equalsIgnoreCase("true");
                boolean notReciprocal = execArgs[3].equalsIgnoreCase("true");

                ContactDatasetGraphAnalysis<Long> stats = new ContactDatasetGraphAnalysis<>(input, "\t", Parsers.lp, directed, notReciprocal);
                stats.analyze(output, limit, list);

                break;
            }
            case KNOWLEDGE:
            {
                double threshold = Parsers.dp.parse(execArgs[2]);
                boolean useRatings = execArgs[3].equalsIgnoreCase("true");
                KnowledgeDataUse dataUse = KnowledgeDataUse.fromString(execArgs[4]);


                WithKnowledgeDatasetGraphAnalysis<Long, Long> stats = new WithKnowledgeDatasetGraphAnalysis<>(input, "::", Parsers.lp, Parsers.lp, threshold, useRatings, dataUse);
                stats.analyze(output, limit, list);
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
        builder.append("\tOutput: file to store the graph\n");

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
                return "ERROR: This program is not allowed to use with streaming datasets";
        }
        return builder.toString();
    }

}
