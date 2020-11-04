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

import es.uam.eps.ir.knnbandit.main.Recommendation;
import es.uam.eps.ir.knnbandit.main.Validation;
import es.uam.eps.ir.knnbandit.main.contact.ContactRecommendation;
import es.uam.eps.ir.knnbandit.main.general.GeneralRecommendation;
import es.uam.eps.ir.knnbandit.main.stream.ReplayerRecommendation;
import es.uam.eps.ir.knnbandit.main.stream.ReplayerValidation;
import es.uam.eps.ir.knnbandit.main.withknowledge.WithKnowledgeRecommendation;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import org.ranksys.formats.parsing.Parsers;
import static es.uam.eps.ir.knnbandit.main.selector.DatasetType.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.function.Supplier;

/**
 * Main class for determining the type of dataset / execution we are using
 * for applying recommendation.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RecommendationSelector
{
    /**
     * Executes the recommendation.
     * @param type the type of the dataset we are using.
     * @param args the execution arguments.
     *
     * @throws IOException if something fails while reading/writing
     * @throws UnconfiguredException if the algorithm selector is badly configured.
     */
    public void recommend(String type, String[] args) throws IOException, UnconfiguredException
    {
        String[] execArgs;
        String errorString = this.getErrorMessage(type);

        int firstIndex;
        int length;
        int lastIndex;

        switch(type)
        {
            case GENERAL:
                length = 8;
                firstIndex = 1;
                lastIndex = 7;
                break;
            case CONTACT:
                length = 7;
                firstIndex = 0;
                lastIndex = 7;
                break;
            case KNOWLEDGE:
                length = 8;
                firstIndex = 0;
                lastIndex = 8;
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
        int interval = 10000;
        for (int i = lastIndex; i < execArgs.length; ++i)
        {
            if ("-k".equals(args[i]))
            {
                ++i;
                k = Parsers.ip.parse(args[i]);
            }
            else if("-interval".equals(args[i]))
            {
                ++i;
                interval = Parsers.ip.parse(args[i]);
            }
        }

        switch(type)
        {
            case GENERAL:
            {
                double threshold = Parsers.dp.parse(execArgs[5]);
                boolean useRatings = execArgs[6].equalsIgnoreCase("true");

                if(args[1].equalsIgnoreCase("movielens"))
                {
                    Recommendation<Long, Long> rec = new GeneralRecommendation<>(input, "::", Parsers.lp, Parsers.lp, threshold, useRatings);
                    rec.recommend(algorithms, output, endCond, resume, k, interval);
                }
                else if(args[1].equalsIgnoreCase("foursquare"))
                {
                    Recommendation<Long, String> rec = new GeneralRecommendation<>(input, "::", Parsers.lp, Parsers.sp, threshold, useRatings);
                    rec.recommend(algorithms, output, endCond, resume, k, interval);
                }
                break;
            }
            case CONTACT:
            {
                boolean directed = execArgs[5].equalsIgnoreCase("true");
                boolean notReciprocal = execArgs[6].equalsIgnoreCase("true");

                Recommendation<Long, Long> rec = new ContactRecommendation<>(input, "::", Parsers.lp, directed, notReciprocal);
                rec.recommend(algorithms, output, endCond, resume, k, interval);

                break;
            }
            case KNOWLEDGE:
            {
                double threshold = Parsers.dp.parse(execArgs[5]);
                boolean useRatings = execArgs[6].equalsIgnoreCase("true");
                KnowledgeDataUse dataUse = KnowledgeDataUse.fromString(execArgs[7]);

                Recommendation<Long, Long> rec = new WithKnowledgeRecommendation<>(input, "::", Parsers.lp, Parsers.lp, threshold, useRatings, dataUse);
                rec.recommend(algorithms, output, endCond, resume, k, interval);
                break;
            }
            case STREAM:
            {
                double threshold = Parsers.dp.parse(execArgs[5]);
                String userIndex = execArgs[6];
                String itemIndex = execArgs[7];

                Recommendation<Integer, Integer> rec = new ReplayerRecommendation<>(input, "\t", userIndex, itemIndex, threshold, Parsers.ip, Parsers.ip);
                rec.recommend(algorithms, output,endCond, resume, k, interval);
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
                builder.append("\tThreshold: true if the graph is directed, false otherwise\n");
                builder.append("\tUser index: file containing a relation of users\n");
                builder.append("\tItem index: file containing a relation of items\n");
                break;
            default:
        }

        builder.append("Optional arguments:\n");
        builder.append("\t-k value : The number of times each individual approach has to be executed (by default: 1)\n");
        builder.append("\t-interval value : Distance between time points in the summary (by default: 10000)");

        return builder.toString();
    }

}
