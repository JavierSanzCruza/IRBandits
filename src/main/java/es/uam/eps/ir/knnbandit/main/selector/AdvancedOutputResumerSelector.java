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

import es.uam.eps.ir.knnbandit.selector.io.IOSelector;
import es.uam.eps.ir.knnbandit.selector.io.IOType;
import es.uam.eps.ir.knnbandit.main.AdvancedOutputResumer;
import es.uam.eps.ir.knnbandit.main.contact.ContactAdvancedOutputResumer;
import es.uam.eps.ir.knnbandit.main.general.GeneralAdvancedOutputResumer;
import es.uam.eps.ir.knnbandit.main.withknowledge.WithKnowledgeAdvancedOutputResumer;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.formats.parsing.Parsers;
import static es.uam.eps.ir.knnbandit.main.selector.DatasetType.*;

import java.io.IOException;
import java.util.Arrays;

/**
 * Main class for summarizing the content of recommendation files.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class AdvancedOutputResumerSelector
{
    /**
     * Summarizes the recommendation files.
     * @param type the type of the dataset we are using.
     * @param args the execution arguments.
     *
     * @throws IOException if something fails while reading/writing
     */
    public void summarize(String type, String[] args) throws IOException
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
                lastIndex = 5;
                break;
            case CONTACT:
            case KNOWLEDGE:
                length = 5;
                firstIndex = 0;
                lastIndex = 5;
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
        String directory = execArgs[1];
        String timePoints = execArgs[2];
        IntList points = new IntArrayList();
        String[] split= timePoints.split(",");
        for(String s : split)
        {
            points.add(Parsers.ip.parse(s));
        }

        boolean recursive = false;
        IOType iotype = IOType.TEXT;
        boolean gzipped = false;
        for (int i = lastIndex; i < execArgs.length; ++i)
        {
            if ("-r".equals(args[i]))
            {
                recursive = true;
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

        IOSelector ioSelector = new IOSelector(iotype, gzipped);

        switch(type)
        {
            case GENERAL:
            {
                double threshold = Parsers.dp.parse(execArgs[3]);
                boolean useRatings = execArgs[4].equalsIgnoreCase("true");

                if(args[0].equalsIgnoreCase("movielens"))
                {
                    GeneralAdvancedOutputResumer<Long, Long> resumer = new GeneralAdvancedOutputResumer<>(input, "::", Parsers.lp, Parsers.lp, threshold, useRatings, ioSelector);
                    resumer.summarize(directory, points, recursive);
                }
                else if(args[0].equalsIgnoreCase("foursquare"))
                {
                    GeneralAdvancedOutputResumer<Long, String> resumer = new GeneralAdvancedOutputResumer<>(input, "::", Parsers.lp, Parsers.sp, threshold, useRatings, ioSelector);
                    resumer.summarize(directory, points, recursive);
                }
                break;
            }
            case CONTACT:
            {
                boolean directed = execArgs[3].equalsIgnoreCase("true");
                boolean notReciprocal = execArgs[4].equalsIgnoreCase("true");

                AdvancedOutputResumer<Long, Long> resumer = new ContactAdvancedOutputResumer<>(input, "\t", Parsers.lp, directed, notReciprocal, ioSelector);
                resumer.summarize(directory, points, recursive);

                break;
            }
            case KNOWLEDGE:
            {
                double threshold = Parsers.dp.parse(execArgs[3]);
                boolean useRatings = execArgs[4].equalsIgnoreCase("true");

                WithKnowledgeAdvancedOutputResumer<Long, Long> resumer = new WithKnowledgeAdvancedOutputResumer<>(input, "::", Parsers.lp, Parsers.lp, threshold, useRatings, ioSelector);
                resumer.summarize(directory, points, recursive);
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
        builder.append("\tDirectory: directory where the recommenders are\n");
        builder.append("\tTime points: comma separated time points\n");

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
        builder.append("\t-r : if we want to make the program recursive over internal directories");
        builder.append("\t-io-type : establishes the format of the input-output files. Possible values:\n");
        builder.append("\t\tbinary : for binary files\n");
        builder.append("\t\ttext : for text files (default value)\n");
        builder.append("\t--gzipped : if we want to compress the recommendation files (by default, they are not compressed)");

        return builder.toString();
    }

}
