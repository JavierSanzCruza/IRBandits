/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit;

import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;

/**
 * Main class for running experiments.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class Main
{
    /**
     * Name for general recommendation.
     */
    private final static String GENERAL = "generalrec";
    /**
     * Name for contact recommendation.
     */
    private final static String CONTACT = "contactrec";
    /**
     * Name for general recommendation with training.
     */
    private final static String GENERALTRAIN = "generalrectrain";
    /**
     * Name for contact recommendation with training.
     */
    private final static String CONTACTTRAIN = "contactrectrain";
    /**
     * Name for contact recommendation validation
     */
    private final static String CONTACTVALID = "contactrecvalid";
    /**
     * Name for general recommendation validation.
     */
    private final static String GENERALVALID = "generalrecvalid";
    /**
     * Name for metric summarization
     */
    private final static String SUMMARIZE = "summarize";
    /**
     * Name for finding training statistics for general recommendation.
     */
    private final static String TRAININGSTATS = "generaltrainstats";
    /**
     * Name for finding training statistics for contact recommendation.
     */
    private final static String CONTACTTRAININGSTATS = "contacttrainstats";

    private final static String GENERALPARALLEL = "generalparallel";

    private final static String CONTACTPARALLEL = "contactparallel";

    /**
     * Main method. Executes the main method in the class specified by the first
     * argument with the rest of run time arguments.
     *
     * @param args Arguments to select the class to run and arguments for its main method
     */
    public static void main(String[] args)
    {
        try
        {
            String main = args[0];
            String className;
            switch (main)
            {
                case GENERAL:
                    className = "es.uam.eps.ir.knnbandit.InteractiveRecommendation";
                    break;
                case CONTACT:
                    className = "es.uam.eps.ir.knnbandit.InteractiveContactRecommendation";
                    break;
                case GENERALTRAIN:
                    className = "es.uam.eps.ir.knnbandit.InteractiveRecommendationWithTraining";
                    break;
                case CONTACTTRAIN:
                    className = "es.uam.eps.ir.knnbandit.InteractiveContactRecommendationWithTraining";
                    break;
                case SUMMARIZE:
                    className = "es.uam.eps.ir.knnbandit.OutputResumer";
                    break;
                case GENERALVALID:
                    className = "es.uam.eps.ir.knnbandit.InteractiveRecommendationValidation";
                    break;
                case CONTACTVALID:
                    className = "es.uam.eps.ir.knnbandit.InteractiveContactRecommendationValidation";
                    break;
                case GENERALPARALLEL:
                    className  = "es.uam.eps.ir.knnbandit.InteractiveRecommendationParallel";
                    break;
                case CONTACTPARALLEL:
                    className = "es.uam.eps.ir.knnbandit.InteractiveContactRecommendationParallel";
                    break;
                case TRAININGSTATS:
                    className = "es.uam.eps.ir.knnbandit.TrainingStatistics";
                    break;
                case CONTACTTRAININGSTATS:
                    className = "es.uam.eps.ir.knnbandit.TrainingContactStatistics";
                    break;
                default:
                    System.err.println("ERROR: Invalid configuration.");
                    return;
            }

            String[] executionArgs = Arrays.copyOfRange(args, 1, args.length);
            Class[] argTypes = {executionArgs.getClass()};
            Object[] passedArgs = {executionArgs};
            Class.forName(className).getMethod("main", argTypes).invoke(null, passedArgs);
        }
        catch (ClassNotFoundException | NoSuchMethodException | SecurityException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex)
        {
            System.err.println("The run time arguments were not correct");
            ex.printStackTrace();
        }
    }
}
