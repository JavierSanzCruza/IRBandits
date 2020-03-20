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

import org.jooq.lambda.tuple.Tuple2;

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
    private final static String RECOMMENDATION = "rec";
    private final static String VALIDATION = "valid";
    private final static String WARMUPRECOMMENDATION = "warmup-rec";
    private final static String WARMUPRECOMMENDATIONPARALLEL = "warmup-rec-parallel";
    private final static String WARMUPVALIDATION = "warmup-valid";
    private final static String TRAININGSTATS = "train-stats";

    private final static String SUMMARIZE = "summarize";
    private final static String OUTPUTRANKER = "outputranker";
    private final static String RANKERCONFIG = "rankerconfig";

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
            int from = 1;
            switch (main)
            {
                case RECOMMENDATION:
                    Tuple2<Integer, String> alg = Main.getAlgorithm(args, "Recommendation");
                    className = alg.v2;
                    from = alg.v1;
                    break;
                case VALIDATION:
                    alg = Main.getAlgorithm(args, "Validation");
                    className = alg.v2;
                    from = alg.v1;
                    break;
                case WARMUPRECOMMENDATION:
                    alg = Main.getAlgorithm(args, "WarmupRecommendation");
                    className = alg.v2;
                    from = alg.v1;
                    break;
                case WARMUPVALIDATION:
                    alg = Main.getAlgorithm(args, "WarmupValidation");
                    className = alg.v2;
                    from = alg.v1;
                    break;
                case WARMUPRECOMMENDATIONPARALLEL:
                    alg = Main.getAlgorithm(args, "WarmupRecommendationParallel");
                    className = alg.v2;
                    from = alg.v1;
                    break;
                case TRAININGSTATS:
                    alg = Main.getAlgorithm(args, "TrainingStatistics");
                    className = alg.v2;
                    from = alg.v1;
                    break;

                case SUMMARIZE:
                    className = "es.uam.eps.ir.knnbandit.main.OutputResumer";
                    break;
                case OUTPUTRANKER:
                    className = "es.uam.eps.ir.knnbandit.main.OutputRanker";
                    break;
                case RANKERCONFIG:
                    className = "es.uam.eps.ir.knnbandit.main.ParallelConfigurator";
                    break;
                default:
                    System.err.println("ERROR: Invalid configuration.");
                    return;
            }

            String[] executionArgs = Arrays.copyOfRange(args, from, args.length);
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

    /**
     * For the different classes that depend on dataset selection, choose the most appropriate one.
     * @param args the list of arguments received by the Main program.
     * @param programName the name of the program we want to find the route to.
     * @return the point of the arguments where the real arguments begin and the class name.
     */
    private static Tuple2<Integer, String> getAlgorithm(String[] args, String programName)
    {
        String command = "es.uam.eps.ir.knnbandit.main";
        String type = args[1];
        Tuple2<Integer, String> tuple = null;
        if(type.equalsIgnoreCase("contact"))
        {
            command += ".contact." + programName;
            tuple = new Tuple2<>(2, command);
        }
        else if(type.equalsIgnoreCase("general"))// general
        {
            String dataset = args[2].toLowerCase();
            command += ".general." + dataset + "." + programName;
            tuple = new Tuple2<>(3, command);
        }
        return tuple;
    }
}
