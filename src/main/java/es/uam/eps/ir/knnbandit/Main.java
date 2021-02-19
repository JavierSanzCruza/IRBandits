/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit;

import es.uam.eps.ir.knnbandit.main.selector.*;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;

import java.io.IOException;
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
    private final static String WARMUPVALIDATION = "warmup-valid";
    private final static String TRAININGSTATS = "train-stats";
    private final static String DATASET = "dataset";
    private final static String ANALYSIS = "dataset-analysis";
    private final static String SUMMARIZE = "summarize";
    private final static String OUTPUTRANKER = "outputranker";

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
                case VALIDATION:
                {
                    ValidationSelector valid = new ValidationSelector();
                    valid.validate(args[1], Arrays.copyOfRange(args, 2, args.length));
                    break;
                }
                case RECOMMENDATION:
                {
                    RecommendationSelector rec = new RecommendationSelector();
                    rec.recommend(args[1], Arrays.copyOfRange(args, 2, args.length));
                    break;
                }
                case WARMUPVALIDATION:
                {
                    WarmupValidationSelector valid = new WarmupValidationSelector();
                    valid.validate(args[1], Arrays.copyOfRange(args, 2, args.length));
                    break;
                }
                case WARMUPRECOMMENDATION:
                {
                    WarmupRecommendationSelector rec = new WarmupRecommendationSelector();
                    rec.recommend(args[1], Arrays.copyOfRange(args, 2, args.length));
                    break;
                }
                case TRAININGSTATS:
                {
                    TrainingStatisticsSelector stats = new TrainingStatisticsSelector();
                    stats.statistics(args[1], Arrays.copyOfRange(args, 2, args.length));
                    break;
                }
                case SUMMARIZE:
                {
                    AdvancedOutputResumerSelector resumer = new AdvancedOutputResumerSelector();
                    resumer.summarize(args[1], Arrays.copyOfRange(args,2, args.length));
                    break;
                }
                case DATASET:
                {
                    DatasetGraphSelector grapher = new DatasetGraphSelector();
                    grapher.graph(args[1], Arrays.copyOfRange(args, 2, args.length));
                    break;
                }
                case ANALYSIS:
                {
                    DatasetGraphAnalysisSelector grapher = new DatasetGraphAnalysisSelector();
                    grapher.analyze(args[1], Arrays.copyOfRange(args, 2, args.length));
                    break;
                }
                case OUTPUTRANKER:
                {
                    className = "es.uam.eps.ir.knnbandit.main.OutputRanker";
                    String[] executionArgs = Arrays.copyOfRange(args, from, args.length);
                    Class[] argTypes = {executionArgs.getClass()};
                    Object[] passedArgs = {executionArgs};
                    Class.forName(className).getMethod("main", argTypes).invoke(null, passedArgs);
                    break;
                }
                default:
                    System.err.println("ERROR: Invalid configuration.");
                    return;
            }


        }
        catch (ClassNotFoundException | NoSuchMethodException | SecurityException | IllegalAccessException | IllegalArgumentException | InvocationTargetException | IOException | UnconfiguredException ex)
        {
            System.err.println("The run time arguments were not correct");
            ex.printStackTrace();
        }
    }
}
