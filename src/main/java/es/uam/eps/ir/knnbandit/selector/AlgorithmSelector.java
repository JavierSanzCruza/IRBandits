/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector;

import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.wisdom.AverageCosineUserDistance;
import es.uam.eps.ir.knnbandit.recommendation.wisdom.ItemCentroidDistance;
import es.uam.eps.ir.knnbandit.recommendation.wisdom.MaximumCosineUserDistance;
import es.uam.eps.ir.knnbandit.selector.algorithms.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.DoublePredicate;

/**
 * Class for selecting the interactive recommendation algorithms to apply in an experiments. The class encapsulates the set
 * of algorithms, as well as some experiment settings.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class AlgorithmSelector<U, I>
{
    /**
     * The identifier for the algorithm name.
     */
    private final static String NAME = "name";
    /**
     * The identifier for the parameters of the algorithms.
     */
    private final static String PARAMS = "params";

    /**
     * A map of recommenders to apply.
     */
    private final Map<String, InteractiveRecommenderSupplier<U, I>> recs;

    /**
     * Indicates if the selector has been previously configured.
     */
    private boolean configured;
    /**
     * Relevance threshold.
     */
    private DoublePredicate relevanceChecker;

    /**
     * Constructor.
     */
    public AlgorithmSelector()
    {
        recs = new TreeMap<>();
    }

    /**
     * Resets the selection.
     */
    public void reset()
    {
        this.recs.clear();
        this.configured = false;
    }

    /**
     * Configures the experiment for traditional item-to-people recommendation, from a full cold start perspective.
     */
    public void configure(DoublePredicate relevance)
    {
        this.configured = true;
        this.relevanceChecker = relevance;
    }

    private AlgorithmConfigurator<U,I> getConfigurator(String name)
    {
        AlgorithmConfigurator<U,I> conf;
        switch(name)
        {
            case AlgorithmIdentifiers.RANDOM:
                conf = new RandomConfigurator<>();
                break;
            case AlgorithmIdentifiers.AVG:
                conf = new AverageConfigurator<>();
                break;
            case AlgorithmIdentifiers.POP:
                conf = new PopularityConfigurator<>(this.relevanceChecker);
                break;
            case AlgorithmIdentifiers.INFTHEOR:
                conf = new InformationTheoryUserDiversityConfigurator<>(this.relevanceChecker);
                break;
            case AlgorithmIdentifiers.ITEMBANDIT:
                conf = new ItemBanditConfigurator<>();
                break;
            case AlgorithmIdentifiers.USERBASEDKNN:
                conf = new UserBasedKNNConfigurator<>();
                break;
            case AlgorithmIdentifiers.ITEMBASEDKNN:
                conf = new ItemBasedKNNConfigurator<>();
                break;
            case AlgorithmIdentifiers.UBBANDIT:
                conf = new UserBasedKNNBanditConfigurator<>();
                break;
            case AlgorithmIdentifiers.IBBANDIT:
                conf = new ItemBasedKNNBanditConfigurator<>();
                break;
            case AlgorithmIdentifiers.MF:
                conf = new MFConfigurator<>();
                break;
            case AlgorithmIdentifiers.PMFBANDIT:
                conf = new PMFBanditConfigurator<>();
                break;
            case AlgorithmIdentifiers.PTS:
                conf = new ParticleThompsonSamplingMFConfigurator<>();
                break;
            case AlgorithmIdentifiers.BAYESIANPTS:
                conf = new BayesianParticleThompsonSamplingMFConfigurator<>();
                break;
            case AlgorithmIdentifiers.COLLABGREEDY:
                conf = new CollabGreedyConfigurator<>();
                break;
            case AlgorithmIdentifiers.CLUB:
                conf = new CLUBConfigurator<>();
                break;
            case AlgorithmIdentifiers.CLUBERDOS:
                conf = new CLUBErdosConfigurator<>();
                break;
            case AlgorithmIdentifiers.COFIBA:
                conf = new COFIBAConfigurator<>();
                break;
            case AlgorithmIdentifiers.AVGUSERCOS:
                conf = new AverageCosineUserDistanceConfigurator<>(this.relevanceChecker);
                break;
            case AlgorithmIdentifiers.MAXUSERCOS:
                conf = new MaximumCosineUserDistanceConfigurator<>(this.relevanceChecker);
                break;
            case AlgorithmIdentifiers.ITEMCENTR:
                conf = new ItemCentroidDistanceConfigurator<>(this.relevanceChecker);
                break;
            default:
                conf = null;
        }

        return conf;
    }

    /**
     * Obtains the suppliers for a single recommendation algorithm.
     * @param json the JSON object.
     * @return the list of recommender suppliers.
     */
    public List<InteractiveRecommenderSupplier<U,I>> getAlgorithms(JSONObject json)
    {
        String algorithm = json.getString(NAME);
        AlgorithmConfigurator<U,I> conf = this.getConfigurator(algorithm);
        return conf.getAlgorithms(json.getJSONArray(PARAMS));
    }

    /**
     * Obtains a single algorithm supplier from a JSON object.
     * @param json the JSON object containing the algorithm configuration parameters.
     * @return the supplier for the algorithm under the corresponding configuration.
     */
    public InteractiveRecommenderSupplier<U,I> getAlgorithm(JSONObject json)
    {
        String algorithm = json.getString(NAME);
        AlgorithmConfigurator<U,I> conf = this.getConfigurator(algorithm);
        return conf.getAlgorithm(json.getJSONObject(PARAMS));
    }

    /**
     * Adds a set of algorithms.
     *
     * @param file File containing the configuration of the algorithms.
     * @throws IOException                                            if something fails while reading
     * @throws es.uam.eps.ir.knnbandit.selector.UnconfiguredException if the selector has not been configured
     */
    public void addFile(String file, boolean single) throws IOException, UnconfiguredException
    {
        if (!this.configured)
        {
            throw new UnconfiguredException("AlgorithmSelector");
        }

        StringBuilder jSon = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file))))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                jSon.append(line);
                jSon.append("\n");
            }
        }

        JSONArray array = new JSONArray(jSon.toString());
        int length = array.length();
        for(int i = 0; i < length; ++i)
        {
            if(single)
            {
                InteractiveRecommenderSupplier<U, I> supplier = this.getAlgorithm(array.getJSONObject(i));
                recs.put(supplier.getName(), supplier);
            }
            else
            {
                List<InteractiveRecommenderSupplier<U,I>> suppliers = this.getAlgorithms(array.getJSONObject(i));
                for(InteractiveRecommenderSupplier<U,I> supplier : suppliers)
                {
                    recs.put(supplier.getName(), supplier);
                }
            }
        }
    }

    /**
     * Adds a set of algorithms.
     *
     * @param array File containing the configuration of the algorithms.
     * @throws es.uam.eps.ir.knnbandit.selector.UnconfiguredException if the selector has not been configured
     */
    public void addList(JSONArray array, boolean single) throws UnconfiguredException
    {
        if (!this.configured)
        {
            throw new UnconfiguredException("AlgorithmSelector");
        }

        int length = array.length();
        for(int i = 0; i < length; ++i)
        {
            if(single)
            {
                InteractiveRecommenderSupplier<U, I> supplier = this.getAlgorithm(array.getJSONObject(i));
                recs.put(supplier.getName(), supplier);
            }
            else
            {
                List<InteractiveRecommenderSupplier<U,I>> suppliers = this.getAlgorithms(array.getJSONObject(i));
                for(InteractiveRecommenderSupplier<U,I> supplier : suppliers)
                {
                    recs.put(supplier.getName(), supplier);
                }
            }
        }
    }

    /**
     * Obtains the selection of recommenders.
     *
     * @return the selection of recommenders.
     */
    public Map<String, InteractiveRecommenderSupplier<U, I>> getRecs()
    {
        return this.recs;
    }





}
