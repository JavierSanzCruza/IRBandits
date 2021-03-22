/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunctions;
import es.uam.eps.ir.knnbandit.recommendation.ensembles.MultiArmedBanditEnsemble;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.algorithms.bandit.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.DoublePredicate;

/**
 * Class for configuring a dynamic ensemble.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class MultiArmedBanditEnsembleConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for selecting whether the algorithm is updated with items unknown by the system or not.
     */
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    /**
     * Identifier for the list of algorithms to retrieve.
     */
    private static final String ALGORITHMS = "algorithms";
    /**
     * Identifier for the multi-armed bandit to use.
     */
    private static final String BANDIT = "bandit";

    /**
     * Checks whether a value is relevant or not for an algorithm.
     */
    private final DoublePredicate predicate;

    /**
     * Constructor.
     * @param predicate checks whether a value is relevant or not for an algorithm.
     */
    public MultiArmedBanditEnsembleConfigurator(DoublePredicate predicate)
    {
        this.predicate = predicate;
    }

    @Override
    public List<InteractiveRecommenderSupplier<U, I>> getAlgorithms(JSONArray array)
    {
        List<InteractiveRecommenderSupplier<U,I>> list = new ArrayList<>();
        int numConfigs = array.length();
        for(int i = 0; i < numConfigs; ++i)
        {
            JSONObject object = array.getJSONObject(i);

            boolean ignoreUnknown = true;

            if(object.has(IGNOREUNKNOWN))
            {
                ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
            }

            AlgorithmSelector<U,I> selector = new AlgorithmSelector<>();
            selector.configure(predicate);

            JSONArray algorithms = object.getJSONArray(ALGORITHMS);
            Map<String, InteractiveRecommenderSupplier<U,I>> recs = new HashMap<>();
            int numAlgs = algorithms.length();
            for(int j = 0; j < numAlgs; ++j)
            {
                InteractiveRecommenderSupplier<U,I> rec = selector.getAlgorithm(algorithms.getJSONObject(j));
                recs.put(rec.getName(), rec);
            }

            // Obtain the corresponding bandit.
            JSONObject bandit = object.getJSONObject(BANDIT);
            MultiArmedBanditSelector banditSelector = new MultiArmedBanditSelector();
            List<BanditSupplier> banditSuppliers = banditSelector.getBandits(bandit);
            for(BanditSupplier supplier : banditSuppliers)
            {
                list.add(new MultiArmedBanditEnsembleRecommenderSupplier(recs, supplier, ignoreUnknown));
            }
        }

        return list;
    }

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }

        AlgorithmSelector<U,I> selector = new AlgorithmSelector<>();

        JSONArray algorithms = object.getJSONArray(ALGORITHMS);
        Map<String, InteractiveRecommenderSupplier<U,I>> recs = new HashMap<>();
        int numAlgs = algorithms.length();
        for(int i = 0; i < numAlgs; ++i)
        {
            InteractiveRecommenderSupplier<U,I> rec = selector.getAlgorithm(algorithms.getJSONObject(i));
            recs.put(rec.getName(), rec);
        }

        // Obtain the corresponding bandit.
        JSONObject bandit = object.getJSONObject(BANDIT);
        MultiArmedBanditSelector banditSelector = new MultiArmedBanditSelector();
        BanditSupplier banditSupplier = banditSelector.getBandit(bandit);

        return new MultiArmedBanditEnsembleRecommenderSupplier(recs, banditSupplier, ignoreUnknown);
    }

    /**
     * Supplier for multi-armed bandit ensemble algorithms.
     */
    private class MultiArmedBanditEnsembleRecommenderSupplier implements InteractiveRecommenderSupplier<U,I>
    {
        /**
         * A map containing algorithm suppliers.
         */
        private final Map<String, InteractiveRecommenderSupplier<U,I>> algs;
        /**
         * True if we ignore the unknown ratings, false otherwise.
         */
        private final boolean ignoreUnknown;
        /**
         * The bandit supplier.
         */
        private final BanditSupplier banditSupplier;

        /**
         * Constructor.
         * @param algs           the list of interactive recommenders to combine.
         * @param banditSupplier a supplier for the multi-armed bandit strategy to use in the ensemble.
         * @param ignoreUnknown  true if we ignore the unknown ratings, false otherwise.
         */
        public MultiArmedBanditEnsembleRecommenderSupplier(Map<String, InteractiveRecommenderSupplier<U,I>> algs, BanditSupplier banditSupplier, boolean ignoreUnknown)
        {
            this.algs = algs;
            this.banditSupplier = banditSupplier;
            this.ignoreUnknown = ignoreUnknown;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            return new MultiArmedBanditEnsemble<>(userIndex, itemIndex, ignoreUnknown, algs, banditSupplier, ValueFunctions.identity());
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return new MultiArmedBanditEnsemble<>(userIndex, itemIndex, ignoreUnknown, rngSeed, algs, banditSupplier, ValueFunctions.identity());
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.BANDITENSEMBLE + "-" + banditSupplier.getName() + "-" + (ignoreUnknown ? "ignore" : "all");
        }

        @Override
        public boolean isEnsemble()
        {
            return true;
        }
    }
}
