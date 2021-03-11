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
import es.uam.eps.ir.knnbandit.recommendation.bandits.ItemBanditRecommender;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunctions;
import es.uam.eps.ir.knnbandit.selector.AlgorithmIdentifiers;
import es.uam.eps.ir.knnbandit.selector.algorithms.bandit.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * Class for selecting the non-personalized bandit-based recommendation algorithms in our experiments.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ItemBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    /**
     * Identifier for selecting whether the algorithm is updated with items unknown by the system or not.
     */
    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    /**
     * Identifier for the bandit algorithms.
     */
    private static final String BANDIT = "bandit";

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

            MultiArmedBanditSelector selector = new MultiArmedBanditSelector();
            List<BanditSupplier> banditSuppliers = selector.getBandits(object.getJSONObject(BANDIT));
            for(BanditSupplier supplier : banditSuppliers)
            {
                list.add(new ItemBanditInteractiveRecommenderSupplier<>(supplier, ignoreUnknown));
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

        JSONObject bandit = object.getJSONObject(BANDIT);
        MultiArmedBanditSelector selector = new MultiArmedBanditSelector();
        BanditSupplier banditSupplier = selector.getBandit(bandit);
        return new ItemBanditInteractiveRecommenderSupplier<>(banditSupplier, ignoreUnknown);
    }

    /**
     * Class for configuring a non-personalized item recommendation algorithm based on simple and context-less multi-armed bandits.
     *
     * @param <U> type of the users.
     * @param <I> type of the items.
     */
    private static class ItemBanditInteractiveRecommenderSupplier<U,I> implements InteractiveRecommenderSupplier<U,I>
    {
        /**
         * A supplier for the applied multi-armed bandit algorithm in the comparison.
         */
        BanditSupplier banditSupplier;
        /**
         * True if we do not use ratings unknown by the dataset, false otherwise.
         */
        boolean ignoreUnknown;

        /**
         * Constructor.
         * @param supplier       a supplier for the multi-armed bandit strategy to use to select the items.
         * @param ignoreUnknown  true if we ignore the unknown ratings, false otherwise.
         */
        public ItemBanditInteractiveRecommenderSupplier(BanditSupplier supplier, boolean ignoreUnknown)
        {
            this.banditSupplier = supplier;
            this.ignoreUnknown = ignoreUnknown;
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex)
        {
            ValueFunction valueFunction = ValueFunctions.identity();
            return new ItemBanditRecommender<>(userIndex, itemIndex, ignoreUnknown, banditSupplier, valueFunction);
        }

        @Override
        public AbstractInteractiveRecommender<U, I> apply(FastUpdateableUserIndex<U> userIndex, FastUpdateableItemIndex<I> itemIndex, int rngSeed)
        {
            return this.apply(userIndex, itemIndex);
        }

        @Override
        public String getName()
        {
            return AlgorithmIdentifiers.ITEMBANDIT + "-" + banditSupplier.getName() + "-" + (ignoreUnknown ? "ignore" : "all");
        }
    }
}
