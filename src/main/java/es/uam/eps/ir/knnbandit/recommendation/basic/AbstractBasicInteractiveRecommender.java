/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.basic;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.userknowledge.fast.SimpleFastUserKnowledgePreferenceData;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.stream.IntStream;

/**
 * Abstract class for basic recommendation algorithms.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class AbstractBasicInteractiveRecommender<U, I> extends InteractiveRecommender<U, I>
{
    /**
     * Values of each item.
     */
    protected double[] values;

    /**
     * Constructor.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param prefData  Preference data.
     * @param hasRating True if (user, item) pairs without training must be ignored.
     */
    public AbstractBasicInteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRating)
    {
        super(uIndex, iIndex, prefData, hasRating);
        this.values = new double[prefData.numItems()];
        IntStream.range(0, prefData.numItems()).forEach(iidx -> values[iidx] = 0);
    }

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      Preference data.
     * @param hasRating     True if (user, item) pairs without training must be ignored.
     * @param notReciprocal True if we do not recommend reciprocal social links, false otherwise.
     */
    public AbstractBasicInteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRating, boolean notReciprocal)
    {
        super(uIndex, iIndex, prefData, hasRating, notReciprocal);
        this.values = new double[prefData.numItems()];
        IntStream.range(0, prefData.numItems()).forEach(iidx -> values[iidx] = 0);
    }

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      Preference data.
     * @param hasRating     True if (user, item) pairs without training must be ignored.
     * @param dataUse       Which data must be used in updates (known, unknown or all)
     */
    public AbstractBasicInteractiveRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData, boolean hasRating, KnowledgeDataUse dataUse)
    {
        super(uIndex, iIndex, prefData, knowledgeData, hasRating, dataUse);
        this.values = new double[prefData.numItems()];
        IntStream.range(0, prefData.numItems()).forEach(iidx -> values[iidx] = 0);
    }

    @Override
    public int next(int uidx)
    {
        IntList list = this.availability.get(uidx);
        if (list == null || list.isEmpty())
        {
            return -1;
        }
        else
        {
            double val = Double.NEGATIVE_INFINITY;
            IntList top = new IntArrayList();

            for (int item : list)
            {
                if (values[item] > val)
                {
                    val = values[item];
                    top = new IntArrayList();
                    top.add(item);
                }
                else if (values[item] == val)
                {
                    top.add(item);
                }
            }

            int nextItem;
            int size = top.size();
            if (size == 1)
            {
                nextItem = top.get(0);
            }
            else
            {
                nextItem = top.get(rng.nextInt(size));
            }

            return nextItem;
        }
    }
}
