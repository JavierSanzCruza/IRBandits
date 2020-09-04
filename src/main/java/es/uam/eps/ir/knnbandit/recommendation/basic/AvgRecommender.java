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
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple3;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Interactive version of an average rating recommendation algorithm.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class AvgRecommender<U, I> extends AbstractBasicInteractiveRecommender<U, I>
{
    /**
     * Number of times an arm has been selected.
     */
    private double[] numTimes;

    /**
     * Constructor.
     *
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param prefData   Preference data.
     * @param hasRatings True if (user, item) pairs without training must be ignored.
     */
    public AvgRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRatings)
    {
        super(uIndex, iIndex, prefData, hasRatings);
        this.numTimes = new double[prefData.numItems()];
        IntStream.range(0, prefData.numItems()).forEach(iidx -> this.numTimes[iidx] = 0);
    }

    /**
     * Constructor.
     *
     * @param uIndex        User index.
     * @param iIndex        Item index.
     * @param prefData      Preference data.
     * @param hasRatings    True if (user, item) pairs without training must be ignored.
     * @param notReciprocal True if we do not recommend reciprocal social links, false otherwise
     */
    public AvgRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean hasRatings, boolean notReciprocal)
    {
        super(uIndex, iIndex, prefData, hasRatings, notReciprocal);
        this.numTimes = new double[prefData.numItems()];
        IntStream.range(0, prefData.numItems()).forEach(iidx -> this.numTimes[iidx] = 0);
    }

    public AvgRecommender(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData, boolean hasRatings, KnowledgeDataUse dataUse)
    {
        super(uIndex, iIndex, prefData, knowledgeData, hasRatings, dataUse);
        this.numTimes = new double[prefData.numItems()];
        IntStream.range(0, prefData.numItems()).forEach(iidx -> this.numTimes[iidx] = 0);
    }

    @Override
    protected void initializeMethod()
    {
        IntStream.range(0, prefData.numItems()).forEach(iidx ->
        {
            this.numTimes[iidx] = 0.0;
            this.values[iidx] = 0.0;
        });

        this.trainData.getIidxWithPreferences().forEach(iidx ->
        {
            this.values[iidx] = this.trainData.getIidxPreferences(iidx).mapToDouble(pref -> pref.v2).average().getAsDouble();
            this.numTimes[iidx] = this.trainData.numUsers(iidx);
        });
    }

    @Override
    public void updateMethod(int uidx, int iidx, double value)
    {
        double oldValue = values[iidx];
        if (numTimes[iidx] <= 0.0)
        {
            this.values[iidx] = value;
        }
        else
        {
            this.values[iidx] = oldValue + (value - oldValue) / (numTimes[iidx] + 1.0);
        }
        this.numTimes[iidx]++;
    }

    @Override
    public void updateMethod(List<Tuple3<Integer, Integer, Double>> train)
    {
        for (int i = 0; i < this.trainData.numItems(); ++i)
        {
            this.values[i] = this.trainData.getIidxPreferences(i).mapToDouble(v -> v.v2).sum();
            this.numTimes[i] = this.trainData.numUsers(i);
            if (this.numTimes[i] > 0)
            {
                this.values[i] /= (this.numTimes[i] + 0.0);
            }
        }
    }
}
