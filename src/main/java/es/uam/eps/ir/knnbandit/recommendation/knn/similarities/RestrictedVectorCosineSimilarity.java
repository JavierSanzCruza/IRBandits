/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.knn.similarities;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.function.IntToDoubleFunction;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Vector cosine similarity.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RestrictedVectorCosineSimilarity implements UpdateableSimilarity
{
    /**
     * The scalar product.
     */
    private final double[][] num;
    /**
     * The number of common items
     */
    private final double[][] common;
    /**
     * The number of users.
     */
    private final int numUsers;

    public RestrictedVectorCosineSimilarity(int numUsers)
    {
        this.numUsers = numUsers;
        this.num = new double[numUsers][numUsers];
        this.common = new double[numUsers][numUsers];
    }

    @Override
    public void update(int uidx, int vidx, int iidx, double uval, double vval)
    {
        if (!Double.isNaN(vval))
        {
            this.num[uidx][vidx] += uval * vval;
            this.num[vidx][uidx] += uval * vval;
            this.common[uidx][vidx] += 1;
            this.common[vidx][uidx] += 1;
        }
    }

    @Override
    public IntToDoubleFunction similarity(int idx)
    {
        return (int idx2) ->
        {
            if(this.common[idx][idx2] == 0)
            {
                return 0.0;
            }
            else
            {
                return this.num[idx][idx2]/this.common[idx][idx2];
            }
        };
    }

    @Override
    public Stream<Tuple2id> similarElems(int idx)
    {
        return IntStream.range(0, this.numUsers).filter(i -> i != idx).mapToObj(i -> new Tuple2id(i, similarity(idx, i))).filter(x -> x.v2 > 0.0);
    }

    @Override
    public void initialize(FastPreferenceData<?, ?> trainData)
    {
        trainData.getAllUidx().forEach(uidx -> trainData.getAllUidx().forEach(vidx ->
        {
            this.num[uidx][vidx] = 0.0;
            this.common[uidx][vidx] = 0.0;
        }));

        trainData.getAllUidx().forEach(uidx -> trainData.getUidxPreferences(uidx).forEach(iidx ->
            trainData.getIidxPreferences(iidx.v1).forEach(vidx ->
            {
                this.num[uidx][vidx.v1] += iidx.v2 * vidx.v2;
                this.common[uidx][vidx.v1] += 1;
            })
        ));
    }
}
