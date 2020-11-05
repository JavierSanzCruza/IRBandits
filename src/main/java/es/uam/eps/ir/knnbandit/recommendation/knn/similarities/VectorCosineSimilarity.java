/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.knn.similarities;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
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
public class VectorCosineSimilarity implements UpdateableSimilarity
{
    /**
     * The scalar product.
     */
    private final Int2ObjectMap<Int2DoubleMap> num;
    /**
     * The norms of each user.
     */
    private final double[] norm;
    /**
     * The number of users.
     */
    private final int numUsers;

    /**
     * Constructor.
     * @param numUsers The number of users in the system.
     */
    public VectorCosineSimilarity(int numUsers)
    {
        this.numUsers = numUsers;
        this.num = new Int2ObjectOpenHashMap<>();
        this.norm = new double[numUsers];
    }

    @Override
    public void updateNorm(int uidx, double value)
    {
        norm[uidx] += value*value;
    }

    @Override
    public void updateNormDel(int uidx, double value)
    {
        norm[uidx] -= value*value;
    }

    @Override
    public void update(int uidx, int vidx, int iidx, double uval, double vval)
    {
        if(!Double.isNaN(vval))
        {
            if(!this.num.containsKey(uidx))
            {
                Int2DoubleMap map = new Int2DoubleOpenHashMap();
                map.defaultReturnValue(0.0);
                this.num.put(uidx, map);
            }

            if(!this.num.containsKey(vidx))
            {
                Int2DoubleMap map = new Int2DoubleOpenHashMap();
                map.defaultReturnValue(0.0);
                this.num.put(vidx, map);
            }

            ((Int2DoubleOpenHashMap) this.num.get(uidx)).addTo(vidx, uval*vval);
            ((Int2DoubleOpenHashMap) this.num.get(vidx)).addTo(uidx, uval*vval);
        }
    }

    @Override
    public void updateDel(int uidx, int vidx, int iidx, double uval, double vval)
    {
        if(!Double.isNaN(vval) && this.num.containsKey(uidx) && this.num.get(uidx).containsKey(vidx))
        {
            ((Int2DoubleOpenHashMap) this.num.get(uidx)).addTo(vidx, -uval*vval);
            ((Int2DoubleOpenHashMap) this.num.get(vidx)).addTo(uidx, -uval*vval);

            if(this.num.get(uidx).get(vidx) == 0.0)
            {
                this.num.get(uidx).remove(vidx);
                this.num.get(vidx).remove(uidx);
            }
        }
    }

    @Override
    public IntToDoubleFunction similarity(int idx)
    {
        return (int idx2) ->
        {
            double sum = Math.sqrt(this.norm[idx]) * Math.sqrt(this.norm[idx2]);
            if (sum == 0.0)
            {
                return 0.0;
            }
            else
            {
                return this.num.getOrDefault(idx, new Int2DoubleOpenHashMap()).getOrDefault(idx2, 0.0) / sum;
            }
        };
    }

    @Override
    public Stream<Tuple2id> similarElems(int idx)
    {
        try
        {
            if (this.norm[idx] > 0.0)
            {
                return this.num.get(idx).int2DoubleEntrySet().stream().filter(v -> v.getIntKey() != idx).map(v -> new Tuple2id(v.getIntKey(), v.getDoubleValue() / this.norm[v.getIntKey()]));
            }
            return Stream.empty();
            //return IntStream.range(0, this.numUsers).filter(i -> i != idx).mapToObj(i -> new Tuple2id(i, similarity(idx, i))).filter(x -> x.v2 > 0.0);
        }
        catch(NullPointerException poi)
        {
            double value = this.norm[idx];
            Int2DoubleOpenHashMap map = (Int2DoubleOpenHashMap) this.num.get(idx);
            System.err.println("Something failed");
            return Stream.empty();
        }
    }

    @Override
    public void initialize()
    {
        this.num.clear();
        IntStream.range(0, this.numUsers).forEach(uidx -> this.norm[uidx] = 0.0);
    }


    @Override
    public void initialize(FastPreferenceData<?, ?> trainData)
    {
        trainData.getAllUidx().forEach(uidx ->
        {
            Int2DoubleOpenHashMap map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.num.put(uidx, map);

            this.norm[uidx] = trainData.getUidxPreferences(uidx).mapToDouble(iidx ->
            {
                trainData.getIidxPreferences(iidx.v1).forEach(vidx -> ((Int2DoubleOpenHashMap) this.num.get(uidx)).addTo(vidx.v1,iidx.v2*vidx.v2));
                return iidx.v2*iidx.v2;
            }).sum();
        });
    }
}
