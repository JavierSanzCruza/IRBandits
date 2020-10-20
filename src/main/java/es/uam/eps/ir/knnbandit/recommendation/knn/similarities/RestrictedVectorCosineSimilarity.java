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
public class RestrictedVectorCosineSimilarity implements UpdateableSimilarity
{
    /**
     * The scalar product.
     */
    private final Int2ObjectMap<Int2DoubleMap> num;
    /**
     * The number of common items
     */
    private final Int2ObjectMap<Int2DoubleMap> common;
    /**
     * The number of users.
     */
    private final int numUsers;

    public RestrictedVectorCosineSimilarity(int numUsers)
    {
        this.numUsers = numUsers;
        this.num = new Int2ObjectOpenHashMap<>();
        this.common = new Int2ObjectOpenHashMap<>();
    }

    @Override
    public void initialize()
    {
        this.num.clear();
        this.common.clear();
    }

    @Override
    public void initialize(FastPreferenceData<?, ?> trainData)
    {
        this.num.clear();
        this.common.clear();

        trainData.getUidxWithPreferences().forEach(uidx ->
        {
            this.common.put(uidx, new Int2DoubleOpenHashMap());
            this.common.get(uidx).defaultReturnValue(0.0);
            this.num.put(uidx, new Int2DoubleOpenHashMap());
            this.num.get(uidx).defaultReturnValue(0.0);

            trainData.getUidxPreferences(uidx).forEach(iidx ->
                trainData.getIidxPreferences(iidx.v1).filter(vidx -> uidx != vidx.v1).forEach(vidx ->
                {
                    ((Int2DoubleOpenHashMap) this.num.get(uidx)).addTo(vidx.v1, iidx.v2*vidx.v2);
                    ((Int2DoubleOpenHashMap) this.common.get(uidx)).addTo(vidx.v1, 1.0);
                })
            );
        });
    }


    @Override
    public void updateNorm(int uidx, double value)
    {

    }

    @Override
    public void updateNormDel(int uidx, double value)
    {

    }

    @Override
    public void update(int uidx, int vidx, int iidx, double uval, double vval)
    {
        if(!this.num.containsKey(uidx))
        {
            Int2DoubleMap map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.num.put(uidx, map);
            map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.common.put(uidx, map);
        }
        if(!this.num.containsKey(vidx))
        {
            Int2DoubleMap map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.num.put(vidx, map);
            map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.common.put(vidx, map);
        }

        if (!Double.isNaN(vval))
        {
            ((Int2DoubleOpenHashMap) this.num.get(uidx)).addTo(vidx, uval*vval);
            ((Int2DoubleOpenHashMap) this.num.get(vidx)).addTo(uidx, uval*vval);
            ((Int2DoubleOpenHashMap) this.common.get(uidx)).addTo(vidx, 1.0);
            ((Int2DoubleOpenHashMap) this.common.get(vidx)).addTo(uidx, 1.0);
        }
    }

    @Override
    public void updateDel(int uidx, int vidx, int iidx, double uval, double vval)
    {
        if(this.num.containsKey(uidx) && this.num.containsKey(vidx) && !Double.isNaN(vval))
        {
            ((Int2DoubleOpenHashMap) this.num.get(uidx)).addTo(vidx, -uval*vval);
            ((Int2DoubleOpenHashMap) this.num.get(vidx)).addTo(uidx, -uval*vval);
            ((Int2DoubleOpenHashMap) this.common.get(uidx)).addTo(vidx, -1.0);
            ((Int2DoubleOpenHashMap) this.common.get(vidx)).addTo(uidx, -1.0);
        }
    }

    @Override
    public IntToDoubleFunction similarity(int idx)
    {
        return (int idx2) ->
        {
            double commonItems = this.common.getOrDefault(idx, new Int2DoubleOpenHashMap()).getOrDefault(idx2, 0.0);
            if(commonItems == 0.0) return 0.0;
            else
            {
                return this.num.get(idx).get(idx2) / this.common.get(idx).get(idx2);
            }
        };
    }

    @Override
    public Stream<Tuple2id> similarElems(int idx)
    {
        if(!this.common.containsKey(idx)) return Stream.empty();
        return this.common.get(idx).int2DoubleEntrySet().stream().map(v -> new Tuple2id(v.getIntKey(), this.num.get(idx).get(v.getIntKey())/v.getDoubleValue())).filter(v -> v.v2 > 0.0);
    }
}
