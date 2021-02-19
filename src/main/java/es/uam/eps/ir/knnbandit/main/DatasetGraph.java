/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main;

import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import it.unimi.dsi.fastutil.objects.Object2LongOpenHashMap;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Finds an undirected, weighted graph indicating the existence of common rated items between users.
 *
 * The weight of an edge is computed as follows:
 * - First, we find the number of common items inter(u,v)
 * - Second, we find the number of relevant items u and v do not have in common (union(u,v)-2 inter(u,v))
 * - Then, the weight is computed as the inverse of that quantity (1.0/(union(x,y)-2 inter(u,v) + 1), so that
 *   the weight value is equal to 1 if they have no uncommon relevant items.
 * In case there is no common relevant items, weight is automatically equal to zero to penalize this.
 *
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class DatasetGraph<U,I>
{
    /**
     * Finds the interactions graph, and writes into a file.
     * @param output the file in which to write the graph.
     * @throws IOException if something fails while writing the graph.
     */
    public void graph(String output) throws IOException
    {
        OfflineDataset<U,I> dataset = this.getDataset();
        System.out.println(dataset.toString());

        AtomicInteger atom = new AtomicInteger(0);
        Map<U, Map<U, Long>> map = new HashMap<>();
        Map<U, Long> numRel = new HashMap<>();

        // Second, we find the network:
        dataset.getAllUsers().forEach(u ->
        {
            Object2LongOpenHashMap<U> uMap = new Object2LongOpenHashMap<>();
            long rel = dataset.getUserPreferences(u).filter(i -> dataset.isRelevant(i.v2)).mapToLong(i ->
            {
                dataset.getItemPreferences(i.v1).filter(v -> !map.containsKey(v.v1) && dataset.isRelevant(v.v2)).forEach(v -> uMap.addTo(v.v1, 1));
                return 1;
            }).sum();
            map.put(u, uMap);
            numRel.put(u, rel);

            int atomicInteger = atom.incrementAndGet();
            if(atomicInteger % 1000 == 0)
            {
                System.out.println("Processed " + atomicInteger + " users");
            }
        });

        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output))))
        {
            bw.write("source\tdest\tweight");
            int i = 0;
            for (Map.Entry<U, Map<U,Long>> entry : map.entrySet())
            {
                U u = entry.getKey();
                Map<U,Long> uMap = entry.getValue();
                for (Map.Entry<U, Long> uEntry : uMap.entrySet())
                {
                    U v = uEntry.getKey();
                    long val = uEntry.getValue();
                    if(val > 1 && u != v)
                        bw.write("\n" + u + "\t" + uEntry.getKey() + "\t" + 1.0 / (numRel.get(u) + numRel.get(v) - 2 * val + 1));
                }

                ++i;
                if (i % 100 == 0)
                {
                    System.out.println("Printed " + i + "users");
                }
            }
        }
    }


    /**
     * Obtains the dataset.
     * @return the dataset.
     */
    protected abstract OfflineDataset<U,I> getDataset();
}
