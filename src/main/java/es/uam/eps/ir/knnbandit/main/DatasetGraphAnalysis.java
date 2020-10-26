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
import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.fast.FastUndirectedWeightedGraph;
import es.uam.eps.ir.knnbandit.recommendation.clusters.ClusteringAlgorithm;
import es.uam.eps.ir.knnbandit.recommendation.clusters.Clusters;
import es.uam.eps.ir.knnbandit.recommendation.clusters.ConnectedComponents;
import es.uam.eps.ir.knnbandit.utils.statistics.GiniIndex2;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.doubles.DoubleList;
import it.unimi.dsi.fastutil.ints.Int2IntMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2LongMap;
import it.unimi.dsi.fastutil.ints.Int2LongOpenHashMap;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Given dataset, finds an undirected and weighted graph indicating the existence of common rated (relevant) items.
 * Then, it provides a metric analysis, depending on the number of common items / the weights.
 *
 * The weight of an edge is computed as follows:
 * <ol>
 *      <li>We find the number of common items inter(u,v)</li>
 *      <li>We find the number of relevant items u and v do not have in common (|u|+|v|-2 inter(u,v))</li>
 *      <li>The weight is the square root of such value.</li>
 * </ol>
 *
 * This function returns the following values:
 * <ul>
 *     <li>Densities of the graphs</li>
 *     <li>Number of connected components</li>
 *     <li>Average weight of the edges</li>
 *     <li>The complement of the Gini index of the component sizes (NaN if there is a single component)</li>
 * </ul>
 *
 * @param <U> type of the users
 * @param <I> type of the items
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class DatasetGraphAnalysis<U,I>
{
    /**
     * Analyzes the different interaction graphs of a dataset according to the minimum number
     * of common ratings / the minimum weight value.
     * @param output the directory in which we want to store the results of the analysis.
     * @param limit the maximum number of common rated items to consider.
     * @param weights a comma separated list indicating the separating weight points.
     * @throws IOException if something fails while reading / writing the dataset.
     */
    public void analyze(String output, int limit, DoubleList weights) throws IOException
    {
        OfflineDataset<U,I> dataset = this.getDataset();
        System.out.println(dataset.toString());

        Map<Integer, Int2IntMap> map = new HashMap<>();
        Int2LongMap numRel = new Int2LongOpenHashMap();
        AtomicInteger atom = new AtomicInteger(0);


        // Find the number of common neighbors:
        dataset.getAllUidx().forEach(u ->
        {
            Int2IntOpenHashMap uMap = new Int2IntOpenHashMap();
            long rel = dataset.getUidxPreferences(u).mapToLong(i ->
            {
                dataset.getIidxPreferences(i.v1).filter(v -> !map.containsKey(v.v1)).forEach(v -> uMap.addTo(v.v1, 1));
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

        DoubleList densities = new DoubleArrayList();
        DoubleList numComp = new DoubleArrayList();
        DoubleList avgWeight = new DoubleArrayList();
        DoubleList compSizeGini = new DoubleArrayList();

        for(int i = 0; i <= limit; ++i)
        {
            System.out.println("------ Starting " + i + "-th graph");
            Graph<Integer> graph = new FastUndirectedWeightedGraph<>();
            dataset.getAllUidx().forEach(graph::addNode);

            int j = 0;
            double averageWeight = 0;
            long counter = 0;
            for (Map.Entry<Integer, Int2IntMap> entry : map.entrySet())
            {
                int u = entry.getKey();
                Int2IntMap uMap = entry.getValue();
                for (Map.Entry<Integer, Integer> uEntry : uMap.entrySet())
                {
                    int v = uEntry.getKey();
                    long val = uEntry.getValue();
                    if(val > i && u != v)
                    {
                        double realVal = Math.sqrt(numRel.get(u) + numRel.get(v) - 2*val);
                        averageWeight += realVal;
                        counter++;
                        graph.addEdge(u, v, realVal);
                    }
                }

                ++j;
                if (j % 100 == 0)
                {
                    System.out.println("Processed " + j + "users");
                }
            }

            avgWeight.add(averageWeight/(counter+0.0));
            densities.add(2.0*(counter+0.0)/(graph.getVertexCount()*(graph.getVertexCount()-1.0)));
            ClusteringAlgorithm<Integer> clust = new ConnectedComponents<>();
            Clusters<Integer> c = clust.detectClusters(graph);
            numComp.add(c.getNumClusters());

            // Find the Gini coefficient.
            Int2LongOpenHashMap sizes = new Int2LongOpenHashMap();
            c.getClusters().forEach(cc -> sizes.put(c.getNumElems(cc), (long) sizes.getOrDefault(c.getNumElems(cc), 1L)));
            GiniIndex2 gini = new GiniIndex2(c.getNumClusters(), sizes);
            compSizeGini.add(gini.getValue());

            System.out.println("------- Finished " + i + " graph");
        }

        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output+ "-neighs.txt"))))
        {
            bw.write("num\tdensity\tnumComp\tavgWeight\tGini");
            for(int i = 0; i < densities.size(); ++i)
            {
                bw.write("\n" + i + "\t" + densities.get(i) + "\t" + numComp.get(i) + "\t" + avgWeight.get(i) + "\t" + (1.0-compSizeGini.get(i)));
            }
        }


        densities.clear();
        numComp.clear();
        avgWeight.clear();
        compSizeGini.clear();

        for(double w : weights)
        {
            System.out.println("------ Starting " + w + "-th graph");
            Graph<Integer> graph = new FastUndirectedWeightedGraph<>();
            dataset.getAllUidx().forEach(graph::addNode);

            int j = 0;
            double averageWeight = 0;
            long counter = 0;

            for(int uidx : map.keySet())
            {
                Int2IntMap uMap = map.get(uidx);
                for(int vidx : map.keySet())
                {
                    long val = uMap.getOrDefault(vidx, 0);
                    double realVal = Math.sqrt(numRel.get(uidx) + numRel.get(vidx) - 2*val);
                    if(realVal < w && uidx != vidx)
                    {
                        averageWeight += realVal;
                        counter++;
                        graph.addEdge(uidx,vidx, realVal);
                    }
                }

                ++j;
                if (j % 100 == 0)
                {
                    System.out.println("Processed " + j + "users");
                }
            }

            avgWeight.add(averageWeight/(counter+0.0));
            densities.add((counter+0.0)/(graph.getVertexCount()*(graph.getVertexCount()-1.0)));
            ClusteringAlgorithm<Integer> clust = new ConnectedComponents<>();
            Clusters<Integer> c = clust.detectClusters(graph);
            numComp.add(c.getNumClusters());

            // Find the Gini coefficient.
            Int2LongOpenHashMap sizes = new Int2LongOpenHashMap();
            c.getClusters().forEach(cc -> sizes.put(c.getNumElems(cc), (long) sizes.getOrDefault(c.getNumElems(cc), 1L)));
            GiniIndex2 gini = new GiniIndex2(c.getNumClusters(), sizes);
            compSizeGini.add(gini.getValue());

            System.out.println("------- Finished " + w + " graph");
        }

        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output + "-weights.txt"))))
        {
            bw.write("num\tdensity\tnumComp\tavgWeight\tGini");
            for(int i = 0; i < densities.size(); ++i)
            {
                bw.write("\n" + i + "\t" + densities.get(i) + "\t" + numComp.get(i) + "\t" + avgWeight.get(i) + "\t" + (1.0-compSizeGini.get(i)));
            }
        }
    }

    /**
     * Obtains the dataset.
     * @return the dataset.
     */
    protected abstract OfflineDataset<U,I> getDataset();
}
