package es.uam.eps.ir.knnbandit.main.general.movielens;

import es.uam.eps.ir.knnbandit.data.datasets.ContactDataset;
import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.fast.FastUndirectedWeightedGraph;
import es.uam.eps.ir.knnbandit.recommendation.clusters.ClusteringAlgorithm;
import es.uam.eps.ir.knnbandit.recommendation.clusters.Clusters;
import es.uam.eps.ir.knnbandit.recommendation.clusters.ConnectedComponents;
import es.uam.eps.ir.knnbandit.utils.statistics.GiniIndex2;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.doubles.DoubleList;
import it.unimi.dsi.fastutil.ints.*;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Given a social network dataset, finds an undirected and weighted one indicating the existence of common rated (relevant) items.
 *
 * The weight of an edge is computed as follows:
 * - First, we find the number of common items inter(u,v)
 * - Second, we find the number of relevant items u and v do not have in common (union(u,v)-2 inter(u,v))
 * - Then, the weight is computed as the inverse of that quantity (1.0/(union(x,y)-2 inter(u,v) + 1), so that
 *   the weight value is equal to 1 if they have no uncommon relevant items.
 * In case there is no common relevant items, weight is automatically equal to zero to penalize this.
 */
public class DatasetGraphAnalysis
{
    /**
     * Program that, given a dataset, finds a network containing an edge if and only if two users have (positively) rated
     * the same items.
     * @param args Program arguments:
     *             <ul>
     *              <li><b>Input:</b> The original network.</li>
     *              <li><b>Output:</b> File in which to store the new network.</li>
     *              <li><b>Threshold:</b> The relevance threshold for the dataset.</li>
     *             </ul>
     */
    public static void main(String[] args) throws IOException
    {
        if(args.length < 2)
        {
            System.err.println("ERROR: Invalid arguments");
            System.err.println("\tData: the complete recommendation data");
            System.err.println("\tOutput: A weighted graph");
            System.err.println("\tDirected: true if the network is directed, false otherwise");
        }

        // First, we read the program parameters:
        String input = args[0];
        String output = args[1];
        double threshold = Parsers.dp.parse(args[2]);
        int limit = Parsers.ip.parse(args[3]);

        // Then, load the dataset.
        Dataset<Long, Long> dataset = Dataset.load(input, Parsers.lp, Parsers.lp, "::", (double x) -> x,  (double x) -> x >= threshold);
        System.out.println("Read the whole data");
        System.out.println(dataset.toString());

        FastPreferenceData<Long, Long> prefData = dataset.getPrefData();
        Map<Integer, Int2IntMap> map = new HashMap<>();
        Int2LongMap numRel = new Int2LongOpenHashMap();
        AtomicInteger atom = new AtomicInteger(0);


        // Find the number of common neighbors:
        prefData.getAllUidx().forEach(u ->
                                      {
                                          Int2IntOpenHashMap uMap = new Int2IntOpenHashMap();
                                          long rel = prefData.getUidxPreferences(u).mapToLong(i ->
                                                                                              {
                                                                                                  prefData.getIidxPreferences(i.v1).filter(v -> !map.containsKey(v.v1)).forEach(v -> uMap.addTo(v.v1, 1));
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
            prefData.getAllUidx().forEach(graph::addNode);

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

        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output+ "-neights.txt"))))
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

        double[] weights = {0.02,0.2,2,2,20,200};

        for(double w : weights)
        {
            System.out.println("------ Starting " + w + "-th graph");
            Graph<Integer> graph = new FastUndirectedWeightedGraph<>();
            prefData.getAllUidx().forEach(graph::addNode);

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
}
