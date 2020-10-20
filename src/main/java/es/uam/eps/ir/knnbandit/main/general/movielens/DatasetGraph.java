package es.uam.eps.ir.knnbandit.main.general.movielens;

import es.uam.eps.ir.knnbandit.data.datasets.GeneralDataset;
import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.fast.FastUndirectedWeightedGraph;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import it.unimi.dsi.fastutil.longs.Long2LongMap;
import it.unimi.dsi.fastutil.longs.Long2LongOpenHashMap;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Given a Movielens dataset, finds an undirected and weighted one indicating the existence of common rated (relevant) items.
 *
 * The weight of an edge is computed as follows:
 * - First, we find the number of common items inter(u,v)
 * - Second, we find the number of relevant items u and v do not have in common (union(u,v)-2 inter(u,v))
 * - Then, the weight is computed as the inverse of that quantity (1.0/(union(x,y)-2 inter(u,v) + 1), so that
 *   the weight value is equal to 1 if they have no uncommon relevant items.
 * In case there is no common relevant items, weight is automatically equal to zero to penalize this.
 */
public class DatasetGraph
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

        String input = args[0];
        String output = args[1];
        double threshold = Parsers.dp.parse(args[2]);

        // First, load the dataset.
        GeneralDataset<Long, Long> dataset = GeneralDataset.load(input, Parsers.lp, Parsers.lp, "::", (double x) -> x, (double x) -> x >= threshold);
        System.out.println("Read the whole data");
        System.out.println(dataset.toString());

        PreferenceData<Long, Long> prefData = dataset.getPrefData();
        Graph<Long> graph = new FastUndirectedWeightedGraph<>();

        // First, we add all the nodes to the graph.
        prefData.getAllUsers().forEach(graph::addNode);

        Map<Long, Long2LongMap> map = new HashMap<>();
        Long2LongMap numRel = new Long2LongOpenHashMap();
        AtomicInteger atom = new AtomicInteger(0);

        // Second, we find the network:
        prefData.getAllUsers().forEach(u ->
        {
            Long2LongOpenHashMap uMap = new Long2LongOpenHashMap();
            long rel = prefData.getUserPreferences(u).filter(i -> i.v2 >= threshold).mapToLong(i ->
            {
                prefData.getItemPreferences(i.v1).filter(v -> !map.containsKey(v.v1) && v.v2 >= threshold).forEach(v -> uMap.addTo(v.v1, 1));
                return 1;
            }).sum();
            map.put(u, uMap);
            numRel.put((long) u, rel);

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
            for (Map.Entry<Long, Long2LongMap> entry : map.entrySet())
            {
                long u = entry.getKey();
                Long2LongMap uMap = entry.getValue();
                for (Map.Entry<Long, Long> uEntry : uMap.entrySet())
                {
                    long v = uEntry.getKey();
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
}
