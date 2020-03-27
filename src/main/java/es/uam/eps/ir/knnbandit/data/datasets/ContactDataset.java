package es.uam.eps.ir.knnbandit.data.datasets;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.fast.FastDirectedUnweightedGraph;
import es.uam.eps.ir.knnbandit.graph.fast.FastUndirectedUnweightedGraph;
import es.uam.eps.ir.knnbandit.graph.io.GraphReader;
import es.uam.eps.ir.knnbandit.graph.io.TextGraphReader;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.parsing.Parsers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

public class ContactDataset<U> extends Dataset<U, U>
{

    private final int numRecipr;
    private final boolean directed;

    /**
     * Constructor.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param prefData  Preference data.
     * @param numEdges  Number of edges
     * @param numRecipr Number of reciprocal edges.
     */
    protected ContactDataset(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<U> iIndex, SimpleFastPreferenceData<U, U> prefData, int numEdges, int numRecipr, boolean directed)
    {
        super(uIndex, iIndex, prefData, numEdges, x -> x > 0);
        this.numRecipr = numRecipr;
        this.directed = directed;
    }

    /**
     * Gets the number of relevant (user, user) pairs.
     *
     * @param notReciprocal true if we do not count the reciprocal, false otherwise.
     * @return the number of relevant (user, user) pairs.
     */
    public int getNumRel(boolean notReciprocal)
    {
        return (notReciprocal ? this.numRel - this.numRecipr / 2 : this.numRel);
    }

    public boolean isDirected()
    {
        return directed;
    }

    @Override
    public String toString()
    {
        return "Users: " +
                this.numUsers() +
                "\nItems: " +
                this.numItems() +
                "\nNum. edges: " +
                this.getNumRel(false) +
                "\nNum. edges (without reciprocal): " +
                this.getNumRel(true);
    }

    /**
     * Loads the dataset.
     *
     * @param filename  name of the file containing the dataset.
     * @param directed  true if the graph is directed, false otherwise
     * @param uParser   parser for the user type.
     * @param separator file delimiter characters.
     * @param <U>       type of the users.
     * @return the contact recommendation dataset.
     */
    public static <U> ContactDataset<U> load(String filename, boolean directed, Parser<U> uParser, String separator)
    {
        // Read the ratings.
        Set<U> users = new HashSet<>();
        List<Tuple3<U, U, Double>> triplets = new ArrayList<>();

        Graph<U> graph;
        GraphReader<U> greader = new TextGraphReader<>(directed, false, false, separator, uParser);
        graph = greader.read(filename);

        graph.getAllNodes().forEach(users::add);
        int numEdges = ((int) graph.getEdgeCount()) * (directed ? 1 : 2);
        int numRecipr = graph.getAllNodes().mapToInt(graph::getMutualNodesCount).sum();

        graph.getAllNodes().forEach(u -> graph.getAdjacentNodes(u).forEach(v -> triplets.add(new Tuple3<>(u, v, 1.0))));

        FastUpdateableUserIndex<U> uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        FastUpdateableItemIndex<U> iIndex = SimpleFastUpdateableItemIndex.load(users.stream());
        SimpleFastPreferenceData<U, U> prefData = SimpleFastPreferenceData.load(triplets.stream(), uIndex, iIndex);

        return new ContactDataset<>(uIndex, iIndex, prefData, numEdges, numRecipr, directed);
    }


    public static <U> ContactDataset<U> load(ContactDataset<U> dataset, List<Tuple2<Integer, Integer>> list, boolean notReciprocal)
    {
        // We build the preference data.
        List<Tuple3<U, U, Double>> validationTriplets = new ArrayList<>();
        Graph<U> graph = (dataset.isDirected() ? new FastDirectedUnweightedGraph<>() : new FastUndirectedUnweightedGraph<>());
        dataset.getUserIndex().getAllUsers().forEach(graph::addNode);
        SimpleFastPreferenceData<U, U> prefData = dataset.getPrefData();

        list.forEach(tuple ->
                     {
                         int uidx = tuple.v1;
                         int iidx = tuple.v2;
                         U u = prefData.uidx2user(uidx);
                         U i = prefData.iidx2item(iidx);

                         if (prefData.numItems(uidx) > 0 && prefData.numUsers(iidx) > 0 && prefData.getPreference(uidx, iidx).isPresent())
                         {
                             validationTriplets.add(new Tuple3<>(prefData.uidx2user(uidx), prefData.iidx2item(iidx), 1.0));
                             graph.addEdge(u, i);
                             if (notReciprocal && prefData.numItems(iidx) > 0 && prefData.numUsers(uidx) > 0 && prefData.getPreference(iidx, uidx).isPresent())
                             {
                                 validationTriplets.add(new Tuple3<>(prefData.uidx2user(iidx), prefData.iidx2item(uidx), 1.0));
                                 graph.addEdge(i, u);
                             }
                         }
                     });

        int numEdges = ((int) graph.getEdgeCount()) * (dataset.isDirected() ? 1 : 2);
        int numRecipr = graph.getAllNodes().mapToInt(graph::getMutualNodesCount).sum();
        SimpleFastPreferenceData<U, U> validData = SimpleFastPreferenceData.load(validationTriplets.stream(), dataset.getUserIndex(), dataset.getItemIndex());

        return new ContactDataset<>(dataset.getUserIndex(), dataset.getItemIndex(), validData, numEdges, numRecipr, dataset.isDirected());
        // Create the validation data, which will be provided as input to recommenders and metrics.
    }

}
