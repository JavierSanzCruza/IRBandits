/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
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
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parser;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Stores a social network-based contact recommendation dataset.
 * @param <U> Type of the users
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ContactDataset<U> extends GeneralDataset<U, U>
{
    /**
     * The number of reciprocal links
     */
    private final int numRecipr;
    /**
     * Indicates whether the graph is directed or not.
     */
    private final boolean directed;

    /**
     * A true value indicates that reciprocal links are counted as one link. False differentiates them.
     */
    private final boolean notReciprocal;

    /**
     * Constructor.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param prefData  Preference data.
     * @param numEdges  Number of edges
     * @param numRecipr Number of reciprocal edges.
     */
    protected ContactDataset(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<U> iIndex, SimpleFastPreferenceData<U, U> prefData, int numEdges, int numRecipr, boolean directed, boolean notReciprocal)
    {
        super(uIndex, iIndex, prefData, numEdges, x -> x > 0);
        this.numRecipr = numRecipr;
        this.directed = directed;
        this.notReciprocal = notReciprocal;
    }

    /**
     * Gets the number of relevant (user, user) pairs.
     *
     * @return the number of relevant (user, user) pairs.
     */
    @Override
    public int getNumRel()
    {
        return (notReciprocal ? this.numRel - this.numRecipr / 2 : this.numRel);
    }

    /**
     * Obtains whether the underlying social network is directed or not.
     * @return true if it is directed, false otherwise.
     */
    public boolean isDirected()
    {
        return directed;
    }

    /**
     * Obtains the value determining whether we shall recommend reciprocal edges
     * to existing ones (i.e. we take reciprocal edges separately) or not.
     * @return true if reciprocal edges are treated separately, false otherwise.
     */
    public boolean useReciprocal()
    {
        return !notReciprocal;
    }

    @Override
    public String toString()
    {
        return "Users: " +
                this.numUsers() +
                "\nItems: " +
                this.numItems() +
                "\nNum. edges: " +
                this.numRel +
                "\nNum. edges (without reciprocal): " +
                (this.numRel - this.numRecipr / 2);
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
    public static <U> ContactDataset<U> load(String filename, boolean directed, boolean notReciprocal, Parser<U> uParser, String separator)
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

        return new ContactDataset<>(uIndex, iIndex, prefData, numEdges, numRecipr, directed, notReciprocal);
    }

    /**
     * Loads a contact recommendation dataset from another dataset.
     * @param dataset the dataset.
     * @param list a list of (user, item) interactions.
     * @param notReciprocal true if we cannot recommend reciprocal links to existing ones.
     * @param <U> type of the users
     * @return the new contact recommendation dataset.
     */
    public static <U> ContactDataset<U> load(ContactDataset<U> dataset, List<Pair<Integer>> list, boolean notReciprocal)
    {
        // We build the preference data.
        List<Tuple3<U, U, Double>> validationTriplets = new ArrayList<>();
        Graph<U> graph = (dataset.isDirected() ? new FastDirectedUnweightedGraph<>() : new FastUndirectedUnweightedGraph<>());
        dataset.userIndex.getAllUsers().forEach(graph::addNode);
        SimpleFastPreferenceData<U, U> prefData = dataset.prefData;

        list.forEach(tuple ->
        {
            int uidx = tuple.v1();
            int iidx = tuple.v2();
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
        SimpleFastPreferenceData<U, U> validData = SimpleFastPreferenceData.load(validationTriplets.stream(), dataset.userIndex, dataset.itemIndex);

        return new ContactDataset<>(dataset.userIndex, dataset.itemIndex, validData, numEdges, numRecipr, dataset.isDirected(), notReciprocal);
        // Create the validation data, which will be provided as input to recommenders and metrics.
    }

    @Override
    public int getNumRatings()
    {
        return this.getNumRel();
    }

    @Override
    public Dataset<U,U> load(List<Pair<Integer>> pairs)
    {
        return ContactDataset.load(this, pairs);
    }

}
