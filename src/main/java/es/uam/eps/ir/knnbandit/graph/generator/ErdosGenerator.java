/*
 *  Copyright (C) 2015 Information Retrieval Group at Universidad AutÃ³noma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.graph.generator;

import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.utils.generator.Generator;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Generates a random graph following the Erdös-Renyi model.
 *
 * Erdös, P., Rényi, A. On Random Graphs. I, Publicationes Mathematicae Debrecen 6, 1959, pp. 290-297
 * @author Javier Sanz-Cruzado Puig
 * @param <U> type of the users
 *
 */
public class ErdosGenerator<U> implements GraphGenerator<U>
{
    /**
     * Indicates if the graph is directed or not.
     */
    private boolean directed;
    /**
     * Number of nodes of the graph.
     */
    private int numNodes;
    /**
     * Probability of joining two edges.
     */
    private double prob;
    /**
     * User generator.
     */
    private Generator<U> generator;
    /**
     * Set of nodes.
     */
    private Collection<U> nodes;
    /**
     * Indicates if the generator has been configured or not.
     */
    private boolean configured = false;

    /**
     * Configures the Erdos graph.
     * @param directed Indicates if the graph edges are directed or not.
     * @param numNodes Number of nodes of the graph.
     * @param prob Probability of joining two edges.
     * @param generator Object that automatically creates the indicated number of nodes
     */
    public void configure(boolean directed, int numNodes, double prob, Generator<U> generator)
    {
        this.directed = directed;
        this.numNodes = numNodes;
        this.prob = prob;
        this.generator = generator;
        this.nodes = null;
        this.configured = true;
    }

    public void configure(boolean directed, double prob, Collection<U> nodes)
    {
        this.directed = directed;
        this.numNodes = nodes.size();
        this.prob = prob;
        this.generator = null;
        this.nodes = nodes;
        this.configured = true;
    }

    @SuppressWarnings("unchecked")
    @Override
    public void configure(Object... configuration)
    {
        if(!(configuration == null) && configuration.length == 4)
        {
            boolean auxDirected = (boolean) configuration[0];
            int auxNumNodes = (int) configuration[1];
            double auxProb = (double) configuration[2];
            Generator<U> auxGenerator = (Generator<U>) configuration[3];

            configure(auxDirected, auxNumNodes, auxProb, auxGenerator);
        }
        else if(!(configuration == null) && configuration.length == 3)
        {
            boolean auxDirected = (boolean) configuration[0];
            double auxProb = (double) configuration[1];
            Collection<U> auxNodes = (Collection<U>) configuration[2];

            configure(auxDirected, auxProb, auxNodes);
        }
        else
        {
            configured = false;
        }

    }


    @Override
    public Graph<U> generate() throws GeneratorNotConfiguredException
    {
        if(!configured)
            throw new GeneratorNotConfiguredException("Erdos Model: Generator was not configured");

        EmptyGraphGenerator<U> gen = new EmptyGraphGenerator<>();
        gen.configure(directed, false);
        Graph<U> graph = gen.generate();

        Random rand = new Random();

        // Step 1: create the nodes
        if(this.generator != null) // Generate the nodes
        {
            for (int i = 0; i < numNodes; ++i)
            {
                graph.addNode(generator.generate());
            }
        }
        else // Just add them.
        {
            for(U node : nodes)
            {
                graph.addNode(node);
            }
        }

        List<U> nodes = graph.getAllNodes().collect(Collectors.toCollection(ArrayList::new));
        double numEdges = Math.ceil(prob*numNodes*(numNodes-1));

        for(int i = 0; i < numEdges; ++i)
        {
            U node1 = nodes.get(rand.nextInt(nodes.size()));
            U node2 = nodes.get(rand.nextInt(nodes.size()));

            if(!graph.containsEdge(node1, node2))
            {
                graph.addEdge(node1, node2);
            }
            else
            {
                --i;
            }
        }
        return graph;
    }


}

