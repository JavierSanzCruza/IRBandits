/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.graph.generator;

import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.complementary.ComplementaryGraph;
import es.uam.eps.ir.knnbandit.graph.complementary.DirectedUnweightedComplementaryGraph;
import es.uam.eps.ir.knnbandit.graph.complementary.UndirectedUnweightedComplementaryGraph;
import es.uam.eps.ir.knnbandit.utils.generator.Generator;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Generates a complete graph.
 *
 * @author Javier Sanz-Cruzado Puig (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 * @param <U> type of the users
 *
 */
public class CompleteGraphGenerator<U> implements GraphGenerator<U>
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
     * @param generator Object that automatically creates the indicated number of nodes
     */
    public void configure(boolean directed, int numNodes, Generator<U> generator)
    {
        this.directed = directed;
        this.numNodes = numNodes;
        this.generator = generator;
        this.nodes = null;
        this.configured = true;
    }

    public void configure(boolean directed, Collection<U> nodes)
    {
        this.directed = directed;
        this.numNodes = nodes.size();
        this.generator = null;
        this.nodes = nodes;
        this.configured = true;
    }

    @SuppressWarnings("unchecked")
    @Override
    public void configure(Object... configuration)
    {
        if(!(configuration == null) && configuration.length == 3)
        {
            boolean auxDirected = (boolean) configuration[0];
            int auxNumNodes = (int) configuration[1];
            Generator<U> auxGenerator = (Generator<U>) configuration[2];

            configure(auxDirected, auxNumNodes, auxGenerator);
        }
        else if(!(configuration == null) && configuration.length == 2)
        {
            boolean auxDirected = (boolean) configuration[0];
            Collection<U> auxNodes = (Collection<U>) configuration[1];

            configure(auxDirected, auxNodes);
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
            throw new GeneratorNotConfiguredException("Complete Model: Generator was not configured");

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

        // To minimize the memory consumption, we define it as the complementary of an empty graph.
        return directed ? new DirectedUnweightedComplementaryGraph<>(graph) : new UndirectedUnweightedComplementaryGraph<>(graph);
    }


}

