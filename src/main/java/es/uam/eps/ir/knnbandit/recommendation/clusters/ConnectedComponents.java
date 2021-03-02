/* 
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es
 * 
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.clusters;

import es.uam.eps.ir.knnbandit.graph.Graph;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 * Computes communities via the Weakly Connected Components
 * @author Pablo Castells Azpilicueta (pablo.castells@uam.es)
 * @author Javier Sanz-Cruzado Puig (javier.sanz-cruzado@uam.es)
 * @param <U> Type of the users.
 */
public class ConnectedComponents<U> implements ClusteringAlgorithm<U>
{
    @Override
    public ClustersImpl<U> detectClusters(Graph<U> graph)
    {
        Collection<Collection<U>> scc = this.findWCC(graph);
        return new ClustersImpl<>(scc);
    }

    @Override
    public ClustersImpl<U> detectClusters(Graph<U> graph, Set<U> nodes)
    {
        Collection<Collection<U>> scc = this.findWCC(graph, nodes);
        ClustersImpl<U> comm = new ClustersImpl<>();

        int i = 0;
        for(Collection<U> cc : scc)
        {
            comm.addCluster();
            for(U u : cc)
            {
                comm.add(u, i);
            }
            ++i;
        }
        return comm;
    }

    private Collection<Collection<U>> findWCC(Graph<U> g, Set<U> nodes)
    {
        Set<U> discovered = new HashSet<>();
        Collection<Collection<U>> components = new ArrayList<>();

        nodes.forEach(u ->
        {
            if(!discovered.contains(u) && g.containsVertex(u))
            {
                Collection<U> component = new HashSet<>();
                visit(u, g, discovered, component, nodes);
                components.add(component);
            }
        });

        return components;
    }
    
    /**
     * Finds the weakly connected components of the graph.
     * @param g The graph
     * @return The weakly connected clusters of the graph.
     */
    private Collection<Collection<U>> findWCC (Graph<U> g)
    {
        Set<U> discovered = new HashSet<>();
        Collection<Collection<U>> components = new ArrayList<>();
        g.getAllNodes().forEach(u ->
        {
            if(!discovered.contains(u))
            {
                Collection<U> component = new HashSet<U>()
                {
                    @Override
                    public boolean equals(Object obj)
                    {
                        return this == obj;
                    }
                };
                visit(u, g, discovered, component);
                components.add(component);
            }
        });

        return components;
    }

    /**
     * Visits a node by using the inlinks
     * @param u The starting node
     * @param g The graph
     * @param discovered The dsiscovered items
     * @param component The component
     */
    private void visit (U u, Graph<U> g, Set<U> discovered, Collection<U> component, Set<U> nodes)
    {
        discovered.add(u);

        if(nodes.contains(u))
        {
            component.add(u);
        }

        g.getNeighbourNodes(u).forEach(v ->
        {
            if (!discovered.contains(v)) visit(v, g, discovered, component);
        });
    }

    /**
     * Visits a node by using the inlinks
     * @param u The starting node
     * @param g The graph
     * @param discovered The dsiscovered items
     * @param component The component
     */
    private void visit (U u, Graph<U> g, Set<U> discovered, Collection<U> component)
    {
        component.add(u);
        discovered.add(u);
        g.getNeighbourNodes(u).forEach(v -> 
        {
            if (!discovered.contains(v)) visit(v, g, discovered, component);
        });            
    }
}
