/* 
 * Copyright (C) 2018 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es
 * 
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.clusters;


import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.utils.Pair;

import java.util.List;
import java.util.Set;

/**
 * Algorithm for detecting the communities of a graph.
 *
 * @author Javier Sanz-Cruzado Puig (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @param <U> Type of the users.
 */
public interface ClusteringAlgorithm<U>
{
    /**
     * Computes the communities for a certain graph.
     * @param graph The full graph.
     * @return The communities if everything went OK, null if not.
     */
    ClustersImpl<U> detectClusters(Graph<U> graph);

    /**
     * Computes the cluster division for a subset of the nodes in the network.
     * @param graph The full graph.
     * @param nodes The nodes to consider.
     * @return The cluster division of the nodes if everything went OK, null otherwise. If a node does not appear in the network,
     * it will not appear in the division.
     */
    ClustersImpl<U> detectClusters(Graph<U> graph, Set<U> nodes);

    /**
     * Computes the communities for a certain graph, given a previous partition.Used for evolution of networks.
     * @param graph The full graph.
     * @param newLinks The links which have newly appeared in the graph.
     * @param disapLinks The links which have disappeared from the graph.
     * @param previous the previous community partition
     * @return the new community partition.
     */
    default ClustersImpl<U> detectClusters(Graph<U> graph, List<Pair<U>> newLinks, List<Pair<U>> disapLinks, ClustersImpl<U> previous)
    {
        return this.detectClusters(graph);
    }
}
