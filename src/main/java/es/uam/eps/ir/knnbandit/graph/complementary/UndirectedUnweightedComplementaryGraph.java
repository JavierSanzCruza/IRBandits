/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.graph.complementary;

import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.UndirectedGraph;
import es.uam.eps.ir.knnbandit.graph.UnweightedGraph;

/**
 * Undirected unweighted complementary graph
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 * @param <U> type of the vertices.
 */
public class UndirectedUnweightedComplementaryGraph<U> extends ComplementaryGraph<U> implements UndirectedGraph<U>, UnweightedGraph<U>
{

    /**
     * Constructor.
     * @param graph Original graph. 
     */
    public UndirectedUnweightedComplementaryGraph(Graph<U> graph)
    {
        super(graph);
    }
    
}
