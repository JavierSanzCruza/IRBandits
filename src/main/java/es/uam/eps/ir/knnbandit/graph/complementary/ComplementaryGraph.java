/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.graph.complementary;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.Weight;
import es.uam.eps.ir.knnbandit.graph.edges.EdgeOrientation;
import es.uam.eps.ir.knnbandit.graph.edges.EdgeType;
import es.uam.eps.ir.knnbandit.graph.edges.EdgeWeight;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

import java.util.stream.Stream;

/**
 * Wrapper for the complementary graph of another one given. Since this is the complementary
 * of another graph, no nodes nor edges can be removed. Every time the original graph is modified,
 * so this graph will be.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @param <U> type of the users.
 */
public abstract class ComplementaryGraph<U> implements Graph<U>
{
    /**
     * Original graph.
     */
    private final Graph<U> graph;
    
    /**
     * Constructor
     * @param graph Original graph. 
     */
    public ComplementaryGraph(Graph<U> graph)
    {
        this.graph = graph;
    }
    
    // Addition and removal of nodes and edges. NOT ALLOWED.
    
    @Override
    public boolean addNode(U node)
    {
        throw new UnsupportedOperationException("This is a complementary graph. No nodes can be added."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean addEdge(U nodeA, U nodeB, double weight, int type, boolean insertNodes)
    {
        throw new UnsupportedOperationException("This is a complementary graph. No edges can be added."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean removeNode(U node)
    {
        throw new UnsupportedOperationException("This is a complementary graph. No nodes can be removed");
    }
    
    @Override
    public boolean removeEdge(U nodeA, U nodeB)
    {
        throw new UnsupportedOperationException("This is a complementary graph. No edges can be removed");
    }
    
    // Getters
    
    @Override
    public Stream<U> getAllNodes()
    {
        return this.graph.getAllNodes();
    }

    @Override
    public Stream<U> getIncidentNodes(U node)
    {
        return this.graph.getAllNodes().filter(v -> !this.graph.containsEdge(v, node));
    }

    @Override
    public Stream<U> getAdjacentNodes(U node)
    {
        return this.graph.getAllNodes().filter(v -> !this.graph.containsEdge(node, v));
    }

    @Override
    public Stream<U> getNeighbourNodes(U node)
    {
        return this.graph.getAllNodes().filter(v -> !this.graph.isMutual(v, node));
    }

    @Override
    public Stream<U> getMutualNodes(U node)
    {
        return this.graph.getAllNodes().filter(v -> !this.graph.containsEdge(v, node) && !this.graph.containsEdge(node, v));
    }
    
    @Override
    public Stream<U> getNeighbourhood(U node, EdgeOrientation direction)
    {
        switch(direction)
        {
            case OUT:
                return this.getAdjacentNodes(node);
            case IN:
                return this.getIncidentNodes(node);
            case MUTUAL:
                return this.getMutualNodes(node);
            default: //case UND
                return this.getNeighbourNodes(node);
        }
    }
    
    @Override
    public int getIncidentEdgesCount(U node)
    {
        return (int) (this.getVertexCount() - this.graph.getIncidentEdgesCount(node));
    }

    @Override
    public int getAdjacentEdgesCount(U node)
    {
        return (int) (this.getVertexCount() - this.graph.getAdjacentEdgesCount(node));
    }

    @Override
    public int getNeighbourEdgesCount(U node)
    {
        return (int) (this.getVertexCount() - this.graph.getMutualEdgesCount(node));
    }
    
    @Override
    public int getMutualEdgesCount(U node)
    {
        return (int) (this.getVertexCount() - this.graph.getNeighbourEdgesCount(node));
    }

    @Override
    public int getNeighbourhoodSize(U node, EdgeOrientation direction)
    {
        switch(direction)
        {
            case OUT:
                return this.getAdjacentEdgesCount(node);
            case IN:
                return this.getIncidentEdgesCount(node);
            case MUTUAL:
                return this.getMutualEdgesCount(node);
            default: //case UND
                return this.getNeighbourEdgesCount(node);
        }
    }

    @Override
    public boolean containsVertex(U node)
    {
        return this.graph.containsVertex(node);
    }

    @Override
    public boolean containsEdge(U nodeA, U nodeB)
    {
        if(this.containsVertex(nodeA) && this.containsVertex(nodeB))
            return !this.graph.containsEdge(nodeA, nodeB);
        return false;
    }

    @Override
    public double getEdgeWeight(U nodeA, U nodeB)
    {
        if(!this.graph.containsEdge(nodeA, nodeB))
            return EdgeWeight.getDefaultValue();
        return EdgeWeight.getErrorValue();
    }

    @Override
    public Stream<Weight<U, Double>> getIncidentNodesWeights(U node)
    {
        return this.getIncidentNodes(node).map(vertex -> new Weight<>(vertex, EdgeWeight.getDefaultValue()));
    }

    @Override
    public Stream<Weight<U, Double>> getAdjacentNodesWeights(U node)
    {
        return this.getAdjacentNodes(node).map(vertex -> new Weight<>(vertex, EdgeWeight.getDefaultValue()));
    }

    @Override
    public Stream<Weight<U, Double>> getNeighbourNodesWeights(U node)
    {
        return this.getNeighbourNodes(node).map(vertex -> new Weight<>(vertex, EdgeWeight.getDefaultValue()));
    }
    
    @Override
    public Stream<Weight<U,Double>> getMutualNodesWeights(U node)
    {
        return this.getMutualNodes(node).map(vertex -> new Weight<>(vertex, EdgeWeight.getDefaultValue()));
    }
    
    @Override
    public Stream<Weight<U,Double>> getAdjacentMutualNodesWeights(U node)
    {
        return this.getMutualNodes(node).map(vertex -> new Weight<>(vertex, EdgeWeight.getDefaultValue()));
    }
    
    @Override
    public Stream<Weight<U,Double>> getIncidentMutualNodesWeights(U node)
    {
        return this.getMutualNodes(node).map(vertex -> new Weight<>(vertex, EdgeWeight.getDefaultValue()));
    }

    @Override
    public Stream<Weight<U, Double>> getNeighbourhoodWeights(U node, EdgeOrientation direction)
    {
        switch(direction)
        {
            case OUT:
                return this.getAdjacentNodesWeights(node);
            case IN:
                return this.getIncidentNodesWeights(node);
            case MUTUAL:
                return this.getMutualNodesWeights(node);
            default: //case UND
                return this.getNeighbourNodesWeights(node);
        }
    }

    @Override
    public int getEdgeType(U nodeA, U nodeB)
    {
        if(this.containsEdge(nodeA, nodeB))
            return EdgeType.getDefaultValue();
        else
            return EdgeType.getErrorType();
    }

    @Override
    public Stream<Weight<U, Integer>> getIncidentNodesTypes(U node)
    {
        return this.getIncidentNodes(node).map(vertex -> new Weight<>(vertex, EdgeType.getDefaultValue()));
    }

    @Override
    public Stream<Weight<U, Integer>> getAdjacentNodesTypes(U node)
    {
        return this.getAdjacentNodes(node).map(vertex -> new Weight<>(vertex, EdgeType.getDefaultValue()));
    }

    @Override
    public Stream<Weight<U, Integer>> getNeighbourNodesTypes(U node)
    {
        return this.getNeighbourNodes(node).map(vertex -> new Weight<>(vertex, EdgeType.getDefaultValue()));
    }
    
    @Override
    public Stream<Weight<U,Integer>> getAdjacentMutualNodesTypes(U node)
    {
        return this.getMutualNodes(node).map(vertex -> new Weight<>(vertex, EdgeType.getDefaultValue()));
    }
    
    @Override
    public Stream<Weight<U,Integer>> getIncidentMutualNodesTypes(U node)
    {
        return this.getMutualNodes(node).map(vertex -> new Weight<>(vertex, EdgeType.getDefaultValue()));
    }

    @Override
    public Stream<Weight<U, Integer>> getNeighbourhoodTypes(U node, EdgeOrientation direction)
    {
        switch(direction)
        {
            case OUT:
                return this.getAdjacentNodesTypes(node);
            case IN:
                return this.getIncidentNodesTypes(node);
            default: //case UND
                return this.getNeighbourNodesTypes(node);
        }
    }

    @Override
    public boolean isDirected()
    {
        return this.graph.isDirected();
    }

    @Override
    public boolean isWeighted()
    {
        return this.graph.isWeighted();
    }

    @Override
    public long getVertexCount()
    {
        return this.graph.getVertexCount();
    }

    @Override
    public long getEdgeCount()
    {
        if(this.isMultigraph())
        {
            throw new UnsupportedOperationException("The number of edges of a multigraph cannot be computed");
        }
        return this.getVertexCount()*this.getVertexCount() - this.graph.getEdgeCount();
    }

    @Override
    public DoubleMatrix2D getAdjacencyMatrix(EdgeOrientation direction)
    {
        if(this.isMultigraph())
        {
            throw new UnsupportedOperationException("The complementary adjacency matrix of a multigraph cannot be computed");
        }
        
        DoubleMatrix2D matrix = new SparseDoubleMatrix2D(Long.valueOf(this.getVertexCount()).intValue(), Long.valueOf(this.getVertexCount()).intValue());
        
        DoubleMatrix2D complAdjMatrix = this.graph.getAdjacencyMatrix(direction);
        
        matrix.assign(complAdjMatrix, (double a, double b) -> 1.0-b);
        
        return matrix;
    }

    @Override
    public Matrix getAdjacencyMatrixMTJ(EdgeOrientation direction)
    {
        if(this.isMultigraph())
        {
            throw new UnsupportedOperationException("The complementary adjacency matrix of a multigraph cannot be computed");
        }
        
        Matrix matrix = new LinkedSparseMatrix(Long.valueOf(this.getVertexCount()).intValue(), Long.valueOf(this.getVertexCount()).intValue());
        Matrix complAdjMatrix = this.graph.getAdjacencyMatrixMTJ(direction);
        for(int i = 0; i < this.getVertexCount(); ++i)
        {
            for(int j = 0; j < this.getVertexCount(); ++j)
            {
                matrix.set(i,j,1.0 - complAdjMatrix.get(i, j));
            }
        }
        return matrix;
    }
    
    @Override
    public boolean updateEdgeWeight(U orig, U dest, double weight)
    {
        throw new UnsupportedOperationException("Edges weights cannot be updated in complementary graphs");
    }
    
    @Override
    public int object2idx(U u)
    {
        return this.graph.object2idx(u);
    }
    
    @Override
    public U idx2object(int idx)
    {
        return this.graph.idx2object(idx);
    }
    
    @Override
    public Stream<U> getIsolatedNodes() 
    {
        return this.graph.getAllNodes().filter(x -> this.getNeighbourEdgesCount(x) == 0);
    }

    @Override
    public Stream<U> getNodesWithEdges(EdgeOrientation direction)
    {
        return this.graph.getAllNodes().filter(x -> this.getNeighbourhoodSize(x, direction) > 0);
    }
    
    @Override
    public Stream<U> getNodesWithAdjacentEdges() 
    {
        return this.graph.getAllNodes().filter(x -> this.getAdjacentNodesCount(x) > 0);
    }

    @Override
    public Stream<U> getNodesWithIncidentEdges() {
        return this.graph.getAllNodes().filter(x -> this.getIncidentNodesCount(x) > 0);
    }

    @Override
    public Stream<U> getNodesWithEdges() {
        return this.graph.getAllNodes().filter(x -> this.getNeighbourNodesCount(x) > 0);
    }

    @Override
    public Stream<U> getNodesWithMutualEdges() {
        return this.graph.getAllNodes().filter(x -> this.getMutualNodesCount(x) > 0);
    }

    @Override
    public boolean hasAdjacentEdges(U u) {
        return this.getAdjacentNodesCount(u) > 0;
    }

    @Override
    public boolean hasIncidentEdges(U u) {
        return this.getIncidentNodesCount(u) > 0;
    }

    @Override
    public boolean hasEdges(U u) {
        return this.getNeighbourNodesCount(u) > 0;
    }

    @Override
    public boolean hasMutualEdges(U u) {
        return this.getMutualNodesCount(u) > 0;
    }
    
}
