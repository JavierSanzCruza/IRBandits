/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.graph.fast;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import es.uam.eps.ir.knnbandit.graph.UndirectedUnweightedGraph;
import es.uam.eps.ir.knnbandit.graph.edges.EdgeOrientation;
import es.uam.eps.ir.knnbandit.graph.edges.fast.FastUndirectedUnweightedEdges;
import es.uam.eps.ir.knnbandit.graph.index.fast.FastIndex;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

/**
 * Fast implementation of an undirected unweighted graph.
 *
 * @param <V> Type of the vertices.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class FastUndirectedUnweightedGraph<V> extends FastGraph<V> implements UndirectedUnweightedGraph<V>
{
    /**
     * Constructor.
     */
    public FastUndirectedUnweightedGraph()
    {
        super(new FastIndex<>(), new FastUndirectedUnweightedEdges());
    }

    @Override
    public DoubleMatrix2D getAdjacencyMatrix(EdgeOrientation direction)
    {
        DoubleMatrix2D matrix = new SparseDoubleMatrix2D(Long.valueOf(this.getVertexCount()).intValue(), Long.valueOf(this.getVertexCount()).intValue());

        // Creation of the adjacency matrix.
        for (int row = 0; row < matrix.rows(); ++row)
        {
            for (int col = 0; col < matrix.rows(); ++col)
            {

                if (this.containsEdge(this.vertices.idx2object(col), this.vertices.idx2object(row)) ||
                        this.containsEdge(this.vertices.idx2object(row), this.vertices.idx2object(col)))
                {
                    matrix.setQuick(row, col, 1.0);
                }

            }
        }

        return matrix;
    }

    @Override
    public Matrix getAdjacencyMatrixMTJ(EdgeOrientation direction)
    {
        Matrix matrix = new LinkedSparseMatrix(Long.valueOf(this.getVertexCount()).intValue(), Long.valueOf(this.getVertexCount()).intValue());
        this.vertices.getAllObjects().forEach(u ->
                                              {
                                                  int uIdx = this.vertices.object2idx(u);
                                                  this.getNeighbourNodes(u).forEach(v ->
                                                                                    {
                                                                                        int vIdx = this.vertices.object2idx(v);
                                                                                        matrix.set(uIdx, vIdx, 1.0);
                                                                                    });
                                              });

        return matrix;
    }
}
