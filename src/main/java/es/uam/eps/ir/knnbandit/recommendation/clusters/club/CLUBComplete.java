/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.clusters.club;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.graph.generator.*;
import java.util.stream.Collectors;

/**
 * Implementation of the CLUstering of Bandits algorithm.
 * We assume that context vectors for the different items are versors (unit vectors)
 * and that we recommend all the possible candidate items.
 *
 * A more complex vector considers different variants for the arm context.
 *
 * Original implementation of the algorithm, taking the complete user graph from the start.
 * <p>
 *     <b>Reference: </b> C. Gentile, S. Li, G. Zapella. Online clustering of bandits. 29th conference on Neural Information Processing Systems (NeurIPS 2015). Montréal, Canada (2015).
 * </p>
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class CLUBComplete<U ,I> extends AbstractCLUB<U,I>
{
    /**
     * Constructor.
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param alpha1 Parameter that manages the importance of the confidence bound for the item selection.
     * @param alpha2 Parameter that manages how difficult is for an edge in the graph to disappear.
     */
    public CLUBComplete(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, ignoreNotRated, alpha1, alpha2);
    }

    /**
     * Constructor.
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param alpha1 Parameter that manages the importance of the confidence bound for the item selection.
     * @param alpha2 Parameter that manages how difficult is for an edge in the graph to disappear.
     */
    public CLUBComplete(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed, alpha1, alpha2);
    }

    @Override
    protected GraphGenerator<Integer> configureGenerator()
    {
        GraphGenerator<Integer> ggen = new CompleteGraphGenerator<>();
        ggen.configure(false, this.uIndex.getAllUidx().boxed().collect(Collectors.toList()));
        return ggen;
    }
}
