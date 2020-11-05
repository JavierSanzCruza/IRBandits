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
import es.uam.eps.ir.knnbandit.graph.generator.ErdosGenerator;
import es.uam.eps.ir.knnbandit.graph.generator.GraphGenerator;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Implementation of the CLUstering of Bandits algorithm.
 * We assume that context vectors for the different items are versors (unit vectors)
 * and that we recommend all the possible candidate items.
 *
 * Following the theoretical analysis from the original paper, we consider sparser
 * Erdös-Renyi graphs instead of empty ones. We use the default value in their experiments:
 * p = 3 log(|U|)/|U|.
 *
 * A more complex vector considers different variants for the arm context.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class CLUBERdos<U ,I> extends AbstractCLUB<U,I>
{
    /**
     * Constructor.
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param alpha1 Parameter that manages the importance of the confidence bound for the item selection.
     * @param alpha2 Parameter that manages how difficult is for an edge in the graph to disappear.
     */
    public CLUBERdos(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, double alpha1, double alpha2)
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
    public CLUBERdos(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed, alpha1, alpha2);
    }

    @Override
    protected GraphGenerator<Integer> configureGenerator()
    {
        double p = 3*Math.log(uIndex.numUsers()+0.0)/(uIndex.numUsers()+0.0);
        GraphGenerator<Integer> ggen = new ErdosGenerator<>();
        ggen.configure(false, p, uIndex.getAllUidx().boxed().collect(Collectors.toCollection(HashSet::new)));
        return ggen;
    }
}
