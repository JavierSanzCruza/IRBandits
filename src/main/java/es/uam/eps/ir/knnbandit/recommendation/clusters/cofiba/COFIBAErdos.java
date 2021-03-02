/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.clusters.cofiba;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.generator.ErdosGenerator;
import es.uam.eps.ir.knnbandit.graph.generator.GeneratorBadConfiguredException;
import es.uam.eps.ir.knnbandit.graph.generator.GeneratorNotConfiguredException;
import es.uam.eps.ir.knnbandit.graph.generator.GraphGenerator;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.clusters.ClusteringAlgorithm;
import es.uam.eps.ir.knnbandit.recommendation.clusters.Clusters;
import es.uam.eps.ir.knnbandit.recommendation.clusters.ClustersImpl;
import es.uam.eps.ir.knnbandit.recommendation.clusters.ConnectedComponents;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import it.unimi.dsi.fastutil.ints.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Implementation of the COllaborative FIltering BAndits algorithm (Li et al. 2016 SIGIR).
 * As it is done in the original paper, we assume that context vectors for the different
 * items are versors (unitary vectors) for each item.
 *
 * A more complex vector considers different variants for the arm context.
 *
 * <p>
 *     <b>Reference: </b> S. Li, A. Karatzoglou, C. Gentile. Collaborative Filtering Bandits. 39th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2016), Pisa, Tuscany, Italy, pp. 539-548 (2016).
 * </p>
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class COFIBAErdos<U,I> extends AbstractCOFIBA<U,I>
{

    /**
     * Constructor.
     *
     * @param uIndex         User index.
     * @param iIndex         Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param alpha1         Parameter that manages the importance of the confidence bound for the item selection.
     * @param alpha2         Parameter that manages how difficult is for an edge in the user and item graphs to disappear.
     */
    public COFIBAErdos(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, ignoreNotRated, alpha1, alpha2);
    }

    /**
     * Constructor.
     *
     * @param uIndex         User index.
     * @param iIndex         Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param rngSeed        Random number generator seed.
     * @param alpha1         Parameter that manages the importance of the confidence bound for the item selection.
     * @param alpha2         Parameter that manages how difficult is for an edge in the user and item graphs to disappear.
     */
    public COFIBAErdos(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed, alpha1, alpha2);
    }

    @Override
    protected GraphGenerator<Integer> configureItemGenerator()
    {
        double p = 3*Math.log(uIndex.numUsers()+0.0)/(uIndex.numUsers()+0.0);
        GraphGenerator<Integer> ggen = new ErdosGenerator<>();
        ggen.configure(false, p, iIndex.getAllIidx().boxed().collect(Collectors.toCollection(HashSet::new)));
        return ggen;
    }

    @Override
    protected GraphGenerator<Integer> configureUserGenerator()
    {
        double p = 3*Math.log(uIndex.numUsers()+0.0)/(uIndex.numUsers()+0.0);
        GraphGenerator<Integer> ggen = new ErdosGenerator<>();
        ggen.configure(false, p, uIndex.getAllUidx().boxed().collect(Collectors.toCollection(HashSet::new)));
        return ggen;
    }


}
