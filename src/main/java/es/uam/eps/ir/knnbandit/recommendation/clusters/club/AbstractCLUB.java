/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.clusters.club;

import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.graph.fast.FastGraph;
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
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Implementation of the CLUstering of Bandits algorithm
 * We assume that context vectors for the different items are versors (unit vectors)
 * and that we recommend all the possible candidate items.
 *
 * A more complex vector considers different variants for the arm context.
 *
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
public abstract class AbstractCLUB<U ,I> extends InteractiveRecommender<U,I>
{
    /**
     * The graph expressing how related the user vectors are.
     */
    private FastGraph<Integer> graph;
    /**
     * The different user clusters (connected components of the network)
     */
    private Clusters<Integer> clusters = new ClustersImpl<>();

    // Individual matrices
    /**
     * The estimation for the user vector.
     */
    private final Map<Integer, Int2DoubleMap> bs;

    /**
     * The number of times each item has been recommended to the different users.
     */
    private final Map<Integer, Int2DoubleMap> ms;

    // Cluster matrices
    /**
     * Vector containing information about the ratings of the different users to the different items.
     */
    private final Map<Integer, Int2DoubleMap> clustB;
    /**
     * Vector containing information about the number of times each item has been recommended to users on each cluster.
     * This is the diagonal of the matrix, since the rest of elements is equal to zero under the versor assumption.
     */
    private final Map<Integer, Int2DoubleMap> clustM;

    /**
     * The number of times each user has been run
     */
    private final Int2IntMap times = new Int2IntOpenHashMap();

    /**
     * The current number of iterations.
     */
    private int iter = 0;

    /**
     * Parameter that manages the importance of the confidence bound for the item selection.
     */
    private final double alpha1;
    /**
     * Parameter that manages how difficult is for an edge in the graph to disappear.
     */
    private final double alpha2;

    /**
     * Constructor.
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param alpha1 Parameter that manages the importance of the confidence bound for the item selection.
     * @param alpha2 Parameter that manages how difficult is for an edge in the graph to disappear.
     */
    public AbstractCLUB(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, ignoreNotRated);
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.bs = new HashMap<>();
        this.ms = new HashMap<>();
        clustM = new HashMap<>();
        clustB = new HashMap<>();
    }

    /**
     * Constructor.
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param rngSeed Random number generator seed.
     * @param alpha1 Parameter that manages the importance of the confidence bound for the item selection.
     * @param alpha2 Parameter that manages how difficult is for an edge in the graph to disappear.
     */
    public AbstractCLUB(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed);
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.bs = new HashMap<>();
        this.ms = new HashMap<>();
        clustM = new HashMap<>();
        clustB = new HashMap<>();
    }

    /**
     * Configures a graph generator for the user graph.
     * @return the configured graph generator.
     */
    protected abstract GraphGenerator<Integer> configureGenerator();

    @Override
    public void init()
    {
        super.init();
        GraphGenerator<Integer> ggen = configureGenerator();

        try
        {
            this.graph = (FastGraph<Integer>) ggen.generate();
        }
        catch (GeneratorNotConfiguredException | GeneratorBadConfiguredException e) // An (impossible) error occurred.
        {
            this.graph = null;
            return;
        }

        // Initialize the different maps.
        bs.clear();
        ms.clear();
        times.clear();

        // Initialize the values of M, b and times for each user
        this.getUidx().forEach(uidx -> times.put(uidx, 0));

        // Initialize the set of clusters
        ClusteringAlgorithm<Integer> wcc = new ConnectedComponents<>();
        this.clusters = wcc.detectClusters(graph);
        this.clusters.getClusters().forEach(cl ->
        {
            Int2DoubleOpenHashMap map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.clustB.put(cl, map);
            map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.clustM.put(cl, new Int2DoubleOpenHashMap());
        });
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();
        values.forEach(t -> this.update(t.uidx(), t.iidx(), t.value()));
    }

    @Override
    public int next(int uidx, IntList availability)
    {
        // We first check if there is a recommendable item.
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }

        // First, we do have to get the cluster of user u
        int cluster = clusters.getCluster(uidx);

        Int2DoubleMap auxM = clustM.get(cluster);
        Int2DoubleMap auxB = clustB.get(cluster);

        double max = Double.NEGATIVE_INFINITY;
        IntList top = new it.unimi.dsi.fastutil.ints.IntArrayList();

        // Then, select the top value
        for (int iidx : availability)
        {
            double bVal = auxB.getOrDefault(iidx, 0.0);
            double mVal = auxM.getOrDefault(iidx, 0.0);

            double val = bVal/(mVal + 1.0) + alpha1*Math.sqrt(1.0/(mVal+1.0) * Math.log(this.iter + 1.0));
            if (top.isEmpty() || val > max)
            {
                top.clear();
                max = val;
                top.add(iidx);
            }
            else if (val == max)
            {
                top.add(iidx);
            }
        }

        int topSize = top.size();
        if (top.isEmpty())
        {
            return availability.get(rng.nextInt(availability.size()));
        }
        else if (topSize == 1)
        {
            return top.get(0);
        }
        return top.get(rng.nextInt(topSize));
    }

    @Override
    public IntList next(int uidx, IntList availability, int k)
    {
        // We first check if there is a recommendable item.
        if (availability == null || availability.isEmpty())
        {
            return new IntArrayList();
        }

        // First, we do have to get the cluster of user u
        int cluster = clusters.getCluster(uidx);

        Int2DoubleMap auxM = clustM.get(cluster);
        Int2DoubleMap auxB = clustB.get(cluster);

        IntList top = new IntArrayList();
        int num = Math.min(availability.size(), k);
        PriorityQueue<Tuple2id> queue = new PriorityQueue<>(num, Comparator.comparingDouble(x -> x.v2));

        // Then, select the top value
        for (int iidx : availability)
        {
            double bVal = auxB.getOrDefault(iidx, 0.0);
            double mVal = auxM.getOrDefault(iidx, 0.0);

            double val = bVal/(mVal + 1.0) + alpha1*Math.sqrt(1.0/(mVal+1.0) * Math.log(this.iter + 1.0));

            if(queue.size() < num)
            {
                queue.add(new Tuple2id(iidx, val));
            }
            else
            {
                Tuple2id newTuple = new Tuple2id(iidx, val);
                if(queue.comparator().compare(queue.peek(), newTuple) < 0)
                {
                    queue.poll();
                    queue.add(newTuple);
                }
            }
        }

        while(!queue.isEmpty())
        {
            top.add(0, queue.poll().v1);
        }

        return top;
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        double newValue;
        if(!Double.isNaN(value))
            newValue = value;
        else if(!this.ignoreNotRated)
            newValue = Constants.NOTRATEDNOTIGNORED;
        else
            return;

        ++this.iter;

        // Step 1: find the cluster
        int cluster = this.clusters.getCluster(uidx);
        double uCB = Math.sqrt((1.0+Math.log(times.get(uidx)+2.0))/(2.0+times.get(uidx)));

        // We obtain the square of the module of uidx
        if(!this.ms.containsKey(uidx))
        {
            Int2DoubleOpenHashMap map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.ms.put(uidx, map);
            map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.bs.put(uidx, new Int2DoubleOpenHashMap());
        }

        Int2DoubleMap uM = this.ms.get(uidx);
        Int2DoubleMap uB = this.bs.get(uidx);

        if(!this.ms.containsKey(uidx))
        {
            this.ms.put(uidx, uM);
        }
        if(!this.bs.containsKey(uidx))
        {
            this.bs.put(uidx, uB);
        }

        ((Int2DoubleOpenHashMap) uB).addTo(iidx, newValue);
        ((Int2DoubleOpenHashMap) uM).addTo(iidx, 1.0);

        // Update the individual values:
        double uVal = uM.keySet().stream().mapToDouble(jidx -> Math.pow(uB.get(jidx) / (uM.get(jidx) + 1.0), 2.0)).sum();

        List<Integer> list = graph.getNeighbourNodes(uidx).collect(Collectors.toList());
        boolean added = list.stream().map(vidx ->
        {
            double dist = uVal;

            Int2DoubleMap vM = this.ms.get(uidx);
            Int2DoubleMap vB = this.bs.get(uidx);
            double vCB = Math.sqrt((1.0+Math.log(times.get(vidx)+1.0))/(1.0+times.get(vidx)));

            if(vM != null && vB != null)
            {
                dist += vM.keySet().stream().mapToDouble(jidx ->
                {
                    double aux = vB.get(jidx) / (vM.get(jidx)+1.0);
                    return aux*(aux - uB.getOrDefault(jidx, 0.0)/(uM.getOrDefault(jidx,0.0)+1.0));
                }).sum();
            }

            dist = Math.sqrt(dist);
            if(dist > alpha2*(uCB + vCB))
            {
                graph.removeEdge(uidx, vidx);
                return true;
            }
            return false;
        }).reduce(false, (x,y) -> x || y);

        if(added) // If an edge has been removed, then, check whether the connected components of the graph have changed.
        {
            ConnectedComponents<Integer> conn = new ConnectedComponents<>();
            Clusters<Integer> clusts = conn.detectClusters(graph, clusters.getElements(cluster).collect(Collectors.toSet()));

            if(clusts.getNumClusters() > 1)
            {
                // a) split the clusters
                int currentNumClusters = this.clusters.getNumClusters();
                this.clusters.divideClusters(cluster, clusts);

                // b) recompute the values for each cluster:
                clusts.getClusters().forEach(cl ->
                {
                    int currentClust = cl == 0 ? cluster : currentNumClusters + cl - 1;

                    Int2DoubleOpenHashMap auxM = new Int2DoubleOpenHashMap();
                    auxM.defaultReturnValue(0.0);
                    Int2DoubleOpenHashMap auxB = new Int2DoubleOpenHashMap();
                    auxB.defaultReturnValue(0.0);

                    clusts.getElements(cl).forEach(auxUidx ->
                    {
                        if(this.ms.containsKey(auxUidx)  && this.bs.containsKey(auxUidx))
                        {
                            Int2DoubleMap auxUM = this.ms.get(auxUidx);
                            Int2DoubleMap auxUB = this.bs.get(auxUidx);

                            auxUM.keySet().forEach(jidx ->
                            {
                                auxM.addTo(jidx, auxUM.get(jidx));
                                auxB.addTo(jidx, auxUB.get(jidx));
                            });
                        }
                    });

                    clustM.put(currentClust, auxM);
                    clustB.put(currentClust, auxB);
             });
            }
            else  // just update the cluster values for this particular case.
            {
                ((Int2DoubleOpenHashMap) this.clustM.get(cluster)).addTo(iidx, 1.0);
                ((Int2DoubleOpenHashMap) this.clustB.get(cluster)).addTo(iidx, newValue);
            }
        }
        else // If no edge is removed, then, just update the cluster value...
        {
            ((Int2DoubleOpenHashMap) this.clustM.get(cluster)).addTo(iidx, 1.0);
            ((Int2DoubleOpenHashMap) this.clustB.get(cluster)).addTo(iidx, newValue);
        }

        times.put(uidx, times.get(uidx)+1);
    }
}
