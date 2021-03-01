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

import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.generator.GeneratorBadConfiguredException;
import es.uam.eps.ir.knnbandit.graph.generator.GeneratorNotConfiguredException;
import es.uam.eps.ir.knnbandit.graph.generator.GraphGenerator;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
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
 * Implementation of the COllaborative FIltering BAndits algorithm.
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
public abstract class AbstractCOFIBA<U,I> extends AbstractInteractiveRecommender<U,I>
{
    // Item structures:
    /**
     * A graph structure determining the relations between items.
     */
    private Graph<Integer> itemGraph;
    /**
     * The cluster (connected components) division of the item network. Each cluster is going to be associated to
     * a different user network and user cluster division.
     */
    private Clusters<Integer> itemClusters = new ClustersImpl<>();

    // Individual matrices
    /**
     * For each user in the system, this stores the sum of the ratings u has provided to each item.
     * If a user does not appear, it has not been recommended anything.
     * If, in for a user, an item does not appear, it has never been recommended to u, and therefore, its value is zero.
     */
    private final Int2ObjectMap<Int2DoubleMap> bs;
    /**
     * For each user in the system, this stores the times each item has been recommended to i.
     * If a user does not appear, it has not been recommended anything.
     * If, in for a user, an item does not appear, it has never been recommended to u, and therefore, its value is zero.
     */
    private final Int2ObjectMap<Int2DoubleMap> ms;

    // User cluster structures:
    /**
     * A list of user graphs (one per item cluster).
     */
    private List<Graph<Integer>> userGraphs;
    /**
     * A list of user cluster division (one per item cluster)
     */
    private final List<Clusters<Integer>> userClusters;
    /**
     * A list containing the values of the b vectors for each cluster division.
     * For each cluster, the value of an item is the sum of the ratings the users in the cluster have provided to the item.
     */
    private final List<Map<Integer, Int2DoubleMap>> clustB;
    /**
     * A list containing the values of the M matrix for each cluster division.
     * For each cluster, the value of an item is the number of times the item has been recommended to the users in the cluster
     * When compared to the original COFIBA method, this just represents the diagonal of the M matrix.
     */
    private final List<Map<Integer, Int2DoubleMap>> clustM;
    /**
     * Parameter that manages the importance of the confidence bound for the item selection.
     */
    private final double alpha1;
    /**
     * Parameter that manages how difficult is for an edge in the graph to disappear.
     */
    private final double alpha2;
    /**
     * Number of iterations.
     */
    private int iter;

    /**
     * Constructor.
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param alpha1 Parameter that manages the importance of the confidence bound for the item selection.
     * @param alpha2 Parameter that manages how difficult is for an edge in the user and item graphs to disappear.
     */
    public AbstractCOFIBA(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, ignoreNotRated);
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.bs = new Int2ObjectOpenHashMap<>();
        this.ms = new Int2ObjectOpenHashMap<>();

        this.clustB = new ArrayList<>();
        this.clustM = new ArrayList<>();
        this.userGraphs = new ArrayList<>();
        this.userClusters = new ArrayList<>();
    }

    /**
     * Constructor.
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param ignoreNotRated True if (user, item) pairs without training must be ignored.
     * @param alpha1 Parameter that manages the importance of the confidence bound for the item selection.
     * @param alpha2 Parameter that manages how difficult is for an edge in the user and item graphs to disappear.
     */
    public AbstractCOFIBA(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean ignoreNotRated, int rngSeed, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, ignoreNotRated, rngSeed);
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.bs = new Int2ObjectOpenHashMap<>();
        this.ms = new Int2ObjectOpenHashMap<>();

        this.clustB = new ArrayList<>();
        this.clustM = new ArrayList<>();
        this.userGraphs = new ArrayList<>();
        this.userClusters = new ArrayList<>();
    }

    @Override
    public void init()
    {
        super.init();
        this.bs.clear();
        this.ms.clear();
        this.clustB.clear();
        this.clustM.clear();
        this.userGraphs.clear();
        this.userClusters.clear();

        // Initialize the first item graph:
        this.itemGraph = this.initializeItemGraph();
        // Then, find the clusters for the item graph:
        // Second, we find the corresponding clusters:
        ClusteringAlgorithm<Integer> wcc = new ConnectedComponents<>();
        Clusters<Integer> clusts = wcc.detectClusters(this.itemGraph);

        // For each item cluster, generate a user cluster.
        clusts.getClusters().forEach(cluster ->
        {
            // Initialize the user graph associated to this cluster:
            Graph<Integer> userGraph = this.initializeUserGraph();
            this.userGraphs.add(userGraph);
            // Since there are no ratings, just find the clusters:
            Clusters<Integer> userClusts = wcc.detectClusters(userGraph);
            this.userClusters.add(userClusts);

            Int2ObjectMap<Int2DoubleMap> auxClusterB = new Int2ObjectOpenHashMap<>();
            Int2ObjectMap<Int2DoubleMap> auxClusterM = new Int2ObjectOpenHashMap<>();

            userClusts.getClusters().forEach(userClust ->
            {
                auxClusterB.put(userClust, new Int2DoubleOpenHashMap());
                auxClusterM.put(userClust, new Int2DoubleOpenHashMap());
            });

            this.clustB.add(auxClusterB);
            this.clustM.add(auxClusterM);
        });

        this.iter = 0;
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        this.init();
        values.forEach(t -> this.fastUpdate(t.uidx(),t.iidx(),t.value()));
    }

    /**
     * Initializes the structure of the item graph.
     * @return the item graph.
     */
    private Graph<Integer> initializeItemGraph()
    {
        try
        {
            GraphGenerator<Integer> ggen = this.configureItemGenerator();
            return ggen.generate();
        }
        catch (GeneratorNotConfiguredException | GeneratorBadConfiguredException e)
        {
            return null;
        }
    }

    /**
     * Initializes the structure of the user graph.
     * @return the user graph.
     */
    private Graph<Integer> initializeUserGraph()
    {
        try
        {
            GraphGenerator<Integer> ggen = this.configureUserGenerator();
            return ggen.generate();
        }
        catch (GeneratorNotConfiguredException | GeneratorBadConfiguredException e)
        {
            return null;
        }
    }

    /**
     * Configures the graph generator for the item graph.
     * @return the graph generator.
     */
    protected abstract GraphGenerator<Integer> configureItemGenerator();

    /**
     * Configures the graph generator for the user graphs.
     * @return the graph generator.
     */
    protected abstract GraphGenerator<Integer> configureUserGenerator();

    @Override
    public int next(int uidx, IntList available)
    {
        if(available == null || available.isEmpty())
        {
            return -1;
        }

        double max = Double.NEGATIVE_INFINITY;
        IntList top = new it.unimi.dsi.fastutil.ints.IntArrayList();

        // Now, obtain the rating for each available item
        for(int iidx : available)
        {
            // Identify the item cluster:
            int itemCluster = this.itemClusters.getCluster(iidx);
            // Then, identify the user cluster for uidx and this item:
            int userCluster = this.userClusters.get(itemCluster).getCluster(uidx);

            double clusterB = this.clustB.get(itemCluster).get(userCluster).getOrDefault(iidx, 0.0);
            double clusterM = this.clustM.get(itemCluster).get(userCluster).getOrDefault(iidx, 0.0) + 1.0;

            double score = clusterB / clusterM + alpha1*Math.sqrt(Math.log(this.iter + 1.0)/clusterM);
            if(score > max)
            {
                top.clear();
                top.add(iidx);
                max = score;
            }
            else if(score == max)
            {
                top.add(iidx);
            }
        }

        // Finally, select among the top performing items the item to recommend.
        int topSize = top.size();
        if (top.isEmpty())
        {
            return available.get(rng.nextInt(available.size()));
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
        IntList top = new IntArrayList();
        int num = Math.min(availability.size(), k);
        PriorityQueue<Tuple2id> queue = new PriorityQueue<>(num, Comparator.comparingDouble(x -> x.v2));

        // Then, select the top value
        for (int iidx : availability)
        {
            // Identify the item cluster:
            int itemCluster = this.itemClusters.getCluster(iidx);
            // Then, identify the user cluster for uidx and this item:
            int userCluster = this.userClusters.get(itemCluster).getCluster(uidx);

            double clusterB = this.clustB.get(itemCluster).get(userCluster).getOrDefault(iidx, 0.0);
            double clusterM = this.clustM.get(itemCluster).get(userCluster).getOrDefault(iidx, 0.0) + 1.0;

            double val = clusterB / clusterM + alpha1*Math.sqrt(Math.log(this.iter + 1.0)/clusterM);

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
    public void fastUpdate(int uidx, int iidx, double value)
    {
        double newValue;
        if(!Double.isNaN(value))
            newValue = value;
        else if(!this.ignoreNotRated)
            newValue = Constants.NOTRATEDNOTIGNORED;
        else
            return;

        // First, find the item cluster
        int itemCluster = this.itemClusters.getCluster(iidx);

        // Update the individual values for the user in the cluster.
        if(!this.ms.containsKey(uidx))
        {
            Int2DoubleMap map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.ms.put(uidx, map);
            map = new Int2DoubleOpenHashMap();
            map.defaultReturnValue(0.0);
            this.bs.put(uidx, map);
        }

        ((Int2DoubleOpenHashMap) this.bs.get(uidx)).addTo(iidx, newValue);
        ((Int2DoubleOpenHashMap) this.ms.get(uidx)).addTo(iidx, 1.0);
        double uCB = Math.log(this.iter + 1)/(this.ms.get(uidx).get(iidx) + 1.0);
        double uW = this.bs.get(uidx).get(iidx) / (this.ms.get(uidx).get(iidx) + 1.0);

        // Then, for this particular item cluster, update the user graph:

        Graph<Integer> userGraph = userGraphs.get(itemCluster);
        Clusters<Integer> oldUserClusters = this.userClusters.get(iidx);
        int userCluster = this.userClusters.get(iidx).getCluster(uidx);


        // Check which edges we have to remove from the user graph.
        List<Integer> neighbors = userGraph.getNeighbourNodes(uidx).collect(Collectors.toList());
        boolean removed = neighbors.stream().map(vidx ->
        {
            double vCB;
            double vW;
            if(this.ms.containsKey(vidx))
            {
                double vM = this.ms.get(vidx).getOrDefault(iidx, 0.0) + 1.0;
                vW = this.bs.get(vidx).getOrDefault(iidx, 0.0) / vM;
                vCB = Math.sqrt(Math.log(this.iter+1.0)/vM);
            }
            else
            {
                vW = 0.0;
                vCB = Math.sqrt(Math.log(this.iter+1.0));
            }

            // if this happens, remove an edge
            if(Math.abs(uW-vW) > alpha2*(uCB + vCB))
            {
                userGraph.removeEdge(uidx, vidx);
                return true;
            }
            return false;
        }).reduce(false, (x,y) -> x || y);


        // If at least an edge has been removed, then, update the user clusters:
        if(removed)
        {
            ClusteringAlgorithm<Integer> wcc = new ConnectedComponents<>();
            Clusters<Integer> clusts = wcc.detectClusters(userGraph, oldUserClusters.getElements(userCluster).collect(Collectors.toSet()));

            // If the number
            if(clusts.getNumClusters() > 1)
            {
                // a) split the clusters
                int currentNumClusters = oldUserClusters.getNumClusters();
                oldUserClusters.divideClusters(userCluster, clusts);

                // b) recompute the values for each cluster
                clusts.getClusters().forEach(cl ->
                {
                    int currentCluster = cl == 0 ? userCluster : currentNumClusters + cl - 1;
                    this.initializeCluster(itemCluster, currentCluster);
                });
            }
            else
            {
                ((Int2DoubleOpenHashMap) this.clustB.get(itemCluster).get(userCluster)).addTo(iidx, newValue);
                ((Int2DoubleOpenHashMap) this.clustM.get(itemCluster).get(userCluster)).addTo(iidx, 1.0);
            }
        }
        else // Otherwise, just update the weights for the cluster.
        {
            ((Int2DoubleOpenHashMap) this.clustB.get(itemCluster).get(userCluster)).addTo(iidx, newValue);
            ((Int2DoubleOpenHashMap) this.clustM.get(itemCluster).get(userCluster)).addTo(iidx, 1.0);
        }

        // Check now the item graph:
        // We first check the neighborhood of the user u
        IntSet iNeighs = userGraph.getNeighbourNodes(uidx).collect(Collectors.toCollection(IntOpenHashSet::new));

        // Now:
        List<Integer> itemNeighbors = itemGraph.getNeighbourNodes(iidx).collect(Collectors.toList());
        boolean deletedItemEdge = itemNeighbors.stream().map(jidx ->
        {
            IntSet jNeighs = new IntOpenHashSet();
            double uidxM = this.ms.get(uidx).getOrDefault(jidx, 0.0) + 1.0;
            double uidxW = this.bs.get(uidx).getOrDefault(jidx, 0.0) / uidxM;
            double uidxCB = Math.sqrt(Math.log(this.iter+1.0)/uidxM);

            this.getUidx().filter(vidx -> uidx != vidx).forEach(vidx ->
            {
                double vidxW;
                double vidxCB;
                if(this.ms.containsKey(vidx))
                {
                    double vidxM = this.ms.get(vidx).getOrDefault(jidx, 0.0) + 1.0;
                    vidxW = this.bs.get(vidx).getOrDefault(jidx, 0.0) / vidxM;
                    vidxCB = Math.sqrt(Math.log(this.iter+1.0)/vidxM);
                }
                else
                {
                    vidxW = 0.0;
                    vidxCB = Math.sqrt(Math.log(this.iter+1.0));
                }

                if(Math.abs(uidxW - vidxW) <= alpha2*(uidxCB+vidxCB))
                {
                    jNeighs.add(vidx);
                }
            });

            if(iNeighs.size() != jNeighs.size())
            {
                itemGraph.removeEdge(iidx, jidx);
                return true;
            }
            for(int neigh : jNeighs)
            {
                if(!iNeighs.contains(neigh))
                {
                    itemGraph.removeEdge(iidx, jidx);
                    return true;
                }
            }

            return false;
        }).reduce(false, (x,y) -> x || y);

        // In case this is deleted, we should check the cluster structure of the item graph
        if(deletedItemEdge)
        {
            ClusteringAlgorithm<Integer> wcc = new ConnectedComponents<>();
            Clusters<Integer> clusts = wcc.detectClusters(itemGraph, itemClusters.getElements(itemCluster).collect(Collectors.toSet()));

            if(clusts.getNumClusters() > 1) // Then, we have to divide the clusters accordingly...
            {
                // a) split the clusters
                int currentNumClusters = itemClusters.getNumClusters();
                itemClusters.divideClusters(itemCluster, clusts);

                clusts.getClusters().forEach(cl ->
                {
                    int currentItemClust = cl == 0 ? itemCluster : currentNumClusters + cl - 1;
                    boolean keepUserClusters = itemClusters.containsElement(iidx, currentItemClust);

                    Graph<Integer> newUserGraph;
                    Clusters<Integer> newUserClusters;
                    if(keepUserClusters)
                    {
                        newUserGraph = userGraph;
                        newUserClusters = oldUserClusters;
                    }
                    else
                    {
                        newUserGraph = this.initializeUserGraph();
                        newUserClusters = wcc.detectClusters(newUserGraph);
                    }

                    // Then, update values:
                    this.userGraphs.set(currentItemClust, newUserGraph);
                    this.userClusters.set(currentItemClust, newUserClusters);
                    this.clustB.set(currentItemClust, new Int2ObjectOpenHashMap<>());
                    this.clustM.set(currentItemClust, new Int2ObjectOpenHashMap<>());

                    // Initialize the clusters:
                    newUserClusters.getClusters().forEach(currentUserCluster -> this.initializeCluster(currentItemClust, currentUserCluster));
                });
            }
        }

        this.iter++;
    }

    /**
     * Initializes the weights for a user cluster.
     * @param itemCluster the item cluster identifier.
     * @param userCluster the user cluster identifier.
     */
    private void initializeCluster(int itemCluster, int userCluster)
    {
        Clusters<Integer> clusts = this.userClusters.get(itemCluster);

        Int2DoubleOpenHashMap auxM = new Int2DoubleOpenHashMap();
        auxM.defaultReturnValue(0.0);
        Int2DoubleOpenHashMap auxB = new Int2DoubleOpenHashMap();
        auxB.defaultReturnValue(0.0);

        clusts.getElements(userCluster).forEach(uidx ->
        {
            if(this.ms.containsKey(uidx) && this.bs.containsKey(uidx))
            {
               Int2DoubleMap uidxM = this.ms.get(uidx);
               Int2DoubleMap uidxB = this.bs.get(uidx);

               uidxM.keySet().stream().filter(iidx -> itemClusters.containsElement(iidx, itemCluster)).forEach(iidx ->
                {
                    auxB.addTo(iidx, uidxB.get(iidx));
                    auxM.addTo(iidx, uidxM.get(iidx));
                });
            }
        });

        this.clustM.get(itemCluster).put(userCluster, auxM);
        this.clustB.get(itemCluster).put(userCluster, auxB);
    }
}
