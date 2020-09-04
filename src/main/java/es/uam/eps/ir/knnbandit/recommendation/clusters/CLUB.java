package es.uam.eps.ir.knnbandit.recommendation.clusters;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.userknowledge.fast.SimpleFastUserKnowledgePreferenceData;
import es.uam.eps.ir.knnbandit.graph.complementary.ComplementaryGraph;
import es.uam.eps.ir.knnbandit.graph.complementary.UndirectedUnweightedComplementaryGraph;
import es.uam.eps.ir.knnbandit.graph.fast.FastGraph;
import es.uam.eps.ir.knnbandit.graph.fast.FastUndirectedUnweightedGraph;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Implementation of the CLUster of Bandits algorithm (Gentile et al. 2014)
 * We assume that context vectors for the different items are versors (unit vectors)
 * and that we recommend all the possible candidate items.
 *
 * A more complex vector considers different variants for the arm context.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public class CLUB<U,I> extends InteractiveRecommender<U,I>
{
    /**
     * The graph expressing how related the user vectors are.
     */
    private FastGraph<U> graph = new FastUndirectedUnweightedGraph<>();
    /**
     * The different user clusters (connected components of the network)
     */
    private ClustersImpl<U> clusters = new ClustersImpl<>();

    // Individual matrices
    /**
     * The estimation for the user vector.
     */
    private final Map<U, DoubleMatrix1D> ws;

    // Cluster matrices
    /**
     * Vector containing the aggregated information of each cluster
     */
    private final Map<Integer, DoubleMatrix1D> clustW;
    /**
     * Vector containing information about the ratings of the different users to the different items.
     */
    private final Map<Integer, DoubleMatrix1D> clustB;
    /**
     * Vector containing information about the number of times each item has been recommended to users on each cluster.
     * This is the diagonal of the matrix, since the rest of elements is equal to zero under the versor assumption.
     */
    private final Map<Integer, DoubleMatrix1D> clustM;

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

    public CLUB(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreNotRated, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, prefData, ignoreNotRated);
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.ws = new HashMap<>();
        clustW = new HashMap<>();
        clustM = new HashMap<>();
        clustB = new HashMap<>();
    }

    public CLUB(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData, boolean ignoreNotRated, KnowledgeDataUse dataUse, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, prefData, knowledgeData, ignoreNotRated, dataUse);
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.ws = new HashMap<>();
        clustW = new HashMap<>();
        clustM = new HashMap<>();
        clustB = new HashMap<>();
    }

    public CLUB(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreNotRated, boolean notReciprocal, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, prefData, ignoreNotRated, notReciprocal);
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.ws = new HashMap<>();
        clustW = new HashMap<>();
        clustM = new HashMap<>();
        clustB = new HashMap<>();
    }

    @Override
    protected void initializeMethod()
    {
        // First, we initialize the network as an empty graph
        this.graph = new FastUndirectedUnweightedGraph<>();

        // Initialize the different maps.
        ws.clear();
        times.clear();

        // Initialize the values of M, b and times for each user
        this.getUsers().forEach(user ->
        {
            graph.addNode(user);
            ws.put(user, new SparseDoubleMatrix1D(this.numItems()));
            times.put(this.uIndex.user2uidx(user), 0);
        });

        System.out.println("Basic values initialized");

        // Update those values depending on the training data.
        this.trainData.getUsersWithPreferences().forEach(u ->
        {
            int uidx = this.uIndex.user2uidx(u);
            this.trainData.getUidxPreferences(uidx).forEach(i -> ws.get(u).setQuick(i.v1, i.v2/2));
            this.times.put(uidx, this.trainData.numItems(uidx));
        });

        // Training values.
        System.out.println("Training values considered");
        Int2DoubleMap CBs = new Int2DoubleOpenHashMap();
        this.getUidx().forEach(uidx -> CBs.put(uidx, Math.sqrt((1.0+Math.log(times.get(uidx)+1)/(1+times.get(uidx))))));

        IntSet visited = new IntOpenHashSet();
        this.trainData.getUidxWithPreferences().forEach(uidx ->
        {
            U u = this.uIndex.uidx2user(uidx);
            double uCB = CBs.get(uidx);
            visited.add(uidx);
            DoubleMatrix1D uW = ws.get(u);
            double uVal = this.trainData.getUidxPreferences(uidx).mapToDouble(i -> uW.getQuick(i.v1)*uW.getQuick(i.v1)).sum();

            this.getUidx().filter(vidx -> !visited.contains(vidx)).forEach(vidx ->
            {
                U v = this.uIndex.uidx2user(vidx);
                DoubleMatrix1D vW = ws.get(v);
                double dist = uVal;
                dist += this.trainData.getUidxPreferences(vidx).mapToDouble(i ->
                {
                    double vWi= vW.getQuick(i.v1);
                    return vWi*vWi - 2*vWi*uW.getQuick(i.v1);
                }).sum();

                dist = Math.sqrt(dist);
                double vCB = CBs.get(vidx);

                if(dist > alpha2*(uCB + vCB))
                {
                    graph.addEdge(u, v);
                }
            });
        });

        System.out.println("Graph found");

        // And finally, reobtain the clusters and the number of iterations.
        ComplementaryGraph<U> compl = new UndirectedUnweightedComplementaryGraph<>(graph);
        ConnectedComponents<U> conn = new ConnectedComponents<>();
        this.clusters = conn.detectClusters(compl);
        this.iter = trainData.numPreferences();

        System.out.println("Clusters found");

        // Compute the cluster values:
        this.clusters.getClusters().forEach(cluster ->
        {
            DoubleMatrix1D auxM = new SparseDoubleMatrix1D(this.numItems());
            DoubleMatrix1D auxB = new SparseDoubleMatrix1D(this.numItems());
            DoubleMatrix1D auxW = new SparseDoubleMatrix1D(this.numItems());

            clusters.getElements(cluster).forEach(u ->
            {
                IntSet visitedItems = new IntOpenHashSet();
                int uidx = this.uIndex.user2uidx(u);
                this.trainData.getUidxPreferences(uidx).forEach(i ->
                {
                    auxB.setQuick(i.v1, auxB.getQuick(i.v1) + i.v2);
                    auxM.setQuick(i.v1, auxM.getQuick(i.v1) + 1);
                    visitedItems.add(i.v1);
                });

                visitedItems.forEach(iidx -> auxW.setQuick(iidx, auxB.getQuick(iidx)/(1+auxM.getQuick(iidx))));
            });

            clustM.put(cluster, auxM);
            clustB.put(cluster, auxB);
            clustW.put(cluster, auxW);
        });

        System.out.println("Values for the clusters found");
    }

    @Override
    public int next(int uidx)
    {
        // We first check if there is a recommendable item.
        IntList list = this.availability.get(uidx);
        if (list == null || list.isEmpty())
        {
            return -1;
        }

        U u = this.uIndex.uidx2user(uidx);

        // First, we do have to get the cluster of user u
        int cluster = clusters.getCluster(u);
        DoubleMatrix1D auxM = clustM.get(cluster);
        DoubleMatrix1D auxW = clustW.get(cluster);

        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();

        // Then, select the top value
        for (int iidx : availability.get(uidx))
        {
            double val = auxW.getQuick(iidx) + alpha1 * Math.sqrt(1.0/(auxM.getQuick(iidx)+1.0) * Math.log(this.iter + 1));
            if (top.isEmpty() || val > max)
            {
                top = new IntArrayList();
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
            return list.get(rng.nextInt(list.size()));
        }
        else if (topSize == 1)
        {
            return top.get(0);
        }
        return top.get(rng.nextInt(topSize));
    }

    @Override
    public void updateMethod(int uidx, int iidx, double value)
    {
        U u = this.uIndex.uidx2user(uidx);

        ++this.iter;

        // Step 1: find the cluster
        DoubleMatrix1D uW = this.ws.get(u);
        uW.setQuick(iidx, value/2.0);
        int cluster = this.clusters.getCluster(u);
        Set<U> elements = this.clusters.getElements(cluster).collect(Collectors.toCollection(HashSet::new));
        double cbU = Math.sqrt((1.0+Math.log(times.get(uidx)+1)/(1+times.get(uidx))));

        double uVal = this.trainData.getUidxPreferences(uidx).mapToDouble(i -> uW.getQuick(i.v1)*uW.getQuick(i.v1)).sum();
        boolean added = false;

        // Check with the elements within the cluster:
        for(U v : elements)
        {
            int vidx = this.uIndex.user2uidx(v);
            if(u == v) continue;

            DoubleMatrix1D vW = this.ws.get(v);
            double dist = uVal;
            dist += this.trainData.getUidxPreferences(vidx).mapToDouble(i ->
            {
                double vWi= vW.getQuick(i.v1);
                return vWi*vWi - 2*vWi*uW.getQuick(i.v1);
            }).sum();

            dist = Math.sqrt(dist);
            double cbV = Math.sqrt((1.0+Math.log(times.get(vidx)+1)/(1+times.get(vidx))));

            if(dist > alpha2*(cbU + cbV))
            {
                graph.addEdge(u,v);
                added = true;
            }
        }

        if(added)
        {
            ComplementaryGraph<U> compl = new UndirectedUnweightedComplementaryGraph<>(graph);
            ConnectedComponents<U> conn = new ConnectedComponents<>();
            Clusters<U> clusts = conn.detectClusters(compl, elements);

            if(clusts.getNumClusters() > 1)
            {
                // a) split the clusters
                int currentNumClusters = this.clusters.getNumClusters();
                this.clusters.divideClusters(cluster, clusts);

                // b) recompute the values for each cluster:
                clusts.getClusters().forEach(cl ->
                {
                    int currentClust = cl == 0 ? cluster : currentNumClusters + cl - 1;

                    DoubleMatrix1D auxM = new SparseDoubleMatrix1D(this.numItems());
                    DoubleMatrix1D auxB = new SparseDoubleMatrix1D(this.numItems());
                    DoubleMatrix1D auxW = new SparseDoubleMatrix1D(this.numItems());
                    clusts.getElements(cl).forEach(auxU ->
                    {
                        IntSet visitedItems = new IntOpenHashSet();
                        int auxUidx = this.uIndex.user2uidx(auxU);
                        this.trainData.getUidxPreferences(auxUidx).forEach(i ->
                        {
                            auxB.setQuick(i.v1, auxB.getQuick(i.v1) + i.v2);
                            auxM.setQuick(i.v1, auxM.getQuick(i.v1) + 1);
                            visitedItems.add(i.v1);
                        });

                        visitedItems.forEach(auxIidx -> auxW.setQuick(auxIidx, auxB.getQuick(auxIidx)/(1+auxM.getQuick(auxIidx))));
                    });

                    clustM.put(currentClust, auxM);
                    clustB.put(currentClust, auxB);
                    clustW.put(currentClust, auxW);
                });
            }
            else  // just update the cluster values for this particular case.
            {
                double mValue = this.clustM.get(cluster).getQuick(iidx)+1.0;
                double bValue = this.clustB.get(cluster).getQuick(iidx)+value;

                this.clustM.get(cluster).setQuick(iidx, mValue);
                this.clustB.get(cluster).setQuick(iidx, bValue);
                this.clustW.get(cluster).setQuick(iidx, bValue/(mValue+1.0));
            }
        }

        times.put(uidx, times.get(uidx)+1);
    }
}
