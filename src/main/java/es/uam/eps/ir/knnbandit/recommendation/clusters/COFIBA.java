package es.uam.eps.ir.knnbandit.recommendation.clusters;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.userknowledge.fast.SimpleFastUserKnowledgePreferenceData;
import es.uam.eps.ir.knnbandit.graph.Graph;
import es.uam.eps.ir.knnbandit.graph.fast.FastGraph;
import es.uam.eps.ir.knnbandit.graph.generator.ErdosGenerator;
import es.uam.eps.ir.knnbandit.graph.generator.GeneratorBadConfiguredException;
import es.uam.eps.ir.knnbandit.graph.generator.GeneratorNotConfiguredException;
import es.uam.eps.ir.knnbandit.graph.generator.GraphGenerator;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import it.unimi.dsi.fastutil.ints.*;
import org.apache.commons.math3.ml.clustering.Cluster;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Implementation of the COllaborative FIltering BAndits algorithm (Li et al. 2016 SIGIR).
 * As it is done in the original paper, we assume that context vectors for the different
 * items are versors (unitary vectors) for each item.
 *
 * A more complex vector considers different variants for the arm context.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 */
public class COFIBA<U,I> extends InteractiveRecommender<U,I>
{
    private final double alpha1;
    private final double alpha2;

    /**
     * The graph expressing relations between items.
     */
    private Graph<Integer> itemGraph;
    /**
     * The different user clusters (connected components of the network)
     */
    private Clusters<Integer> itemClusters = new ClustersImpl<>();
    /**
     * A list of user graphs (one per item cluster).
     */
    private List<Graph<Integer>> userGraphs;
    private final List<Clusters<Integer>> userClusters;

    // Individual matrices
    /**
     * The estimation for the user vector.
     */
    private final Int2ObjectMap<DoubleMatrix1D> ws;

    /**
     * Vector containing the aggregated information of each cluster
     */
    private final List<Map<Integer, DoubleMatrix1D>> clustW;
    /**
     * Vector containing information about the ratings of the different users to the different items.
     */
    private final List<Map<Integer, DoubleMatrix1D>> clustB;
    /**
     * Vector containing information about the number of times each item has been recommended to users on each cluster.
     * This is the diagonal of the matrix, since the rest of elements is equal to zero under the versor assumption.
     */
    private final List<Map<Integer, DoubleMatrix1D>> clustM;
    /**
     * The number of times each user has been run
     */
    private final Int2IntMap times = new Int2IntOpenHashMap();

    public COFIBA(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreNotRated, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, prefData, ignoreNotRated);
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.ws = new Int2ObjectOpenHashMap<>();
        this.clustW = new ArrayList<>();
        this.clustB = new ArrayList<>();
        this.clustM = new ArrayList<>();
        this.userGraphs = new ArrayList<>();
        this.userClusters = new ArrayList<>();
    }

    public COFIBA(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData, boolean ignoreNotRated, KnowledgeDataUse dataUse, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, prefData, knowledgeData, ignoreNotRated, dataUse);
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.ws = new Int2ObjectOpenHashMap<>();
        this.clustW = new ArrayList<>();
        this.clustB = new ArrayList<>();
        this.clustM = new ArrayList<>();
        this.userGraphs = new ArrayList<>();
        this.userClusters = new ArrayList<>();
    }

    public COFIBA(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, boolean ignoreNotRated, boolean notReciprocal, double alpha1, double alpha2)
    {
        super(uIndex, iIndex, prefData, ignoreNotRated, notReciprocal);
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.ws = new Int2ObjectOpenHashMap<>();
        this.clustW = new ArrayList<>();
        this.clustB = new ArrayList<>();
        this.clustM = new ArrayList<>();
        this.userGraphs = new ArrayList<>();
        this.userClusters = new ArrayList<>();
    }

    @Override
    protected void initializeMethod()
    {
        ws.clear();
        times.clear();
        // First, initialize times and the user values:
        this.uIndex.getAllUidx().forEach(uidx ->
         {
             this.ws.put(uidx, new SparseDoubleMatrix1D(this.numItems()));
             this.times.put(uidx, 0);
         });

        // First, configure the item graph.
        double p = 3 * Math.log(iIndex.numItems() + 0.0) / (iIndex.numItems() + 0.0);
        GraphGenerator<Integer> ggen = new ErdosGenerator<>();
        ggen.configure(false, p, iIndex.getAllIidx().boxed().collect(Collectors.toCollection(HashSet::new)));

        try
        {
            this.itemGraph = ggen.generate();
        }
        catch (GeneratorNotConfiguredException | GeneratorBadConfiguredException e)
        {
            this.itemGraph = null;
            return;
        }

        // Find the clusters for the item graph.
        ClusteringAlgorithm<Integer> itemCluster = new ConnectedComponents<>();
        this.itemClusters = itemCluster.detectClusters(itemGraph);

        // Create the user graphs for each cluster in the item graph.
        this.userGraphs.clear();
        this.userClusters.clear();
        this.clustM.clear();
        this.clustB.clear();
        this.clustW.clear();

        try
        {
            for(int i = 0; i < itemClusters.getNumClusters(); ++i)
            {
                this.initializeUserGraph();
            }
        }
        catch(GeneratorNotConfiguredException | GeneratorBadConfiguredException e)
        {
            this.userGraphs.clear();
        }
    }


    /**
     * Initializes a user graph-
     * @throws GeneratorNotConfiguredException if the graph generator is not configured.
     * @throws GeneratorBadConfiguredException if the graph generator parameters are not valid.
     */
    private void initializeUserGraph() throws GeneratorNotConfiguredException, GeneratorBadConfiguredException
    {
        // First, we find the graph.
        double p = 3 * Math.log(uIndex.numUsers() + 0.0) / (uIndex.numUsers() + 0.0);
        GraphGenerator<Integer> ggen = new ErdosGenerator<>();
        ggen.configure(false, p, uIndex.getAllUidx().boxed().collect(Collectors.toCollection(HashSet::new)));
        Graph<Integer> graph = ggen.generate();

        // Second, we find the corresponding clusters:
        ClusteringAlgorithm<Integer> wcc = new ConnectedComponents<>();
        Clusters<Integer> clusts = wcc.detectClusters(graph);

        // Then, find the values
        Int2ObjectMap<DoubleMatrix1D> auxMs = new Int2ObjectOpenHashMap<>();
        Int2ObjectMap<DoubleMatrix1D> auxBs = new Int2ObjectOpenHashMap<>();
        Int2ObjectMap<DoubleMatrix1D> auxWs = new Int2ObjectOpenHashMap<>();

        clusts.getClusters().forEach(cluster ->
        {
            DoubleMatrix1D auxM = new SparseDoubleMatrix1D(this.numItems());
            DoubleMatrix1D auxB = new SparseDoubleMatrix1D(this.numItems());
            DoubleMatrix1D auxW = new SparseDoubleMatrix1D(this.numItems());

            clusts.getElements(cluster).forEach(uidx ->
            {
                IntSet visitedItems = new IntOpenHashSet();
                this.trainData.getUidxPreferences(uidx).forEach(i ->
                {
                    auxB.setQuick(i.v1, auxB.getQuick(i.v1) + i.v2);
                    auxM.setQuick(i.v1, auxM.getQuick(i.v1) + 1);
                    visitedItems.add(i.v1);
                });

                visitedItems.forEach(iidx -> auxW.setQuick(iidx, auxB.getQuick(iidx)/(1+auxM.getQuick(iidx))));
            });

            auxMs.put(cluster, auxM);
            auxBs.put(cluster, auxB);
            auxWs.put(cluster, auxW);
        });

        this.userGraphs.add(graph);
        this.userClusters.add(clusts);
        this.clustM.add(auxMs);
        this.clustB.add(auxBs);
        this.clustW.add(auxWs);
    }

    @Override
    public int next(int uidx)
    {
        return 0;
    }

    @Override
    public void updateMethod(int uidx, int iidx, double value)
    {

    }
}
