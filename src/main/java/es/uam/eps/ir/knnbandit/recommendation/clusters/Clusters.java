package es.uam.eps.ir.knnbandit.recommendation.clusters;

import java.util.Collection;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Interface for defining cluster partitions.
 * @param <E> the type of the elements
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface Clusters<E>
{
    // Methods related to the elements in the division:
    /**
     * Obtains the total number of elements.
     * @return the total number of elements.
     */
    int getNumElems();

    /**
     * Obtains the whole set of elements.
     * @return an stream containing the elements.
     */
    Stream<E> getElems();

    /**
     * Check whether the cluster contains an element or not.
     * @return true if it contains the element, false otherwise.
     */
    boolean containsElem(E elem);

    // Methods related to the clusters
    /**
     * Obtains the number of clusters
     * @return The number of clusters
     */
    int getNumClusters();
    /**
     * Obtains the different clusters.
     * @return an int stream containing the different clusters.
     */
    IntStream getClusters();

    // Getters related to individual clusters/elements
    /**
     * Obtains the cluster an element belongs to
     * @param elem the element.
     * @return the cluster the element belongs to.
     */
    int getCluster(E elem);

    /**
     * Gets the elements inside a cluster.
     * @param cluster The cluster.
     * @return a stream containing the elements in the cluster if exists, an empty stream if not.
     */
    Stream<E> getElements(int cluster);

    /**
     * Obtains the size of a cluster.
     * @param cluster The cluster whose size we want to obtain.
     * @return the size of the cluster if it exists, 0 if it does not.
     */
    int getClusterSize(int cluster);

    // Editing methods
    /**
     * Adds a new empty cluster to the list.
     */
    void addCluster();

    /**
     * Adds a pair element/cluster
     * @param elem the new element of the cluster. It must not be already in the object.
     * @param cluster the associated cluster. The community has to previously exist.
     * @return true if everything goes OK, false otherwise.
     */
    boolean add(E elem, int cluster);

    /**
     * Adds a list of elements to a cluster.
     * @param elems the list of elements.
     * @param cluster the associated cluster. The cluster has to previously exist.
     * @return true if everything goes OK, false otherwise;
     */
    int add(Collection<E> elems, int cluster);

    /**
     * Adds a list of elements to a cluster.
     * @param elems the list of elements.
     * @param cluster the associated cluster. The cluster has to previously exist.
     * @return true if everything goes OK, false otherwise;
     */
    int add(Stream<E> elems, int cluster);

    /**
     * Divides the clusters in several
     * @param cluster the identifier of the cluster.
     * @param division the cluster division.
     * @return true if everything went ok, false otherwise.
     */
     boolean divideClusters(int cluster, Clusters<E> division);
}
