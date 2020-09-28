/* 
 * Copyright (C) 2018 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es
 * 
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.clusters;

import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Class for storing clusters.
 * @author Javier Sanz-Cruzado Puig
 * @param <E> type of the elements.
 */
public class ClustersImpl<E> implements Clusters<E>
{
    /**
     * For each element, indicates the cluster the element belongs to.
     */
    private final Object2IntMap<E> elemCluster;
    /**
     * Indicates the elements of each cluster.
     */
    private final List<List<E>> clusterElems;

    /**
     * Constructor.
     */
    public ClustersImpl()
    {
        elemCluster = new Object2IntOpenHashMap<>();
        clusterElems = new ArrayList<>();
        elemCluster.defaultReturnValue(-1);
    }

    /**
     * Constructor.
     * @param clusters the element partition.
     */
    public ClustersImpl(Collection<Collection<E>> clusters)
    {
        this();

        int i = 0;
        for(Collection<E> cluster : clusters)
        {
            List<E> clusterEl = new ArrayList<>();
            for(E elem : cluster)
            {
                elemCluster.put(elem, i);
                clusterEl.add(elem);
            }

            ++i;
            clusterElems.add(clusterEl);
        }
    }

    @Override
    public int getNumClusters()
    {
        return this.clusterElems.size();
    }

    @Override
    public int getNumElems()
    {
        return this.elemCluster.size();
    }

    @Override
    public int getNumElems(int cluster)
    {
        if(cluster > 0 || cluster < this.getNumClusters())
            return clusterElems.get(cluster).size();
        return -1;
    }

    @Override
    public Stream<E> getElems()
    {
        return this.elemCluster.keySet().stream();
    }

    @Override
    public boolean containsElem(E elem)
    {
        return this.elemCluster.containsKey(elem);
    }

    @Override
    public IntStream getClusters()
    {
        return IntStream.range(0, this.getNumClusters());
    }
    
    @Override
    public int getCluster(E elem)
    {
        return elemCluster.getInt(elem);
    }

    @Override
    public Stream<E> getElements(int cluster)
    {
        if(cluster >= 0 && cluster < this.getNumClusters())
        {
            return clusterElems.get(cluster).stream();
        }
        else
        {
            return Stream.empty();
        }
    }
    
    @Override
    public void addCluster()
    {
        this.clusterElems.add(new ArrayList<>());
    }
    
    @Override
    public boolean add(E elem, int cluster)
    {
        if(cluster >= 0 && cluster < this.getNumClusters() && !elemCluster.containsKey(elem))
        {
            this.clusterElems.get(cluster).add(elem);
            this.elemCluster.put(elem, cluster);
        }
        return false;
    }

    private boolean update(E elem, int cluster)
    {
        if(cluster >= 0 && cluster < this.getNumClusters() && elemCluster.containsKey(elem))
        {
            this.elemCluster.put(elem, cluster);
            this.clusterElems.get(cluster).add(elem);
            return true;
        }
        return false;
    }

    @Override
    public int add(Collection<E> elems, int cluster)
    {
        int i = 0;
        if(cluster >= 0 && cluster < this.getNumClusters())
        {
            for(E elem : elems)
            {
                i += (this.add(elem,cluster)) ? 1 : 0;
            }
        }
        else
        {
            i = -1;
        }
        return i;
    }

    @Override
    public int add(Stream<E> elems, int cluster)
    {
        int i;
        if(cluster >= 0 && cluster < this.getNumClusters())
        {
            i = elems.filter(elem -> this.add(elem, cluster)).mapToInt(x -> 1).sum();
        }
        else
        {
            i = -1;
        }
        return i;
    }

    @Override
    public int getClusterSize(int cluster)
    {
        if(cluster >= 0 && cluster < this.getNumClusters())
        {
            return clusterElems.get(cluster).size();
        }
        else
        {
            return 0;
        }
    }

    @Override
    public boolean divideClusters(int cluster, Clusters<E> division)
    {
        // First, check the correctness:

        // We check that sizes are consistent
        if(this.getClusterSize(cluster) != division.getNumElems()) return false;
        // We get the element list in the cluster:
        List<E> list = this.getElements(cluster).collect(Collectors.toList());
        for(E e : list)
        {
            if(!division.containsElem(e)) return false;
        }

        // Now that we have checked that the elements of the partition are correct, we divide the clusters.
        if(division.getNumClusters() == 1) return true; // in this case, we do not have to do anything.
        else
        {
            int numClusters = this.getNumClusters();
            // We empty the elements in the cluster.
            this.clusterElems.get(cluster).clear();
            // And we add the elements in the "0" element there:

            division.getElements(0).forEach(elem -> this.update(elem, cluster));

            for(int i = 1; i < division.getNumClusters();++i)
            {
                this.addCluster();
                int clustId = numClusters;
                division.getElements(i).forEach(elem -> this.update(elem, clustId));
                ++numClusters;
            }
        }

        return false;
    }
}
