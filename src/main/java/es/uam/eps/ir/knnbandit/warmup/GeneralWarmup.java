/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.warmup;

import es.uam.eps.ir.knnbandit.data.datasets.OfflineDataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Class for the storing the warm-up data for general recommendations.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class GeneralWarmup implements OfflineWarmup
{
    /**
     * A list containing only the ratings in the dataset.
     */
    private final List<FastRating> cleanTraining;
    /**
     * A list containing all the ratings in the warm-up data.
     */
    private final List<FastRating> fullTraining;
    /**
     * Lists for indicating the remaining items to be computed.
     */
    private final List<IntList> availability;
    /**
     * Number of relevant items.
     */
    private final int numRel;

    /**
     * Constructor.
     * @param cleanTraining a list containing the ratings that appear in the dataset.
     * @param fullTraining  a list containing all the ratings in the warm-up data.
     * @param availability  lists for indicating the remaining items to be computed.
     * @param numRel        the number of relevant items.
     */
    protected GeneralWarmup(List<FastRating> cleanTraining, List<FastRating> fullTraining, List<IntList> availability, int numRel)
    {
        this.cleanTraining = cleanTraining;
        this.fullTraining = fullTraining;
        this.availability = availability;
        this.numRel = numRel;
    }

    @Override
    public List<IntList> getAvailability()
    {
        return availability;
    }

    @Override
    public int getNumRel()
    {
        return numRel;
    }

    @Override
    public List<FastRating> getFullTraining()
    {
        return fullTraining;
    }

    @Override
    public List<FastRating> getCleanTraining()
    {
        return cleanTraining;
    }

    /**
     * Loads the warm-up data.
     * @param dataset   the dataset.
     * @param training  the full list of user-item pairs in the warm-up.
     * @param type      filters the list of pairs. If ALL, it does not apply a filter. If ONLYRATED, ignores those user-item pairs not in the dataset.
     * @return the warm-up data.
     */
    public static GeneralWarmup load(OfflineDataset<?,?> dataset, Stream<Pair<Integer>> training, WarmupType type)
    {
        List<FastRating> fullTraining = new ArrayList<>();
        List<FastRating> cleanTraining = new ArrayList<>();
        List<IntList> availability = new ArrayList<>();
        IntList itemList = new IntArrayList();
        dataset.getAllIidx().forEach(itemList::add);

        IntStream.range(0, dataset.numUsers()).forEach(uidx -> availability.add(new IntArrayList(itemList)));

        int numRel = training.mapToInt(t ->
        {
            int uidx = t.v1();
            int iidx = t.v2();
            double value = 0.0;
            Optional<Double> opt = dataset.getPreference(uidx, iidx);
            if(opt.isPresent())
            {
                value = opt.get();
                cleanTraining.add(new FastRating(uidx, iidx, value));
                fullTraining.add(new FastRating(uidx, iidx, value));
                availability.get(uidx).removeInt(availability.get(uidx).indexOf(iidx));
            }
            else if(type == WarmupType.FULL)
            {
                value = Double.NaN;
                fullTraining.add(new FastRating(uidx, iidx, value));
                availability.get(uidx).removeInt(availability.get(uidx).indexOf(iidx));
            }

            return dataset.isRelevant(value) ? 1 : 0;
        }).sum();

        return new GeneralWarmup(fullTraining, cleanTraining, availability, numRel);
    }
}
