/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.selection;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.GeneralDataset;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.user.UserSelector;
import es.uam.eps.ir.knnbandit.warmup.OfflineWarmup;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import it.unimi.dsi.fastutil.ints.*;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Target user / candidate item selection mechanism for non-sequential offline datasets,
 * i.e. for the cases where the order of the ratings in the dataset is not important.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class NonSequentialSelection<U,I> implements Selection<U,I>
{
    /**
     * List of users we can recommend items to.
     */
    protected final IntList userList;

    /**
     * A list indicating the set of items we can recommend to each user in the system.
     */
    protected final Int2ObjectMap<IntList> availability;

    /**
     * Random seed.
     */
    private final int rngSeed;

    /**
     * Object that selects users for the next iteration.
     */
    private final UserSelector uSel;
    /**
     * Random number generator.
     */
    private Random rng;

    /**
     * The current number of target users.
     */
    private int numUsers;
    /**
     * The last removed index from the target user list.
     */
    private int lastRemovedIndex;

    /**
     * Constructor.
     * @param rngSeed random seed.
     * @param uSel user selector.
     */
    public NonSequentialSelection(int rngSeed, UserSelector uSel)
    {
        this.rngSeed = rngSeed;
        this.rng = new Random(rngSeed);
        this.userList = new IntArrayList();
        this.availability = new Int2ObjectOpenHashMap<>();
        this.numUsers = 0;
        this.uSel = uSel;
        this.lastRemovedIndex = -1;
    }

    @Override
    public int selectTarget()
    {
        int index = uSel.next(numUsers, lastRemovedIndex);
        if(index >= 0)
        {
            if(uSel.reshuffle()) Collections.shuffle(userList, rng);
            return userList.get(index);
        }
        return -1;
    }

    @Override
    public IntList selectCandidates(int uidx)
    {
        if(this.availability.get(uidx) == null || this.availability.get(uidx).isEmpty())
        {
            int index = this.userList.indexOf(uidx);
            if(index >= 0)
            {
                this.userList.remove(index);
                this.numUsers--;
                this.lastRemovedIndex = index;
            }

        }
        return this.availability.get(uidx);
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        int index = this.availability.get(uidx).indexOf(iidx);
        if(index >= 0)
        {
            this.availability.get(uidx).removeInt(index);
        }

        if(this.availability.get(uidx).isEmpty())
        {
            int auxIndex = this.userList.indexOf(uidx);
            this.userList.remove(auxIndex);
            this.numUsers--;
            this.lastRemovedIndex = auxIndex;
        }
    }

    @Override
    public void init(Dataset<U, I> dataset)
    {
        GeneralDataset<U,I> general = ((GeneralDataset<U,I>) dataset);
        this.userList.clear();
        this.availability.clear();
        this.rng = new Random(rngSeed);

        general.getUidxWithPreferences().forEach(uidx ->
        {
            userList.add(uidx);
            availability.put(uidx, dataset.getAllIidx().boxed().collect(Collectors.toCollection(IntArrayList::new)));
        });

        Collections.shuffle(this.userList, rng);
        this.numUsers = userList.size();
        this.lastRemovedIndex = -1;
    }

    @Override
    public void init(Dataset<U, I> dataset, Warmup warmup)
    {
        GeneralDataset<U,I> general = ((GeneralDataset<U,I>) dataset);
        this.userList.clear();
        this.availability.clear();
        this.rng = new Random(rngSeed);

        List<IntList> warmupAvailability = ((OfflineWarmup) warmup).getAvailability();
        general.getUidxWithPreferences().forEach(uidx ->
        {
            // First, the availability.
            IntList uAvailable = warmupAvailability.get(uidx);
            if(uAvailable != null && !uAvailable.isEmpty())
            {
                this.userList.add(uidx);
                this.availability.put(uidx, new IntArrayList(uAvailable));
            }
        });

        IntSet keySet = availability.keySet();
        keySet.forEach(uidx ->
        {
            if(availability.get(uidx).isEmpty())
            {
                this.userList.remove(uidx);
            }
        });

        Collections.shuffle(this.userList, rng);
        this.numUsers = userList.size();
        this.lastRemovedIndex = -1;
    }

    @Override
    public boolean isAvailable(int uidx, int iidx)
    {
        return this.availability.containsKey(uidx) && this.availability.get(uidx).contains(iidx);
    }
}
