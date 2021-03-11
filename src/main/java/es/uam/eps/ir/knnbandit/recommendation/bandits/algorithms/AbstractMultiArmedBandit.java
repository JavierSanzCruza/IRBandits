/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms;

import es.uam.eps.ir.knnbandit.UntieRandomNumber;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.util.Random;

/**
 * Bandit in which arms are items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class AbstractMultiArmedBandit implements MultiArmedBandit
{
    /**
     * Untie random.
     */
    protected final Random untierng;

    /**
     * The number of arms in the bandit.
     */
    protected final int numArms;

    /**
     * Constructor.
     */
    public AbstractMultiArmedBandit(int numArms)
    {
        this.untierng = new Random(UntieRandomNumber.RNG);
        this.numArms = numArms;
    }

    @Override
    public IntList next(IntList available, ValueFunction valFunc, int k)
    {
        IntList avCopy = new IntArrayList();
        available.forEach(avCopy::add);

        IntList list = new IntArrayList();
        int num = Math.min(available.size(), k);
        for(int i = 0; i < num; ++i)
        {
            int elem = this.next(avCopy, valFunc);
            list.add(elem);
            avCopy.remove(avCopy.indexOf(elem));
        }

        return list;
    }

    @Override
    public int getNumArms()
    {
        return numArms;
    }
}
