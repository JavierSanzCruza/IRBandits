/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import es.uam.eps.ir.knnbandit.Constants;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AbstractSimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.AbstractInteractiveRecommender;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.mf.Factorization;
import es.uam.eps.ir.ranksys.mf.Factorizer;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.core.util.tuples.Tuple2id;

import java.util.Comparator;
import java.util.Enumeration;
import java.util.Optional;
import java.util.PriorityQueue;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.stream.Stream;

/**
 * Interactive version of matrix factorization algorithms. Legacy version.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class AbstractInteractiveMF<U, I> extends AbstractInteractiveRecommender<U, I>
{
    /**
     * Number of hits before the recommender is updated.
     */
    protected final int limitCounter;
    /**
     * Factorizer for obtaining the factorized matrices.
     */
    protected final Factorizer<U, I> factorizer;
    /**
     * Number of latent factors to use.
     */
    protected final int k;
    /**
     * Decomposition in different matrices.
     */
    protected Factorization<U, I> factorization;

    /**
     * The current rating matrix.
     */
    protected AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData;
    /**
     * Current hit counter.
     */
    protected int currentCounter = 0;

    /**
     * Constructor.
     *
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param hasRating  True if we must ignore unknown items when updating.
     * @param k          Number of latent factors to use.
     * @param factorizer Factorizer for obtaining the factorized matrices.
     */
    public AbstractInteractiveMF(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int k, Factorizer<U, I> factorizer, AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData, int limitCounter)
    {
        super(uIndex, iIndex, hasRating);
        this.factorizer = factorizer;
        this.k = (k > 0) ? k : uIndex.numUsers();
        Enumeration<String> loggers = LogManager.getLogManager().getLoggerNames();
        while (loggers.hasMoreElements())
        {
            LogManager.getLogManager().getLogger(loggers.nextElement()).setLevel(Level.OFF);
        }

        this.retrievedData = retrievedData;
        this.limitCounter = limitCounter;
    }

    /**
     * Constructor.
     *
     * @param uIndex     User index.
     * @param iIndex     Item index.
     * @param hasRating  True if we must ignore unknown items when updating.
     * @param k          Number of latent factors to use.
     * @param factorizer Factorizer for obtaining the factorized matrices.
     */
    public AbstractInteractiveMF(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int rngSeed, int k, Factorizer<U, I> factorizer, AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData, int limitCounter)
    {
        super(uIndex, iIndex, hasRating, rngSeed);
        this.factorizer = factorizer;
        this.k = (k > 0) ? k : uIndex.numUsers();
        Enumeration<String> loggers = LogManager.getLogManager().getLoggerNames();
        while (loggers.hasMoreElements())
        {
            LogManager.getLogManager().getLogger(loggers.nextElement()).setLevel(Level.OFF);
        }

        this.retrievedData = retrievedData;
        this.limitCounter = limitCounter;
    }

    @Override
    public void init()
    {
        super.init();
        this.retrievedData.clear();
        this.factorization = factorizer.factorize(k, retrievedData);
    }

    @Override
    public void init(Stream<FastRating> values)
    {
        super.init();
        this.retrievedData.clear();
        values.forEach(triplet -> this.retrievedData.updateRating(triplet.uidx(), triplet.iidx(), triplet.value()));
        this.factorization = factorizer.factorize(k, retrievedData);
    }

    /*@Override
    public void init(FastPreferenceData<U, I> prefData)
    {
        this.retrievedData.clear();
        this.factorization = factorizer.factorize(k, retrievedData);
    }*/


    @Override
    public int next(int uidx, IntList availability)
    {
        if (availability == null || availability.isEmpty())
        {
            return -1;
        }

        DoubleMatrix1D pu = factorization.getUserVector(uIndex.uidx2user(uidx));
        if (pu == null)
        {
            return availability.get(rng.nextInt(availability.size()));
        }

        DoubleMatrix2D itemMatrix = factorization.getItemMatrix();
        double max = Double.NEGATIVE_INFINITY;
        IntList top = new IntArrayList();

        for (int iidx : availability)
        {
            double val = itemMatrix.viewRow(iidx).zDotProduct(pu);
            if (Double.isNaN(val))
            {
                val = Double.NEGATIVE_INFINITY;
            }
            if (top.isEmpty() || max < val)
            {
                top = new IntArrayList();
                top.add(iidx);
                max = val;
            }
            else if (max == val)
            {
                top.add(iidx);
            }
        }

        int topSize = top.size();
        if (topSize == 1)
        {
            return top.get(0);
        }
        else
        {
            return top.get(rng.nextInt(top.size()));
        }
    }

    @Override
    public IntList next(int uidx, IntList availability, int k)
    {
        if (availability == null || availability.isEmpty())
        {
            return new IntArrayList();
        }

        IntList top = new IntArrayList();
        int num = Math.min(k, availability.size());

        DoubleMatrix1D pu = factorization.getUserVector(uIndex.uidx2user(uidx));
        if (pu != null)
        {
            DoubleMatrix2D itemMatrix = factorization.getItemMatrix();
            PriorityQueue<Tuple2id> queue = new PriorityQueue<>(num, Comparator.comparingDouble(x -> x.v2));
            for (int iidx : availability)
            {
                double val = itemMatrix.viewRow(iidx).zDotProduct(pu);
                if (Double.isNaN(val))
                {
                    val = Double.NEGATIVE_INFINITY;
                }

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
        }
        else
        {
            while(top.size() < num)
            {
                int idx = rng.nextInt(availability.size());
                int item = availability.get(idx);
                if(!top.contains(item)) top.add(item);
            }
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

        boolean hasRating = false;
        double oldValue = 0;
        // First, we find whether we have a rating or not:
        if(this.retrievedData.numItems(uidx) > 0 && this.retrievedData.numUsers(iidx) > 0)
        {
            Optional<IdxPref> opt = this.retrievedData.getPreference(uidx, iidx);
            hasRating = opt.isPresent();
            if(hasRating)
            {
                oldValue = opt.get().v2;
            }
        }

        if(!hasRating)
        {
            this.retrievedData.updateRating(uidx, iidx, newValue);
            if(value > 0.0)
                ++this.currentCounter;
        }
        else if(this.retrievedData.updateRating(uidx, iidx, newValue))
        {
            Optional<IdxPref> opt = this.retrievedData.getPreference(uidx, iidx);
            if(opt.isPresent())
            {
                double auxNewValue = opt.get().v2;
                if(auxNewValue != oldValue || auxNewValue > 0.0)
                {
                    ++this.currentCounter;
                }
            }
        }

        if (currentCounter >= this.limitCounter)
        {
            this.currentCounter = 0;
            this.factorization = factorizer.factorize(k, retrievedData);
        }
    }

}
