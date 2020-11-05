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
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AbstractSimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.SimpleFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.mf.Factorization;
import es.uam.eps.ir.ranksys.mf.Factorizer;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;

import java.util.Enumeration;
import java.util.Optional;
import java.util.function.BiPredicate;
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
public abstract class AbstractInteractiveMF<U, I> extends InteractiveRecommender<U, I>
{
    /**
     * Number of hits before the recommender is updated.
     */
    protected final static int LIMITCOUNTER = 100;
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
    public AbstractInteractiveMF(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int k, Factorizer<U, I> factorizer, AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData)
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
    public AbstractInteractiveMF(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, boolean hasRating, int rngSeed, int k, Factorizer<U, I> factorizer, AbstractSimpleFastUpdateablePreferenceData<U,I> retrievedData)
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
    public void update(int uidx, int iidx, double value)
    {
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
            this.retrievedData.updateRating(uidx, iidx, value);
            if(value > 0.0)
                ++this.currentCounter;
        }
        else if(this.retrievedData.updateRating(uidx, iidx, value))
        {
            Optional<IdxPref> opt = this.retrievedData.getPreference(uidx, iidx);
            if(opt.isPresent())
            {
                double newValue = opt.get().v2;
                if(newValue != oldValue || newValue > 0.0)
                {
                    ++this.currentCounter;
                }
            }
        }

        if (currentCounter >= LIMITCOUNTER)
        {
            this.currentCounter = 0;
            this.factorization = factorizer.factorize(k, retrievedData);
        }
    }

}
