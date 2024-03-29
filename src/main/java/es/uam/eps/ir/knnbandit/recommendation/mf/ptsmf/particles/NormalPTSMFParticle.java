/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles;

import cern.colt.matrix.DoubleMatrix1D;
import es.uam.eps.ir.knnbandit.recommendation.mf.Particle;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;

public class NormalPTSMFParticle<U, I> extends PTSMFParticle<U, I>
{
    /**
     * Constructor.
     *
     * @param uIndex User index.
     * @param iIndex Item index.
     * @param K      Number of latent factors.
     * @param sigma  Variance of the ratings.
     * @param sigmaP Variance of the user latent factors.
     * @param sigmaQ Variance of the item latent factors.
     */
    public NormalPTSMFParticle(FastUserIndex<U> uIndex, FastItemIndex<I> iIndex, int K, double sigma, double sigmaP, double sigmaQ)
    {
        super(uIndex, iIndex, K, sigma, sigmaP, sigmaQ, false);
    }


    @Override
    public Particle<U, I> clone()
    {
        PTSMFParticle<U, I> particle = new NormalPTSMFParticle<>(this.getUserIndex(), this.getItemIndex(), this.K, this.sigma, this.sigmaP, this.sigmaQ);
        this.clone(particle);
        return particle;
    }

    @Override
    protected double updateSigmaP()
    {
        return this.sigmaP;
    }

    @Override
    protected void updateNormP(int uidx, DoubleMatrix1D pu)
    {

    }
}
