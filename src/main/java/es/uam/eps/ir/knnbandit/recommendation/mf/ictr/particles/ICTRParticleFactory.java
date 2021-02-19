/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.mf.ictr.particles;

import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;

/**
 * Particle factory for the ICTR algorithm.
 *
 * @param <U> Type of the users.
 * @param <I> Type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ICTRParticleFactory<U, I>
{
    /**
     * Creates a new particle.
     *
     * @param uIndex user index.
     * @param iIndex item index.
     * @param K      the number of latent factors for users/items.
     * @return the created particle if everything went OK, null otherwise.
     */
    public ICTRParticle<U, I> create(FastUserIndex<U> uIndex, FastItemIndex<I> iIndex, int K)
    {
        return new ICTRParticle<>(uIndex, iIndex, K);
    }
}
