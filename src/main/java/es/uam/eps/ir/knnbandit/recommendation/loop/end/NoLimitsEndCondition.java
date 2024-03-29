/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.end;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.ranksys.fast.FastRecommendation;

/**
 * End condition specifying that the loop has no end.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class NoLimitsEndCondition implements EndCondition
{
    @Override
    public void init(Dataset<?,?> dataset)
    {

    }

    @Override
    public boolean hasEnded()
    {
        return false;
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {

    }

    @Override
    public void update(FastRecommendation fastRec)
    {

    }
}
