/*
 * Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.main.selector;

import es.uam.eps.ir.knnbandit.recommendation.loop.end.EndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.NoLimitsEndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.NumIterEndCondition;
import es.uam.eps.ir.knnbandit.recommendation.loop.end.PercentagePositiveRatingsEndCondition;

import java.util.function.Supplier;

/**
 * Class for selecting a suitable end condition for the loops.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class EndConditionSelector
{
    /**
     * Given a value, determines the end condition of the recommendation interactive loop to use.
     * @param value the value. If 0, the loop does not end unless no recommendation can be done. If between 0 and 1,
     *              it fixes a percentage of the positive ratings to recover. Finally, an integer value greater than
     *              1 fixes the number of iterations of the recommendation loop.
     * @return a supplier for the end condition.
     */
    public static Supplier<EndCondition> select(Double value)
    {
        if(value == 0.0 || value >= 1.0)
        {
            int numIter = (value >= 1.0) ? value.intValue() : Integer.MAX_VALUE;
            return value == 0.0 ? NoLimitsEndCondition::new : () -> new NumIterEndCondition(numIter);
        }
        else
        {
            return () -> new PercentagePositiveRatingsEndCondition(value);
        }
    }
}
