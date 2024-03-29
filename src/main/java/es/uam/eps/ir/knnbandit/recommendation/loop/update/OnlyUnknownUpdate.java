/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.update;

import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;

/**
 * Class that selects (user, item) pairs for the update, but takes the ratings
 * only if the user did not know the item. Otherwise, it takes the zero value.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class OnlyUnknownUpdate<U,I> extends WithKnowledgeUpdate<U,I>
{
    /**
     * Constructor.
     */
    public OnlyUnknownUpdate()
    {
        super(KnowledgeDataUse.ONLYUNKNOWN);
    }
}
