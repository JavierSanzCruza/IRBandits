/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.recommendation.knn.similarities;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.sim.Similarity;

/**
 * Updateable version of similarity.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface UpdateableSimilarity extends Similarity
{
    /**
     * In case there is a norm to update, this updates it.
     * @param uidx Identifier of the user.
     * @param value Rating of the user.
     */
    void updateNorm(int uidx, double value);

    /**
     * In case there is a norm to update, this updates it considering that the rating disappears.
     * @param uidx Identifier of the user.
     * @param value Rating of the user that disappears.
     */
    void updateNormDel(int uidx, double value);

    /**
     * Updates the similarity between two users.
     *
     * @param uidx Identifier of the first user.
     * @param vidx Identifier of the second user.
     * @param iidx Identifier of the item.
     * @param uval Rating of the first user for the item.
     * @param vval Rating of the second user for the item.
     */
    void update(int uidx, int vidx, int iidx, double uval, double vval);

    /**
     * Updates the similarity between two users, when a rating is removed.
     * @param uidx Identifier of the user whose rating is removed.
     * @param vidx Identifier of the second user.
     * @param iidx Identifier of the item whose rating has been removed.
     * @param uval The value of the removed rating of the first user for the item.
     * @param vval Rating of the second user for the item.
     */
    void updateDel(int uidx, int vidx, int iidx, double uval, double vval);


    /**
     * Initializes the similarity when no data is available.
     */
    void initialize();

    /**
     * Initializes the similarity.
     *
     * @param trainData Training data.
     */
    void initialize(FastPreferenceData<?, ?> trainData);
}
