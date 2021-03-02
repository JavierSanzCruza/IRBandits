/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.data.preference.updateable;

import es.uam.eps.ir.ranksys.core.preference.PreferenceData;


/**
 * Interface for updateable preference data.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface UpdateablePreferenceData<U, I> extends PreferenceData<U, I>, Updateable<U, I>
{
    /**
     * Given two values (a new one, and an old one), obtains the updated value.
     * @param newValue the new value.
     * @param oldValue the old value.
     * @return the updated value.
     */
    double updatedValue(double newValue, double oldValue);

    /**
     * Clears the preference data.
     */
    void clear();
}
