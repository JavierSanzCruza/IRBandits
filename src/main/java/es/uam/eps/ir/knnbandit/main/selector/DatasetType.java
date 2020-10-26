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

/**
 * Class with constants for defining the different types of datasets.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class DatasetType
{
    /**
     * Identifier for general domain-independent recommendation datasets.
     */
    final static String GENERAL = "general";
    /**
     * Identifier for contact recommendation datasets.
     */
    final static String CONTACT = "contact";
    /**
     * Identifier for datasets containing information about whether the user knew about the items
     * prior to giving them a rating.
     */
    final static String KNOWLEDGE = "knowledge";
    /**
     * Identifier for streaming datasets.
     */
    final static String STREAM = "stream";
}
