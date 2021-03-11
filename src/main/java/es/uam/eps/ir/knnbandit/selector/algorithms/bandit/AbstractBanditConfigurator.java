/*
 * Copyright (C) 2021 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * Abstract implementation of a class that allows the configuration of multi-armed bandit suppliers.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class AbstractBanditConfigurator implements BanditConfigurator
{
    @Override
    public List<BanditSupplier> getBandits(JSONArray array)
    {
        List<BanditSupplier> list = new ArrayList<>();
        int numConfigs = array.length();
        for(int i = 0; i < numConfigs; ++i)
        {
            JSONObject obj = array.getJSONObject(i);
            list.add(this.getBandit(obj));
        }
        return list;
    }
}
