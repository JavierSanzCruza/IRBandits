/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.recommendation.loop.update;

import es.uam.eps.ir.knnbandit.data.datasets.ContactDataset;
import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.recommendation.loop.selection.Selection;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.Warmup;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Considering a contact recommendation dataset, where the items are users, this
 * mechanism always selects the (uidx, iidx, value) rating. If the network is undirected,
 * the rating (iidx, uidx, value) is also added to the list. Finally, if the network is
 * directed, we do not allow recommending reciprocal links and the link (uidx, iidx) exists,
 * then, we add the (iidx, uidx, value2) triplet to the list for updating the recommenders.
 *
 * Only the original (uidx, iidx, value) is used for updating the metrics.
 *
 * @param <U> type of the users.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class ContactUpdate<U> implements UpdateStrategy<U,U>
{
    /**
     * True if we do not want to recommend reciprocal links to existing ones, false otherwise.
     */
    private final boolean notReciprocal;
    /**
     * The contact recommendation dataset.
     */
    private ContactDataset<U> dataset;

    /**
     * Constructor.
     * @param notReciprocal true if we do not want to recommend reciprocal links to existing ones, false otherwise.
     */
    public ContactUpdate(boolean notReciprocal)
    {
        this.notReciprocal = notReciprocal;
    }

    /**
     * Constructor. We assume that we want to recommend reciprocal links to existing ones.
     */
    public ContactUpdate()
    {
        this.notReciprocal = false;   
    }

    @Override
    public void init(Dataset<U, U> dataset)
    {
        this.dataset = ((ContactDataset<U>) dataset);
    }

    @Override
    public Pair<List<FastRating>> selectUpdate(int uidx, int iidx, Selection<U,U> selection)
    {
        if(selection.isAvailable(uidx, iidx))
        {
            List<FastRating> list = new ArrayList<>();
            List<FastRating> metricList = new ArrayList<>();

            Optional<Double> value = dataset.getPreference(uidx, iidx);
            if (value.isPresent())
            {
                FastRating pair = new FastRating(uidx, iidx, value.get());
                list.add(pair);
                metricList.add(pair);
                if (!dataset.isDirected())
                {
                    pair = new FastRating(iidx, uidx, value.get());
                    list.add(pair);
                }
                else if (this.notReciprocal && selection.isAvailable(iidx, uidx))
                {
                    value = dataset.getPreference(iidx, uidx);
                    pair = new FastRating(iidx, uidx, value.orElse(0.0));
                    list.add(pair);
                }
            }

            return new Pair<>(list, metricList);
        }
        return new Pair<>(new ArrayList<>(), new ArrayList<>());
    }

    @Override
    public List<FastRating> getList(Warmup warmup)
    {
        List<FastRating> list = new ArrayList<>(warmup.getFullTraining());
        for(FastRating rating : list)
        {
            if(!dataset.isDirected())
            {
                list.add(new FastRating(rating.iidx(), rating.uidx(), rating.value()));
            }
            else if(notReciprocal && rating.value() > 0.0)
            {
                list.add(new FastRating(rating.iidx(), rating.uidx(), dataset.getPreference(rating.uidx(), rating.iidx()).orElse(0.0)));
            }
        }

        return list;
    }
}
