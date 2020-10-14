package es.uam.eps.ir.knnbandit.data.datasets;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;

import java.util.function.DoublePredicate;

/**
 * Dataset represented as a stream of logged data, advancing over time.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class StreamDataset<U,I>
{
    protected final FastUpdateableUserIndex<U> uIndex;
    protected final FastUpdateableItemIndex<I> iIndex;

    public StreamDataset(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;
    }
}
