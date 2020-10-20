package es.uam.eps.ir.knnbandit.data.datasets;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;

public interface Dataset<U,I> extends FastUpdateableUserIndex<U>, FastUpdateableItemIndex<I>
{

}
