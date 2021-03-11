package es.uam.eps.ir.knnbandit.data.datasets.builder;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.utils.FastRating;

import java.util.List;
import java.util.stream.Stream;

public interface DatasetBuilder<U,I>
{
    Dataset<U,I> buildFromStream(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, List<FastRating> ratings);
}