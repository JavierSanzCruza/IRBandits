package es.uam.eps.ir.knnbandit.data.datasets.builder;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.GeneralDataset;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple3;

import java.util.List;
import java.util.function.DoublePredicate;

/**
 * Class for building general-type datasets.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class BinaryGeneralDatasetBuilder<U,I> implements DatasetBuilder<U,I>
{
    @Override
    public Dataset<U, I> buildFromStream(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, List<FastRating> ratings)
    {
        DoublePredicate relevance = (r) -> r > 0.0;
        SimpleFastPreferenceData<U,I> prefData = SimpleFastPreferenceData.load(ratings.stream().map(r -> new Tuple3<>(uIndex.uidx2user(r.uidx()), iIndex.iidx2item(r.iidx()), r.value())), uIndex, iIndex);
        int numRel = ratings.stream().mapToInt(r -> (r.value() > 0.0) ? 1 : 0).sum();
        return new GeneralDataset<>(uIndex, iIndex, prefData, numRel, relevance);
    }
}