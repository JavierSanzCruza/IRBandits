package es.uam.eps.ir.knnbandit.recommendation.loop.selection;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import it.unimi.dsi.fastutil.ints.IntList;

/**
 * Interface for classes that select the target and candidate items for the interactive recommendation
 * dataset.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public interface Selection<U,I>
{
    /**
     * Selects the next target user of the recommendation.
     * @return the target user of the recommendation, -1 if no one can be selected.
     */
    int selectTarget();

    /**
     * Selects a list of candidate items for the recommendation.
     * @param uidx the index of the target user
     * @return the list of candidate items for the recommendation, null if there is not any.
     */
    IntList selectCandidates(int uidx);

    /**
     * Updates the selection strategy.
     * @param uidx user index.
     * @param iidx item index.
     * @param value payoff of the user/index term.
     */
    void update(int uidx, int iidx, double value);

    /**
     * Initializes the selector.
     * @param dataset the dataset containing information.
     */
    void init(Dataset<U,I> dataset);

    /**
     * Initializes the selector after some warmup data has been processed.
     * @param dataset the dataset.
     * @param warmup the warm-up
     */
    void init(Dataset<U,I> dataset, Warmup warmup);

    boolean isAvailable(int uidx, int iidx);


}
