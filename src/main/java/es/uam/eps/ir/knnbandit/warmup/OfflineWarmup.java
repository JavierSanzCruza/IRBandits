package es.uam.eps.ir.knnbandit.warmup;

import it.unimi.dsi.fastutil.ints.IntList;

import java.util.List;

public interface OfflineWarmup extends Warmup
{
    /**
     * Gets the availability lists.
     *
     * @return the availability lists, null if the Initializer has not been configured.
     */
    List<IntList> getAvailability();
}
