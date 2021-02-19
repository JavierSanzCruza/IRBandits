package es.uam.eps.ir.knnbandit.selector.metric;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;

@FunctionalInterface
public interface CumulativeMetricFunction<U, I>
{
    /**
     * Initializes the metric, given a dataset.
     * @param dataset the dataset.
     * @return a cumulative metric.
     */
    CumulativeMetric<U,I> apply(Dataset<U,I> dataset);
}
