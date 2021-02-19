package es.uam.eps.ir.knnbandit.selector.metric;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.metrics.ClickthroughRate;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;

public class MetricFunctions
{
    public static <U,I> CumulativeMetricFunction<U,I> recall()
    {
        return (Dataset<U,I> dataset) -> new CumulativeRecall<>();
    }

    public static <U,I> CumulativeMetricFunction<U,I> clicktroughRate()
    {
        return (Dataset<U,I> dataset) -> new ClickthroughRate<>();
    }
}
