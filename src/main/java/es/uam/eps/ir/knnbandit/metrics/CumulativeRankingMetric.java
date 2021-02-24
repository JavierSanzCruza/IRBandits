package es.uam.eps.ir.knnbandit.metrics;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import es.uam.eps.ir.ranksys.metrics.RecommendationMetric;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Class that wraps a classical recommendation metric (i.e. P@k, nDCG@k...).
 * It averages the recommendation metric over the different recommendations
 * that are done.
 * @param <U> the type of the users.
 * @param <I> the type of the items.
 */
public class CumulativeRankingMetric<U,I> implements CumulativeMetric<U,I>
{
    /**
     * The dataset.
     */
    private Dataset<U,I> dataset;
    /**
     * The recommendation metric we want to apply.
     */
    private final RecommendationMetric<U,I> recMetric;
    /**
     * The number of iterations.
     */
    private int numRecs;
    /**
     * The current metric value.
     */
    private double currentValue;

    /**
     * Constructor.
     * @param recMetric A recommendation metric.
     */
    public CumulativeRankingMetric(RecommendationMetric<U,I> recMetric)
    {
        numRecs = 0;
        currentValue = 0.0;
        this.recMetric = recMetric;
    }


    @Override
    public double compute()
    {
        return this.currentValue;
    }

    @Override
    public void initialize(Dataset<U, I> dataset)
    {
        this.dataset = dataset;
        this.numRecs = 0;
        this.currentValue = 0.0;
    }

    @Override
    public void initialize(Dataset<U, I> dataset, List<FastRating> train)
    {
        // TODO: Deal with training pairs. As of now, this has not been contemplated for metrics that require past information (i.e. EPC, EPD).
        this.dataset = dataset;
        this.numRecs = 0;
        this.currentValue = 0.0;
    }

    @Override
    public void update(int uidx, int iidx, double value)
    {
        U u = dataset.uidx2user(uidx);
        Tuple2od<I> i = new Tuple2od<>(dataset.iidx2item(iidx), value);
        List<Tuple2od<I>> rec = new ArrayList<>();
        rec.add(i);
        Recommendation<U,I> recom = new Recommendation<>(u, rec);
        double val = recMetric.evaluate(recom);
        ++this.numRecs;
        this.currentValue = this.currentValue + (val - currentValue)/(numRecs + 0.0);
    }

    @Override
    public void update(FastRecommendation fastRec)
    {
        Recommendation<U,I> rec = new Recommendation<>(this.dataset.uidx2user(fastRec.getUidx()),
                                                       fastRec.getIidxs().stream().map(dataset::iidx2item).collect(Collectors.toList()));
        double val = recMetric.evaluate(rec);
        ++this.numRecs;
        this.currentValue = this.currentValue + (val - currentValue)/(numRecs + 0.0);
    }

    @Override
    public void reset()
    {
        this.currentValue = 0.0;
        this.numRecs = 0;
    }
}
