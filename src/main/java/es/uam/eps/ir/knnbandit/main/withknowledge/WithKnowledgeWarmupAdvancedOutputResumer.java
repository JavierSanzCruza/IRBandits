package es.uam.eps.ir.knnbandit.main.withknowledge;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.DatasetWithKnowledge;
import es.uam.eps.ir.knnbandit.io.IOType;
import es.uam.eps.ir.knnbandit.main.AdvancedOutputResumer;
import es.uam.eps.ir.knnbandit.main.WarmupAdvancedOutputResumer;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
import es.uam.eps.ir.knnbandit.metrics.CumulativeKnowledgeRecall;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.knnbandit.warmup.GeneralWarmup;
import es.uam.eps.ir.knnbandit.warmup.Warmup;
import es.uam.eps.ir.knnbandit.warmup.WarmupType;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;

public class WithKnowledgeWarmupAdvancedOutputResumer<U,I> extends WarmupAdvancedOutputResumer<U,I>
{
    private final DatasetWithKnowledge<U,I> dataset;
    private final Map<String, Supplier<CumulativeMetric<U,I>>> metrics;

    /**
     * Constructor.
     * @param input file containing the information about the ratings.
     * @param separator a separator for reading the file.
     * @param uParser parser for reading the set of users.
     * @param iParser parser for reading the set of items.
     * @param threshold the relevance threshold.
     * @param useRatings true if we have to consider the real ratings, false to binarize them according to the threshold value.
     * @throws IOException if something fails while reading the dataset.
     */
    public WithKnowledgeWarmupAdvancedOutputResumer(String input, String separator, Parser<U> uParser, Parser<I> iParser, double threshold, boolean useRatings, IOType ioType, boolean gzipped, IOType warmupIoType, boolean warmupGzipped) throws IOException
    {
        super(ioType, gzipped, warmupIoType, warmupGzipped);
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x : (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
        dataset = DatasetWithKnowledge.load(input, uParser, iParser, separator, weightFunction, relevance);
        this.metrics = new HashMap<>();
        metrics.put("recall", CumulativeRecall::new);
        metrics.put("known-recall", () -> new CumulativeKnowledgeRecall<>(KnowledgeDataUse.ONLYKNOWN));
        metrics.put("unknown-recall", () -> new CumulativeKnowledgeRecall<>(KnowledgeDataUse.ONLYUNKNOWN));
        metrics.put("gini", CumulativeGini::new);
    }

    @Override
    protected Dataset<U, I> getDataset()
    {
        return dataset;
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, I>>> getMetrics()
    {
        return metrics;
    }

    @Override
    protected Warmup getWarmup(List<Pair<Integer>> trainData)
    {
        return GeneralWarmup.load(dataset, trainData.stream(), WarmupType.ONLYRATINGS);
    }
}
