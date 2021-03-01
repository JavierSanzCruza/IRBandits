package es.uam.eps.ir.knnbandit.main.contact;

import es.uam.eps.ir.knnbandit.data.datasets.ContactDataset;
import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.GeneralDataset;
import es.uam.eps.ir.knnbandit.io.IOType;
import es.uam.eps.ir.knnbandit.main.AdvancedOutputResumer;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;

public class ContactAdvancedOutputResumer<U> extends AdvancedOutputResumer<U,U>
{
    private final ContactDataset<U> dataset;
    private final Map<String, Supplier<CumulativeMetric<U,U>>> metrics;
    /**
     * Constructor.
     * @param input file containing the information about the ratings.
     * @param separator a separator for reading the file.
     */
    public ContactAdvancedOutputResumer(String input, String separator, Parser<U> parser, boolean directed, boolean notReciprocal, IOType ioType, boolean gzipped)
    {
        super(ioType, gzipped);
        dataset = ContactDataset.load(input, directed, notReciprocal, parser, separator);
        this.metrics = new HashMap<>();
        metrics.put("recall", CumulativeRecall::new);
        metrics.put("gini", CumulativeGini::new);
    }

    @Override
    protected Dataset<U, U> getDataset()
    {
        return dataset;
    }

    @Override
    protected Map<String, Supplier<CumulativeMetric<U, U>>> getMetrics()
    {
        return metrics;
    }
}
