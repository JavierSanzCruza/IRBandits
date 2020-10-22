package es.uam.eps.ir.knnbandit.main.auxiliar;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.DatasetWithKnowledge;
import es.uam.eps.ir.knnbandit.data.datasets.GeneralOfflineDataset;
import es.uam.eps.ir.knnbandit.data.datasets.ReplayerStreamDataset;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

public class DatasetSelector<U,I>
{
    private final String input;
    private final String separator;
    private final double threshold;
    private final String userIndex;
    private final String itemIndex;
    private final Parser<U> uParser;
    private final Parser<I> iParser;
    private final boolean useRatings;
    protected DatasetType datasetType;

    /**
     * Basic constructor.
     * @param input file containing the input data.
     * @param separator the separator.
     * @param uParser user parser.
     * @param iParser item parser.
     */
    protected DatasetSelector(String input, String separator, Parser<U> uParser, Parser<I> iParser)
    {
        this.input = input;
        this.separator = separator;
        this.uParser = uParser;
        this.iParser = iParser;
        this.threshold = 0.5;
        this.userIndex = "";
        this.itemIndex = "";
        this.useRatings = false;
    }

    public DatasetSelector(String input, String separator, Parser<U> uParser, Parser<I> iParser, double threshold, boolean useRatings, DatasetType datasetType)
    {
        this.input = input;
        this.separator = separator;
        this.uParser = uParser;
        this.iParser = iParser;
        this.threshold = threshold;
        this.useRatings = useRatings;
        this.userIndex = "";
        this.itemIndex = "";
        this.datasetType = datasetType;
    }

    public DatasetSelector(String input, String separator, Parser<U> uParser, Parser<I> iParser, String userIndex, String itemIndex)
    {
        this.input = input;
        this.separator = separator;
        this.userIndex = userIndex;
        this.itemIndex = itemIndex;
        this.datasetType = DatasetType.STREAM;
        this.useRatings = false;
        this.threshold = 0.5;
        this.uParser = uParser;
        this.iParser = iParser;
    }


    public Dataset<U,I> readDataset() throws IOException
    {
        switch (datasetType)
        {
            case GENERAL:
                DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x : (double x) -> (x >= threshold ? 1.0 : 0.0);
                DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
                return GeneralOfflineDataset.load(input, uParser, iParser, separator, weightFunction, relevance);
            case WITHKNOWLEDGE:
                weightFunction = useRatings ? (double x) -> x : (double x) -> (x >= threshold ? 1.0 : 0.0);
                relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
                return DatasetWithKnowledge.load(input, uParser, iParser, separator, weightFunction, relevance);
            case STREAM:
                return ReplayerStreamDataset.load(input, userIndex, itemIndex, separator, uParser, iParser);
        }
    }
}
