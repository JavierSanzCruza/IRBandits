package es.uam.eps.ir.knnbandit.main.withknowledge;

import es.uam.eps.ir.knnbandit.data.datasets.Dataset;
import es.uam.eps.ir.knnbandit.data.datasets.DatasetWithKnowledge;
import es.uam.eps.ir.knnbandit.main.AdvancedOutputResumer;
import es.uam.eps.ir.knnbandit.main.TrainingStatistics;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

public class WithKnowledgeTrainingStatistics<U,I> extends TrainingStatistics<U,I>
{
    private final DatasetWithKnowledge<U,I> dataset;

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
    public WithKnowledgeTrainingStatistics(String input, String separator, Parser<U> uParser, Parser<I> iParser, double threshold, boolean useRatings) throws IOException
    {
        DoubleUnaryOperator weightFunction = useRatings ? (double x) -> x : (double x) -> (x >= threshold ? 1.0 : 0.0);
        DoublePredicate relevance = useRatings ? (double x) -> (x >= threshold) : (double x) -> (x > 0.0);
        dataset = DatasetWithKnowledge.load(input, uParser, iParser, separator, weightFunction, relevance);
    }

    @Override
    protected Dataset<U, I> getDataset()
    {
        return dataset;
    }
}
