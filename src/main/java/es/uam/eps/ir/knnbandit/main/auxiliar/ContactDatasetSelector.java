package es.uam.eps.ir.knnbandit.main.auxiliar;

import org.ranksys.formats.parsing.Parser;

public class ContactDatasetSelector<U> extends DatasetSelector<U,U>
{
    private final boolean directed;
    public ContactDatasetSelector(String input, String separator, Parser<U> uParser, boolean directed)
    {
        super(input, separator, uParser, uParser);
        this.directed = directed;
    }

    public ContactDatasetSelector(String input, String separator, Parser<U> uParser, double threshold, boolean useRatings, DatasetType datasetType)
    {
        super(input, separator, uParser, uParser, threshold, useRatings, datasetType);
        this.directed = false;
    }

    public ContactDatasetSelector(String input, String separator, Parser<U> uParser, String userIndex, String itemIndex)
    {
        super(input, separator, uParser, uParser, userIndex, itemIndex);
        this.directed = false;
    }
}
