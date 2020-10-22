package es.uam.eps.ir.knnbandit.data.datasets.reader;

import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.parsing.Parsers;

import java.util.Collection;
import java.util.HashSet;

public class StreamCandidateSelectionDatasetReader<U,I> extends StreamDatasetReader<U,I>
{
    public StreamCandidateSelectionDatasetReader(String file, Parser<U> uParser, Parser<I> iParser, String separator)
    {
        super(file, uParser, iParser, separator);
    }

    protected LogRegister<U,I> processRegister(String line)
    {
        LogRegister<U,I> register;

        // Process the register:
        String[] split = line.split(separator);
        if(split.length < 3)
        {
            return null;
        }

        U u = uParser.parse(split[0]);
        I i = iParser.parse(split[1]);
        double value = Parsers.dp.parse(split[2]);
        Collection<I> candidates = new HashSet<>();
        for (int j = 3; j < split.length; ++j)
        {
            candidates.add(iParser.parse(split[j]));
        }
        if (!candidates.contains(i)) candidates.add(i);

        return new LogRegister<>(u,i,value, candidates);
    }
}
