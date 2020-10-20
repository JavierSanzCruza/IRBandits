package es.uam.eps.ir.knnbandit.data.datasets.reader;

import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

public abstract class StreamCandidateSelectionDatasetReader<U,I> extends StreamDatasetReader<U,I>
{
    public StreamCandidateSelectionDatasetReader(String file, Parser<U> uParser, Parser<I> iParser, String separator)
    {
        super(file, uParser, iParser, separator);
    }

    protected LogRegister<U,I> processRegister(String line)
    {
        LogRegister<U,I> register = processRegister(line);
        // Process the register:
        String[] split = line.split(separator);
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
