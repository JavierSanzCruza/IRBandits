package es.uam.eps.ir.knnbandit.data.datasets.reader;

import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collection;
import java.util.HashSet;

public abstract class StreamDatasetReader<U,I>
{
    protected final String file;
    protected boolean finished;
    private BufferedReader br;
    protected final Parser<U> uParser;
    protected final Parser<I> iParser;
    protected final String separator;

    public StreamDatasetReader(String file, Parser<U> uParser, Parser<I> iParser, String separator)
    {
        this.file = file;
        this.uParser = uParser;
        this.iParser = iParser;
        this.separator = separator;
        this.finished = true;
    }

    public void initialize() throws IOException
    {
        if(br != null) br.close();
        br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
        this.finished = false;
    }

    public LogRegister<U,I> readRegister() throws IOException
    {
        LogRegister<U,I> register = null;
        if(finished)
        {
            String line = br.readLine();
            if (line == null)
            {
                br.close();
                br = null;
                finished = true;
            }
            else
            {
                register = processRegister(line);
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
            }
        }
        return register;
    }

    protected abstract LogRegister<U,I> processRegister(String line);

    public boolean hasEnded()
    {
        return this.finished;
    }

    public void close() throws IOException
    {
        if(this.br != null)
        {
            this.br.close();
            br = null;
            this.finished = true;
        }
    }
}
