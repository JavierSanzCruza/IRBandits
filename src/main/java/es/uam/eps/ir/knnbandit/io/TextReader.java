package es.uam.eps.ir.knnbandit.io;

import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2id;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TextReader implements ReaderInterface
{
    /**
     * A writer, for printing the results into a file.
     */
    private BufferedReader br;

    String nextLine;

    @Override
    public void initialize(InputStream inputStream) throws IOException
    {
        if(this.br != null) throw new IOException("ERROR: there is a file currently open");
        this.br = new BufferedReader(new InputStreamReader(inputStream));
    }

    @Override
    public void initialize(String filename) throws IOException
    {
        if(this.br != null) throw new IOException("ERROR: there is a file currently open");
        this.br = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
    }

    @Override
    public List<String> readHeader() throws IOException
    {
        String line = br.readLine();
        String[] split = line.split("\t");
        return new ArrayList<>(Arrays.asList(split));
    }


    public Tuple3<Integer, FastRecommendation, Long> readIteration() throws IOException
    {
        if(nextLine == null)
        {
            nextLine = br.readLine();
            if(nextLine == null)
            {
                return null;
            }
        }

        String[] split = nextLine.split("\t");
        int currentIter = Parsers.ip.parse(split[0]);
        int uidx = Parsers.ip.parse(split[1]);
        int iidx = Parsers.ip.parse(split[2]);
        long time = Parsers.lp.parse(split[split.length-1]);
        IntList recs = new IntArrayList();
        recs.add(iidx);

        boolean stop = false;

        while(!stop)
        {
            nextLine = br.readLine();
            if(nextLine != null)
            {
                split = nextLine.split("\t");
                int iter = Parsers.ip.parse(split[0]);
                if(iter != currentIter) stop = true;
                else
                {
                    iidx = Parsers.ip.parse(split[2]);
                    recs.add(iidx);
                }
            }
            else
            {
                stop = true;
            }
        }

        List<Tuple2id> rec = new ArrayList<>();
        for(int i = 0; i < recs.size(); ++i)
        {
            rec.add(new Tuple2id(recs.getInt(i), (recs.size()-i+0.0)/(recs.size()+0.0)));
        }

        return new Tuple3<>(currentIter, new FastRecommendation(uidx, rec), time);


    }

    @Override
    public void close() throws IOException
    {
        if (this.br != null)
        {
            this.br.close();
        }
        this.br = null;
    }
}