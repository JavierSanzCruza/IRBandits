package es.uam.eps.ir.knnbandit.io;

import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import org.ranksys.core.util.tuples.Tuple2id;

import java.io.*;

public class TextWriter implements WriterInterface
{
    /**
     * A writer, for printing the results into a file.
     */
    private BufferedWriter bw;

    @Override
    public void initialize(String filename) throws IOException
    {
        if(this.bw != null) throw new IOException("ERROR: there is a file currently open");
        this.bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)));
    }

    @Override
    public void initialize(OutputStream outputStream) throws IOException
    {
        if(this.bw != null) throw new IOException("ERROR: there is a file currently open");
        this.bw = new BufferedWriter(new OutputStreamWriter(outputStream));
    }

    @Override
    public void writeHeader() throws IOException
    {
        bw.write("numIter\tuidx\tiidx\ttime");
    }

    @Override
    public void writeLine(int numIter, int uidx, int iidx, long time) throws IOException
    {
        StringBuilder builder = new StringBuilder();
        builder.append("\n");
        builder.append(numIter);
        builder.append("\t");
        builder.append(uidx);
        builder.append("\t");
        builder.append(iidx);
        builder.append("\t");
        builder.append(time);
        bw.write(builder.toString());
    }

    @Override
    public void writeRanking(int numIter, FastRecommendation rec, long time) throws IOException
    {
        int uidx = rec.getUidx();
        for(Tuple2id iidx : rec.getIidxs())
        {
            this.writeLine(numIter, uidx, iidx.v1, time);
        }
    }

    @Override
    public void close() throws IOException
    {
        if (this.bw != null)
        {
            this.bw.close();
        }
        this.bw = null;
    }
}