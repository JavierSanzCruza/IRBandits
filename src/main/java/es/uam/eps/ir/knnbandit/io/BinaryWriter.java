package es.uam.eps.ir.knnbandit.io;

import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import org.ranksys.core.util.tuples.Tuple2id;

import java.io.*;

public class BinaryWriter implements WriterInterface
{
    private DataOutputStream output = null;

    @Override
    public void initialize(String filename) throws IOException
    {
        if(this.output != null) throw new IOException("ERROR: there is a file currently open");
        this.output = new DataOutputStream(new FileOutputStream(filename));
    }

    @Override
    public void initialize(OutputStream outputStream) throws IOException
    {
        if(this.output != null) throw new IOException("ERROR: there is a file currently open");
        this.output = new DataOutputStream(outputStream);
    }

    @Override
    public void writeHeader()
    {
    }

    @Override
    public void writeLine(int numIter, int uidx, int iidx, long time) throws IOException
    {
        output.writeInt(numIter);
        output.writeInt(uidx);
        output.writeInt(1);
        output.writeInt(iidx);
        output.writeLong(time);
    }

    @Override
    public void writeRanking(int numIter, FastRecommendation rec, long time) throws IOException
    {
        output.writeInt(numIter);
        output.writeInt(rec.getUidx());
        output.writeInt(rec.getIidxs().size());
        for(Tuple2id iidx : rec.getIidxs())
        {
            output.writeInt(iidx.v1);
        }
        output.writeLong(time);
    }

    @Override
    public void close() throws IOException
    {
        output.close();
        output = null;
    }
}