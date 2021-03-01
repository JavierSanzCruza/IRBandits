package es.uam.eps.ir.knnbandit.io;

import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2id;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class BinaryReader implements ReaderInterface
{

    private DataInputStream inputStream = null;

    @Override
    public void initialize(String filename) throws IOException
    {
        if(this.inputStream != null) throw new IOException("ERROR: there is a file currently open");
        this.inputStream = new DataInputStream(new FileInputStream(filename));
    }

    @Override
    public void initialize(InputStream inputStream) throws IOException
    {
        if(this.inputStream != null) throw new IOException("ERROR: there is a file currently open");
        this.inputStream = new DataInputStream(inputStream);
    }

    @Override
    public Tuple3<Integer, FastRecommendation, Long> readIteration() throws IOException
    {
        int numIter;
        int uidx;
        int numItems;
        if(inputStream.available() >= 3*Integer.BYTES )
        {
            numIter = inputStream.readInt();
            uidx = inputStream.readInt();
            numItems = inputStream.readInt();
        }
        else
        {
            return null;
        }

        List<Tuple2id> list = new ArrayList<>();
        long time;
        if(inputStream.available() >= 3*Integer.BYTES + Long.BYTES)
        {
            for(int i = 0; i < numItems; ++i)
            {
                list.add(new Tuple2id(inputStream.readInt(), (numItems-i+0.0)/(numItems)));
            }
             time = inputStream.readLong();
        }
        else
        {
            return null;
        }

        return new Tuple3<>(numIter, new FastRecommendation(uidx, list), time);
    }

    @Override
    public void close() throws IOException
    {
        inputStream.close();
        inputStream = null;
    }

    @Override
    public List<String> readHeader()
    {
        return new ArrayList<>();
    }
}