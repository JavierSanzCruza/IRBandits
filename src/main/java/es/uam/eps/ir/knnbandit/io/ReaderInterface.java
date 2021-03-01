package es.uam.eps.ir.knnbandit.io;

import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import org.jooq.lambda.tuple.Tuple3;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public interface ReaderInterface
{
    void initialize(String filename) throws IOException;
    void initialize(InputStream inputStream) throws IOException;
    Tuple3<Integer, FastRecommendation, Long> readIteration() throws IOException;

    void close() throws IOException;

    List<String> readHeader() throws IOException;

    List<Pair<Integer>> readFile(String filename) throws IOException;
    List<Pair<Integer>> readFile(InputStream stream) throws IOException;

}
