package es.uam.eps.ir.knnbandit.io;

import es.uam.eps.ir.ranksys.fast.FastRecommendation;

import java.io.IOException;
import java.io.OutputStream;

/**
 * Interface for writing recommendation registers.
 */
public interface WriterInterface
{
    /**
     * Initializes the writer.
     * @param filename the name of the file in which to store the recommendations.
     */
    void initialize(String filename) throws IOException;

    /**
     * Initializes the writer.
     * @param stream an output stream.
     */
    void initialize(OutputStream stream) throws IOException;

    /**
     * Writes the header of the file (if any)
     * @throws IOException if something fails while writing.
     */
    void writeHeader() throws IOException;

    /**
     * Writes a line into the file.
     * @param numIter the iteration number.
     * @param uidx the user identifier.
     * @param iidx the item identifier.
     * @param time the execution time.
     * @throws IOException if something fails while writing.
     */
    void writeLine(int numIter, int uidx, int iidx, long time) throws IOException;

    /**
     * Writes a ranking into the file.
     * @param numIter the iteration number.
     * @param rec the recommendation ranking.
     * @param time the execution time.
     * @throws IOException if something fails while writing.
     */
    void writeRanking(int numIter, FastRecommendation rec, long time) throws IOException;

    /**
     * Closes the writer.
     *
     * @throws IOException if something fails while closing.
     */
    void close() throws IOException;
}
