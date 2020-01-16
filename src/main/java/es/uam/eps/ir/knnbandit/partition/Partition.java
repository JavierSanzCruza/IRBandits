package es.uam.eps.ir.knnbandit.partition;

import org.jooq.lambda.tuple.Tuple2;

import java.util.List;

/**
 * Interface for partitioning training data.
 */
public interface Partition
{
    /**
     * Given a list of tuples, divides it in a given number of parts.
     * @param trainingData the training data.
     * @param numParts the number of parts.
     * @return a list containing the split points.
     */
    public List<Integer> split(List<Tuple2<Integer,Integer>> trainingData, int numParts);

    /**
     * Given a list of tuples, divides it in two parts given a percentage.
     * @param trainingData the training data.
     * @param percentage the percentage of training.
     * @return the split point.
     */
    public int split(List<Tuple2<Integer, Integer>> trainingData, double percentage);
}
