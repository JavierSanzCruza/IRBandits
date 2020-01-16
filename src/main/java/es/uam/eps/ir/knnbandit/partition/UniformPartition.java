package es.uam.eps.ir.knnbandit.partition;

import org.jooq.lambda.tuple.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class UniformPartition implements Partition
{
    @Override
    public List<Integer> split(List<Tuple2<Integer, Integer>> trainingData, int numParts)
    {
        List<Integer> splitPoints = new ArrayList<>();
        int size = trainingData.size();
        for(int part = 1; part <= numParts; ++part)
        {
            int point = (size*part)/numParts;
            splitPoints.add(point);
        }

        return splitPoints;
    }

    @Override
    public int split(List<Tuple2<Integer, Integer>> trainingData, double percentage)
    {
        int size = trainingData.size();
        Double point = percentage*size;
        return point.intValue();
    }
}
