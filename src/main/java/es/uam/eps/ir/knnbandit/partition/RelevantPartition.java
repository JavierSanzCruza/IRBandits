package es.uam.eps.ir.knnbandit.partition;

import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import org.jooq.lambda.tuple.Tuple2;
import org.ranksys.fast.preference.FastPointWisePreferenceData;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.DoublePredicate;

public class RelevantPartition implements Partition
{
    private final FastPointWisePreferenceData<?, ?> prefData;
    private final DoublePredicate relevanceChecker;

    /**
     * Constructor.
     *
     * @param prefData         Preference data.
     * @param relevanceChecker Function that determines if a value indicates a relevant (user,item) pair or not.
     */
    public RelevantPartition(FastPointWisePreferenceData<?, ?> prefData, DoublePredicate relevanceChecker)
    {
        this.prefData = prefData;
        this.relevanceChecker = relevanceChecker;
    }

    @Override
    public List<Integer> split(List<Tuple2<Integer, Integer>> trainingData, int numParts)
    {
        List<Integer> splitPoints = new ArrayList<>();

        // Count the total number of relevant pairs
        int numRel = trainingData.stream().mapToInt(tuple ->
                                                    {
                                                        Optional<? extends IdxPref> optional = prefData.getPreference(tuple.v1, tuple.v2);
                                                        return (optional.isPresent() && relevanceChecker.test(optional.get().v2)) ? 1 : 0;
                                                    }).sum();

        int nextPoint = numRel / numParts;
        int counter = 1;
        int i = 0;
        int j = 0;
        for (Tuple2<Integer, Integer> tuple : trainingData)
        {
            Optional<? extends IdxPref> optional = prefData.getPreference(tuple.v1, tuple.v2);
            if (optional.isPresent() && relevanceChecker.test(optional.get().v2))
            {
                i++;
            }

            ++j;
            if (i == nextPoint && counter < numParts)
            {
                splitPoints.add(j);
                counter++;
                nextPoint = numRel * (counter) / numParts;
            }
        }
        splitPoints.add(j);
        return splitPoints;
    }

    @Override
    public int split(List<Tuple2<Integer, Integer>> trainingData, double percentage)
    {
        int size = trainingData.size();

        // Count the total number of relevant pairs
        int numRel = trainingData.stream().mapToInt(tuple ->
                                                    {
                                                        Optional<? extends IdxPref> optional = prefData.getPreference(tuple.v1, tuple.v2);
                                                        return (optional.isPresent() && relevanceChecker.test(optional.get().v2)) ? 1 : 0;
                                                    }).sum();

        Double point = percentage * numRel;
        int splitPoint = point.intValue();

        int count = 0;
        int j = 0;
        for (Tuple2<Integer, Integer> tuple : trainingData)
        {
            Optional<? extends IdxPref> optional = prefData.getPreference(tuple.v1, tuple.v2);
            if (optional.isPresent() && relevanceChecker.test(optional.get().v2))
            {
                count++;
            }
            ++j;
            if (count == splitPoint)
            {
                return j;
            }
        }

        return point.intValue();
    }
}
