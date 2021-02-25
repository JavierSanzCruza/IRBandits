package es.uam.eps.ir.knnbandit.recommendation.ensembles;

import es.uam.eps.ir.knnbandit.utils.Pair;

public interface Ensemble<U, I>
{
    int getCurrentAlgorithm();
    String getAlgorithmName(int idx);
    Pair<Integer> getAlgorithmStats(int idx);
}
