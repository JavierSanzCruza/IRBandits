package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;

public interface AlgorithmConfigurator<U,I>
{
    List<InteractiveRecommenderSupplier<U,I>> getAlgorithms(JSONArray array);
    InteractiveRecommenderSupplier<U,I> getAlgorithm(JSONObject object);
}
