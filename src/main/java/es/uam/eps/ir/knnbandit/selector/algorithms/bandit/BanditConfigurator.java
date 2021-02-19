package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;

public interface BanditConfigurator<U,I>
{
    List<BanditSupplier<U,I>> getBandits(JSONArray array);
    BanditSupplier<U,I> getBandit(JSONObject object);
}
