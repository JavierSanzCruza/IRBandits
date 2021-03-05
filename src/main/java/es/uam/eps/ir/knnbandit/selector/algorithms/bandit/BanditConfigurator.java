package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;

public interface BanditConfigurator
{
    List<BanditSupplier> getBandits(JSONArray array);
    BanditSupplier getBandit(JSONObject object);
}
