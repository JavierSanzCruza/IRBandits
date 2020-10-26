package es.uam.eps.ir.knnbandit.selector.algorithms.factorizer;

import es.uam.eps.ir.knnbandit.selector.algorithms.bandit.BanditSupplier;
import es.uam.eps.ir.ranksys.mf.Factorizer;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;

public interface FactorizerConfigurator<U,I>
{
    List<FactorizerSupplier<U,I>> getFactorizers(JSONArray array);
    FactorizerSupplier<U,I> getFactorizer(JSONObject object);
}
