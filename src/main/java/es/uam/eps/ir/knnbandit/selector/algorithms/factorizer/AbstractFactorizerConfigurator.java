package es.uam.eps.ir.knnbandit.selector.algorithms.factorizer;

import es.uam.eps.ir.knnbandit.selector.algorithms.bandit.BanditSupplier;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractFactorizerConfigurator<U,I> implements FactorizerConfigurator<U,I>
{
    @Override
    public List<FactorizerSupplier<U, I>> getFactorizers(JSONArray array)
    {
        List<FactorizerSupplier<U,I>> list = new ArrayList<>();
        int numConfigs = array.length();
        for(int i = 0; i < numConfigs; ++i)
        {
            JSONObject obj = array.getJSONObject(i);
            list.add(this.getFactorizer(obj));
        }
        return list;
    }
}
