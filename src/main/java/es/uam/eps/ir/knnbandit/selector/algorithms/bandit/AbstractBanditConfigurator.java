package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractBanditConfigurator<U,I> implements BanditConfigurator
{
    @Override
    public List<BanditSupplier> getBandits(JSONArray array)
    {
        List<BanditSupplier> list = new ArrayList<>();
        int numConfigs = array.length();
        for(int i = 0; i < numConfigs; ++i)
        {
            JSONObject obj = array.getJSONObject(i);
            list.add(this.getBandit(obj));
        }
        return list;
    }
}
