package es.uam.eps.ir.knnbandit.selector.algorithms;

import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.selector.PMFBanditIdentifiers;
import es.uam.eps.ir.knnbandit.selector.algorithms.factorizer.*;
import es.uam.eps.ir.knnbandit.selector.algorithms.pmfbandit.EpsilonGreedyPMFBanditConfigurator;
import es.uam.eps.ir.knnbandit.selector.algorithms.pmfbandit.GeneralizedLinearUCBPMFBanditConfigurator;
import es.uam.eps.ir.knnbandit.selector.algorithms.pmfbandit.LinearUCBPMFBanditConfigurator;
import es.uam.eps.ir.knnbandit.selector.algorithms.pmfbandit.ThompsonSamplingPMFBanditConfigurator;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class PMFBanditConfigurator<U,I> extends AbstractAlgorithmConfigurator<U,I>
{
    private static final String LAMBDAP = "lambdaP";
    private static final String LAMBDAQ = "lambdaQ";
    private static final String STDEV = "stdev";
    private static final String NUMITER = "numIter";

    private static final String IGNOREUNKNOWN = "ignoreUnknown";
    private static final String VARIANT = "variant";
    private static final String K = "k";
    private static final String NAME = "name";
    private static final String PARAMS = "params";

    @Override
    public List<InteractiveRecommenderSupplier<U, I>> getAlgorithms(JSONArray array)
    {
        List<InteractiveRecommenderSupplier<U,I>> list = new ArrayList<>();
        int numConfigs = array.length();
        for(int i = 0; i < numConfigs; ++i)
        {
            JSONObject object = array.getJSONObject(i);
            boolean ignoreUnknown = true;
            if(object.has(IGNOREUNKNOWN))
            {
                ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
            }
            int k = object.getInt(K);
            double lambdaP = object.getDouble(LAMBDAP);
            double lambdaQ = object.getDouble(LAMBDAQ);
            double stdev = object.getDouble(STDEV);
            int numIter = object.getInt(NUMITER);

            // And, now, we obtain
            JSONObject variant = object.getJSONObject(VARIANT);
            String name = variant.getString(NAME);

            AlgorithmConfigurator<U,I> conf = this.selectInterPMFVariant(name, k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
            list.addAll(conf.getAlgorithms(variant.getJSONArray(PARAMS)));
        }
        return list;
    }

    @Override
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(JSONObject object)
    {
        boolean ignoreUnknown = true;
        if(object.has(IGNOREUNKNOWN))
        {
            ignoreUnknown = object.getBoolean(IGNOREUNKNOWN);
        }
        int k = object.getInt(K);
        double lambdaP = object.getDouble(LAMBDAP);
        double lambdaQ = object.getDouble(LAMBDAQ);
        double stdev = object.getDouble(STDEV);
        int numIter = object.getInt(NUMITER);

        // And, now, we obtain
        JSONObject variant = object.getJSONObject(VARIANT);
        String name = variant.getString(NAME);

        AlgorithmConfigurator<U,I> conf = this.selectInterPMFVariant(name, k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
        assert conf != null;
        return conf.getAlgorithm(variant.getJSONObject(PARAMS));
    }

    private AlgorithmConfigurator<U,I> selectInterPMFVariant(String name, int k, double lambdaP, double lambdaQ, double stdev, int numIter, boolean ignoreUnknown)
    {
        switch(name)
        {
            case PMFBanditIdentifiers.EGREEDY:
                return new EpsilonGreedyPMFBanditConfigurator<>(k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
            case PMFBanditIdentifiers.UCB:
                return new LinearUCBPMFBanditConfigurator<>(k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
            case PMFBanditIdentifiers.GENERALIZEDUCB:
                return new GeneralizedLinearUCBPMFBanditConfigurator<>(k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
            case PMFBanditIdentifiers.THOMPSON:
                return new ThompsonSamplingPMFBanditConfigurator<>(k, lambdaP, lambdaQ, stdev, numIter, ignoreUnknown);
            default:
                return null;
        }
    }

    protected FactorizerConfigurator<U,I> selectFactorizerConfigurator(String name)
    {
        switch(name)
        {
            case FactorizerIdentifiers.IMF:
                return new HKVFactorizerConfigurator<>();
            case FactorizerIdentifiers.FASTIMF:
                return new PZTFactorizerConfigurator<>();
            case FactorizerIdentifiers.PLSA:
                return new PLSAFactorizerConfigurator<>();
            default:
                return null;
        }
    }
}
