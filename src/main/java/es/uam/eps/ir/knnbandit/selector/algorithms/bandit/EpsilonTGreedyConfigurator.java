package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.item.*;
import org.jooq.lambda.tuple.Tuple2;
import org.json.JSONObject;

public class EpsilonTGreedyConfigurator<U,I> extends AbstractBanditConfigurator<U,I>
{

    private final static String UPDATEFUNC = "updateFunc";
    private final static String FUNCTION = "function";
    private final static String ALPHA = "alpha";

    @Override
    public BanditSupplier<U, I> getBandit(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        JSONObject function = object.getJSONObject(UPDATEFUNC);
        Tuple2<String, EpsilonGreedyUpdateFunction> updFunction = EpsilonGreedyConfigurator.getUpdateFunction(function);
        if(updFunction == null) return null;

        String functionName = updFunction.v1;
        EpsilonGreedyUpdateFunction updateFunction = updFunction.v2;
        return new EpsilonTGreedyBanditSupplier<>(alpha, functionName, updateFunction);
    }

    private static class EpsilonTGreedyBanditSupplier<U,I> implements BanditSupplier<U,I>
    {
        private final double alpha;
        private final EpsilonGreedyUpdateFunction updateFunction;
        private final String functionName;

        public EpsilonTGreedyBanditSupplier(double alpha, String functionName, EpsilonGreedyUpdateFunction updateFunction)
        {
            this.alpha = alpha;
            this.functionName = functionName;
            this.updateFunction = updateFunction;
        }

        @Override
        public ItemBandit<U, I> apply(int numItems)
        {
            return new EpsilonTGreedyItemBandit<>(alpha, numItems, updateFunction);
        }

        @Override
        public String getName()
        {
            return ItemBanditIdentifiers.ETGREEDY + "-" + alpha + "-" + functionName;
        }
    }
}
