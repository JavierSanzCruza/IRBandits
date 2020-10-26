package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.item.EpsilonGreedyItemBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.item.EpsilonGreedyUpdateFunction;
import es.uam.eps.ir.knnbandit.recommendation.bandits.item.EpsilonGreedyUpdateFunctions;
import es.uam.eps.ir.knnbandit.recommendation.bandits.item.ItemBandit;
import es.uam.eps.ir.knnbandit.selector.EpsilonGreedyUpdateFunctionIdentifiers;
import org.jooq.lambda.tuple.Tuple2;
import org.json.JSONObject;

public class EpsilonGreedyConfigurator<U,I> extends AbstractBanditConfigurator<U,I>
{

    private final static String EPSILON = "epsilon";
    private final static String UPDATEFUNC = "updateFunc";
    private final static String FUNCTION = "function";
    private final static String ALPHA = "alpha";

    @Override
    public BanditSupplier<U, I> getBandit(JSONObject object)
    {
        double epsilon = object.getDouble(EPSILON);
        JSONObject function = object.getJSONObject(UPDATEFUNC);
        Tuple2<String, EpsilonGreedyUpdateFunction> updFunction = getUpdateFunction(function);
        if(updFunction == null) return null;

        String functionName = updFunction.v1;
        EpsilonGreedyUpdateFunction updateFunction = updFunction.v2;
        return new EpsilonGreedyBanditSupplier<>(epsilon, functionName, updateFunction);
    }

    /**
     * Obtains a function to update an Epsilon-greedy algorithm.
     *
     * @return the update function if everything is OK, null otherwise.
     */
    static Tuple2<String, EpsilonGreedyUpdateFunction> getUpdateFunction(JSONObject object)
    {
        String name = object.getString(FUNCTION);
        switch (name)
        {
            case EpsilonGreedyUpdateFunctionIdentifiers.STATIONARY:
                return new Tuple2<>(name, EpsilonGreedyUpdateFunctions.stationary());
            case EpsilonGreedyUpdateFunctionIdentifiers.NONSTATIONARY:
                double alpha = object.getDouble(ALPHA);
                return new Tuple2<>(name + "-" + alpha, EpsilonGreedyUpdateFunctions.nonStationary(alpha));
            case EpsilonGreedyUpdateFunctionIdentifiers.USEALL:
                return new Tuple2<>(name, EpsilonGreedyUpdateFunctions.useall());
            case EpsilonGreedyUpdateFunctionIdentifiers.COUNT:
                return new Tuple2<>(name, EpsilonGreedyUpdateFunctions.count());
            default:
                return null;
        }
    }

    private static class EpsilonGreedyBanditSupplier<U,I> implements BanditSupplier<U,I>
    {
        private final double epsilon;
        private final EpsilonGreedyUpdateFunction updateFunction;
        private final String functionName;

        public EpsilonGreedyBanditSupplier(double epsilon, String functionName, EpsilonGreedyUpdateFunction updateFunction)
        {
            this.epsilon = epsilon;
            this.functionName = functionName;
            this.updateFunction = updateFunction;
        }

        @Override
        public ItemBandit<U, I> apply(int numItems)
        {
            return new EpsilonGreedyItemBandit<>(epsilon, numItems, updateFunction);
        }

        @Override
        public String getName()
        {
            return ItemBanditIdentifiers.EGREEDY + "-" + epsilon + "-" + functionName;
        }
    }
}
