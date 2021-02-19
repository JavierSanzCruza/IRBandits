package es.uam.eps.ir.knnbandit.selector.algorithms.factorizer;

import es.uam.eps.ir.ranksys.mf.Factorizer;
import es.uam.eps.ir.ranksys.mf.als.HKVFactorizer;
import org.json.JSONObject;

import java.util.function.DoubleUnaryOperator;

public class HKVFactorizerConfigurator<U,I> extends AbstractFactorizerConfigurator<U,I>
{
    private final static String ALPHA = "alpha";
    private final static String LAMBDA = "lambda";
    private final static String NUMITER = "numIter";

    @Override
    public FactorizerSupplier<U, I> getFactorizer(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        double lambda = object.getDouble(LAMBDA);
        int numIter = object.getInt(NUMITER);
        return new HKVFactorizerSupplier<>(alpha, lambda, numIter);
    }

    private static class HKVFactorizerSupplier<U,I> implements FactorizerSupplier<U,I>
    {
        private final double alpha;
        private final double lambda;
        private final int numIter;

        public HKVFactorizerSupplier(double alpha, double lambda, int numIter)
        {
            this.alpha = alpha;
            this.lambda = lambda;
            this.numIter = numIter;
        }

        @Override
        public Factorizer<U, I> apply()
        {
            DoubleUnaryOperator confidence = (double x) -> 1 + alpha * x;
            return new HKVFactorizer<>(lambda, confidence, numIter);
        }

        @Override
        public String getName()
        {
            return FactorizerIdentifiers.IMF + "-" + alpha + "-" + lambda + "-" + numIter;
        }
    }
}
