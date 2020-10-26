package es.uam.eps.ir.knnbandit.selector.algorithms.factorizer;

import es.uam.eps.ir.knnbandit.recommendation.mf.PZTFactorizer;
import es.uam.eps.ir.ranksys.mf.Factorizer;
import org.json.JSONObject;

import java.util.function.DoubleUnaryOperator;

public class PZTFactorizerConfigurator<U,I> extends AbstractFactorizerConfigurator<U,I>
{
    private final static String ALPHA = "alpha";
    private final static String LAMBDA = "lambda";
    private final static String NUMITER = "numIter";
    private final static String USEZEROES = "useZeroes";

    @Override
    public FactorizerSupplier<U, I> getFactorizer(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        double lambda = object.getDouble(LAMBDA);
        int numIter = object.getInt(NUMITER);
        boolean useZeroes = object.getBoolean(USEZEROES);
        return new PZTFactorizerSupplier<>(alpha, lambda, numIter, useZeroes);
    }

    private static class PZTFactorizerSupplier<U,I> implements FactorizerSupplier<U,I>
    {
        private final double alpha;
        private final double lambda;
        private final int numIter;
        private final boolean useZeroes;

        public PZTFactorizerSupplier(double alpha, double lambda, int numIter, boolean useZeroes)
        {
            this.alpha = alpha;
            this.lambda = lambda;
            this.numIter = numIter;
            this.useZeroes = useZeroes;
        }

        @Override
        public Factorizer<U, I> apply()
        {
            DoubleUnaryOperator confidence = (double x) -> 1 + alpha * x;
            return new PZTFactorizer<>(lambda, confidence, numIter, useZeroes);
        }

        @Override
        public String getName()
        {
            return FactorizerIdentifiers.FASTIMF + "-" + alpha + "-" + lambda + "-" + numIter + "-" + (useZeroes ? "true" : "false");
        }
    }
}
