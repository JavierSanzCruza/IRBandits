package es.uam.eps.ir.knnbandit.selector.algorithms.factorizer;

import es.uam.eps.ir.ranksys.mf.Factorizer;
import es.uam.eps.ir.ranksys.mf.plsa.PLSAFactorizer;
import org.json.JSONObject;

public class PLSAFactorizerConfigurator<U,I> extends AbstractFactorizerConfigurator<U,I>
{
    private final static String NUMITER = "numIter";

    @Override
    public FactorizerSupplier<U, I> getFactorizer(JSONObject object)
    {
        int numIter = object.getInt(NUMITER);
        return new PLSAFactorizerSupplier<>(numIter);
    }

    private static class PLSAFactorizerSupplier<U,I> implements FactorizerSupplier<U,I>
    {
        private final int numIter;

        public PLSAFactorizerSupplier(int numIter)
        {
            this.numIter = numIter;
        }

        @Override
        public Factorizer<U, I> apply()
        {
            return new PLSAFactorizer<>(numIter);
        }

        @Override
        public String getName()
        {
            return FactorizerIdentifiers.PLSA + numIter;
        }
    }
}
