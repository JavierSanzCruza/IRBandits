package es.uam.eps.ir.knnbandit.selector.algorithms.factorizer;

import es.uam.eps.ir.ranksys.mf.Factorizer;

public interface FactorizerSupplier<U,I>
{
    Factorizer<U,I> apply();
    String getName();
}
