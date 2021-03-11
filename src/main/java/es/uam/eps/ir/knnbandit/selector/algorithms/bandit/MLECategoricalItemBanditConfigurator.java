package es.uam.eps.ir.knnbandit.selector.algorithms.bandit;

import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.AbstractMultiArmedBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.PopularityMLE;
import org.json.JSONObject;

/**
 * Class for configuring a bandit that selects an item proportionally to its popularity.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 *
 * @see es.uam.eps.ir.knnbandit.recommendation.bandits.algorithms.PopularityMLE
 */
public class MLECategoricalItemBanditConfigurator extends AbstractBanditConfigurator
{
    /**
     * Identifier of the initial value of the alpha parameter of the Beta distribution.
     */
    private final static String ALPHA = "alpha";

    @Override
    public BanditSupplier getBandit(JSONObject object)
    {
        double alpha = object.getDouble(ALPHA);
        return new MLECategoricalItemBanditSupplier(alpha);
    }

    /**
     * Class for configuring a bandit that recommends items proportionally to their popularity.
     */
    private static class MLECategoricalItemBanditSupplier implements BanditSupplier
    {
        /**
         * The initial alpha value for each item.
         */
        private final double alpha;

        /**
         * Constructor.
         * @param alpha the initial alpha value for each item.
         */
        public MLECategoricalItemBanditSupplier(double alpha)
        {
            this.alpha = alpha;
        }

        @Override
        public AbstractMultiArmedBandit apply(int numItems)
        {
            return new PopularityMLE(numItems, alpha);
        }

        @Override
        public String getName()
        {
            return MultiArmedBanditIdentifiers.MLEPOP + "-" + alpha;
        }
    }
}
