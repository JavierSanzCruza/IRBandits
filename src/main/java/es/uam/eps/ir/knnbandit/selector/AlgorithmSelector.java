/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit.selector;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.userknowledge.fast.SimpleFastUserKnowledgePreferenceData;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommenderSupplier;
import es.uam.eps.ir.knnbandit.recommendation.KnowledgeDataUse;
import es.uam.eps.ir.knnbandit.recommendation.bandits.ItemBanditRecommender;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunctions;
import es.uam.eps.ir.knnbandit.recommendation.bandits.item.*;
import es.uam.eps.ir.knnbandit.recommendation.basic.AvgRecommender;
import es.uam.eps.ir.knnbandit.recommendation.basic.PopularityRecommender;
import es.uam.eps.ir.knnbandit.recommendation.basic.RandomRecommender;
import es.uam.eps.ir.knnbandit.recommendation.clusters.club.CLUBComplete;
import es.uam.eps.ir.knnbandit.recommendation.clusters.club.CLUBERdos;
import es.uam.eps.ir.knnbandit.recommendation.clusters.cofiba.COFIBAErdos;
import es.uam.eps.ir.knnbandit.recommendation.knn.item.InteractiveItemBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.UpdateableSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.VectorCosineSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.similarities.stochastic.BetaStochasticSimilarity;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.CollaborativeGreedy;
import es.uam.eps.ir.knnbandit.recommendation.knn.user.InteractiveUserBasedKNN;
import es.uam.eps.ir.knnbandit.recommendation.mf.InteractiveMF;
import es.uam.eps.ir.knnbandit.recommendation.mf.PZTFactorizer;
import es.uam.eps.ir.knnbandit.recommendation.mf.icf.EpsilonGreedyInteractivePMFRecommender;
import es.uam.eps.ir.knnbandit.recommendation.mf.icf.GeneralizedLinearUCBPMFRecommenderInteractive;
import es.uam.eps.ir.knnbandit.recommendation.mf.icf.LinearUCBPMFRecommenderInteractive;
import es.uam.eps.ir.knnbandit.recommendation.mf.icf.ThompsonSamplingInteractivePMFRecommender;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.ParticleThompsonSamplingMF;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles.PTSMFParticleFactory;
import es.uam.eps.ir.knnbandit.recommendation.mf.ptsmf.particles.PTSMParticleFactories;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import es.uam.eps.ir.ranksys.mf.Factorizer;
import es.uam.eps.ir.ranksys.mf.als.HKVFactorizer;
import es.uam.eps.ir.ranksys.mf.plsa.PLSAFactorizer;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.function.DoubleUnaryOperator;

/**
 * Class for selecting the interactive recommendation algorithms to apply in an experiments. The class encapsulates the set
 * of algorithms, as well as some experiment settings.
 *
 * @param <U> User type.
 * @param <I> Item type.
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class AlgorithmSelector<U, I>
{
    /**
     * A map of recommenders to apply.
     */
    private final Map<String, InteractiveRecommenderSupplier<U, I>> recs;
    /**
     * A cursor for reading the line configuration.
     */
    private int cursor;
    /**
     * Indicates if the selector has been previously configured.
     */
    private boolean configured;
    /**
     * User index.
     */
    private FastUpdateableUserIndex<U> uIndex;
    /**
     * Item index.
     */
    private FastUpdateableItemIndex<I> iIndex;
    /**
     * Preference data.
     */
    private SimpleFastPreferenceData<U, I> prefData;
    /**
     * Information about previous knowledge of the user
     */
    private SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData;
    /**
     *
     */
    private KnowledgeDataUse dataUse;
    /**
     * True if contact recommendation algorithms must be configured, false otherwise.
     */
    private boolean contactRec;
    /**
     * True if reciprocal links should not be recommended, false otherwise.
     */
    private boolean notReciprocal;
    /**
     * Relevance threshold.
     */
    private double threshold;

    /**
     * Constructor.
     */
    public AlgorithmSelector()
    {
        recs = new HashMap<>();
        notReciprocal = false;
    }

    /**
     * Resets the selection.
     */
    public void reset()
    {
        this.recs.clear();
        this.uIndex = null;
        this.iIndex = null;
        this.prefData = null;
        this.contactRec = false;
        this.notReciprocal = false;
        this.configured = false;
    }

    /**
     * Configures the experiment for traditional item-to-people recommendation, from a full cold start perspective.
     *
     * @param threshold Relevance threshold
     */
    public void configure(double threshold)
    {
        this.notReciprocal = false;
        this.contactRec = false;
        this.threshold = threshold;
        this.dataUse = KnowledgeDataUse.ALL;
        this.knowledgeData = null;
        this.configured = true;
    }

    /**
     * Configures the experiment for traditional item-to-people recommendation, from a full cold start perspective.
     *
     * @param uIndex    User index.
     * @param iIndex    Item index.
     * @param prefData  Preference data.
     * @param threshold Relevance threshold
     */
    public void configure(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, double threshold, SimpleFastUserKnowledgePreferenceData<U, I> knowledgeData, KnowledgeDataUse dataUse)
    {
        this.notReciprocal = false;
        this.contactRec = false;
        this.threshold = threshold;
        this.dataUse = dataUse;
        this.configured = true;
    }

    /**
     * Given a string containing its configuration, obtains an interactive recommendation algorithm.
     *
     * @param algorithm The string containing the configuration of the algorithm.
     * @return an interactive recommender.
     * @throws es.uam.eps.ir.knnbandit.selector.UnconfiguredException if the experiment is not configured.
     */
    public InteractiveRecommenderSupplier<U, I> getAlgorithm(String algorithm) throws UnconfiguredException
    {
        if (!this.configured)
        {
            throw new UnconfiguredException("The experiment is not configured");
        }
        cursor = 0;
        if (!algorithm.startsWith("//"))
        {
            String[] split = algorithm.split("-");
            List<String> fullAlgorithm = new ArrayList<>(Arrays.asList(split));
            boolean hasRating;
            switch (fullAlgorithm.get(0))
            {
                case AlgorithmIdentifiers.RANDOM: // Random recommendation.
                {
                    return RandomRecommender::new;
                }
                case AlgorithmIdentifiers.AVG: // Average rating recommendation.
                {
                    cursor++;
                    if (fullAlgorithm.size() == cursor)
                    {
                        hasRating = false;
                    }
                    else
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                    }

                    return (userIndex, itemIndex) -> new AvgRecommender<>(userIndex, itemIndex, hasRating);
                }
                case AlgorithmIdentifiers.POP: // Popularity recommendation.
                {
                    cursor++;
                    return (userIndex, itemIndex) -> new PopularityRecommender<>(userIndex, itemIndex, threshold);
                }
                case AlgorithmIdentifiers.ITEMBANDIT: // Non-personalized bandits.
                {
                    cursor++;
                    BanditSupplier<U, I> itemBandit = this.getItemBandit(fullAlgorithm.subList(1, split.length));
                    if (itemBandit == null)
                    {
                        break;
                    }
                    ValueFunction valFunc = ValueFunctions.identity();

                    if (fullAlgorithm.size() == cursor)
                    {
                        hasRating = false;
                    }
                    else
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        cursor++;
                    }

                    return (userIndex, itemIndex) ->
                    {
                        ItemBandit<U,I> bandit = itemBandit.apply(itemIndex.numItems());
                        return new ItemBanditRecommender<>(userIndex, itemIndex, hasRating, bandit, valFunc);
                    };
                }
                case AlgorithmIdentifiers.USERBASEDKNN: // User-based kNN.
                {
                    cursor++;
                    int k = Parsers.ip.parse(fullAlgorithm.get(cursor));
                    cursor++;

                    UpdateableSimilarity sim = new VectorCosineSimilarity(prefData.numUsers());
                    boolean ignoreZeroes;
                    if (fullAlgorithm.size() == cursor)
                    {
                        hasRating = true;
                        ignoreZeroes = true;
                    }
                    else if (fullAlgorithm.size() == (cursor + 1))
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        ignoreZeroes = true;
                        cursor++;
                    }
                    else
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        ignoreZeroes = fullAlgorithm.get(cursor + 1).equalsIgnoreCase("ignore");
                        cursor += 2;
                    }

                    return (uIndex, iIndex) -> new InteractiveUserBasedKNN<>(uIndex, iIndex, hasRating, ignoreZeroes, k, sim);
                }
                case AlgorithmIdentifiers.ITEMBASEDKNN: // User-based kNN.
                {
                    cursor++;
                    int itemK = Parsers.ip.parse(fullAlgorithm.get(cursor));
                    int userK = 0;
                    cursor++;

                    UpdateableSimilarity sim = new VectorCosineSimilarity(prefData.numItems());
                    boolean ignoreZeroes;
                    if (fullAlgorithm.size() == cursor)
                    {
                        hasRating = true;
                        ignoreZeroes = true;
                    }
                    else if (fullAlgorithm.size() == (cursor + 1))
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        ignoreZeroes = true;
                        cursor++;
                    }
                    else
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        ignoreZeroes = fullAlgorithm.get(cursor + 1).equalsIgnoreCase("ignore");
                        cursor += 2;
                    }

                    return (uIndex, iIndex) -> new InteractiveItemBasedKNN<>(uIndex, iIndex, hasRating, ignoreZeroes, userK, itemK, sim);
                }
                case AlgorithmIdentifiers.BANDITKNN:
                {
                    cursor++;
                    String user = fullAlgorithm.get(cursor);
                    cursor++;
                    switch (user)
                    {
                        case KNNBanditIdentifiers.USER:
                        {
                            int k = Parsers.ip.parse(fullAlgorithm.get(cursor));
                            cursor++;
                            double alpha = Parsers.dp.parse(fullAlgorithm.get(cursor));
                            cursor++;
                            double beta = Parsers.dp.parse(fullAlgorithm.get(cursor));
                            cursor++;
                            UpdateableSimilarity sim = new BetaStochasticSimilarity(prefData.numUsers(), alpha, beta);

                            boolean ignoreZeroes;
                            if (fullAlgorithm.size() == cursor)
                            {
                                hasRating = true;
                                ignoreZeroes = true;
                            }
                            else if (fullAlgorithm.size() == (cursor + 1))
                            {
                                hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                                ignoreZeroes = true;
                                cursor++;
                            }
                            else
                            {
                                hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                                ignoreZeroes = fullAlgorithm.get(cursor + 1).equalsIgnoreCase("ignore");
                                cursor += 2;
                            }

                            return (uIndex, iIndex) -> new InteractiveUserBasedKNN<>(uIndex, iIndex, hasRating, ignoreZeroes, k, sim);
                        }
                        case KNNBanditIdentifiers.ITEM:
                        {
                            int userK = Parsers.ip.parse(fullAlgorithm.get(cursor));
                            cursor++;
                            int itemK = Parsers.ip.parse(fullAlgorithm.get(cursor));
                            cursor++;
                            double alpha = Parsers.dp.parse(fullAlgorithm.get(cursor));
                            cursor++;
                            double beta = Parsers.dp.parse(fullAlgorithm.get(cursor));
                            UpdateableSimilarity sim = new BetaStochasticSimilarity(prefData.numItems(), alpha, beta);
                            cursor++;

                            boolean ignoreZeroes;
                            if (fullAlgorithm.size() == cursor)
                            {
                                hasRating = true;
                                ignoreZeroes = true;
                            }
                            else if (fullAlgorithm.size() == (cursor + 1))
                            {
                                hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                                ignoreZeroes = true;
                                cursor++;
                            }
                            else
                            {
                                hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                                ignoreZeroes = fullAlgorithm.get(cursor + 1).equalsIgnoreCase("ignore");
                                cursor += 2;
                            }

                            return (uIndex, iIndex) -> new InteractiveItemBasedKNN<>(uIndex, iIndex, hasRating, ignoreZeroes, userK, itemK, sim);
                        }
                        default:
                        {
                            break;
                        }
                    }
                }
                case AlgorithmIdentifiers.MF:
                {
                    cursor++;
                    int k = Parsers.ip.parse(fullAlgorithm.get(cursor));
                    cursor++;
                    Factorizer<U, I> factorizer = this.getFactorizer(fullAlgorithm.subList(cursor, split.length));
                    if (factorizer == null)
                    {
                        break;
                    }

                    if (fullAlgorithm.size() == cursor)
                    {
                        hasRating = true;
                    }
                    else
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        cursor++;
                    }

                    return (uIndex, iIndex) -> new InteractiveMF<>(uIndex, iIndex, hasRating, k, factorizer);
                }
                case AlgorithmIdentifiers.PMFBANDIT:
                {
                    cursor++;
                    int k = Parsers.ip.parse(fullAlgorithm.get(cursor));
                    double lambdaP = Parsers.dp.parse(fullAlgorithm.get(cursor + 1));
                    double lambdaQ = Parsers.dp.parse(fullAlgorithm.get(cursor + 2));
                    double stdev = Parsers.dp.parse(fullAlgorithm.get(cursor + 3));
                    int numIter = Parsers.ip.parse(fullAlgorithm.get(cursor + 4));

                    cursor += 5;
                    return this.getPMFBanditAlgorithm(fullAlgorithm.subList(cursor, split.length), k, lambdaP, lambdaQ, stdev, numIter);
                }
                case AlgorithmIdentifiers.PTS:
                {
                    cursor++;
                    int k = Parsers.ip.parse(fullAlgorithm.get(cursor));
                    int numP = Parsers.ip.parse(fullAlgorithm.get(cursor + 1));
                    double sigmaP = Parsers.dp.parse(fullAlgorithm.get(cursor + 2));
                    double sigmaQ = Parsers.dp.parse(fullAlgorithm.get(cursor + 3));
                    double stdev = Parsers.dp.parse(fullAlgorithm.get(cursor + 4));
                    PTSMFParticleFactory<U, I> factory = PTSMParticleFactories.normalFactory(k, stdev, sigmaP, sigmaQ);
                    cursor += 5;

                    if (fullAlgorithm.size() == cursor)
                    {
                        hasRating = true;
                    }
                    else
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        cursor++;
                    }

                    return (uIndex, iIndex) -> new ParticleThompsonSamplingMF<>(uIndex, iIndex, hasRating, numP, factory);
                }
                case AlgorithmIdentifiers.BAYESIANPTS:
                {
                    cursor++;
                    int k = Parsers.ip.parse(fullAlgorithm.get(cursor));
                    int numP = Parsers.ip.parse(fullAlgorithm.get(cursor + 1));
                    double sigmaQ = Parsers.dp.parse(fullAlgorithm.get(cursor + 3));
                    double alpha = Parsers.dp.parse(fullAlgorithm.get(cursor + 4));
                    double beta = Parsers.dp.parse(fullAlgorithm.get(cursor + 5));
                    double stdev = Parsers.dp.parse(fullAlgorithm.get(cursor + 6));
                    PTSMFParticleFactory<U, I> factory = PTSMParticleFactories.bayesianFactory(k, stdev, sigmaQ, alpha, beta);
                    cursor += 7;

                    if (fullAlgorithm.size() == cursor)
                    {
                        hasRating = true;
                    }
                    else
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        cursor++;
                    }

                    return (uIndex, iIndex) -> new ParticleThompsonSamplingMF<>(uIndex, iIndex, hasRating, numP, factory);
                }
                case AlgorithmIdentifiers.COLLABGREEDY:
                {
                    cursor++;
                    double threshold = Parsers.dp.parse(fullAlgorithm.get(cursor));
                    double alpha = Parsers.dp.parse(fullAlgorithm.get(cursor+1));
                    cursor += 2;

                    if (fullAlgorithm.size() == cursor)
                    {
                        hasRating = true;
                    }
                    else
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        cursor++;
                    }

                    return (uIndex, iIndex) -> new CollaborativeGreedy<>(uIndex, iIndex, hasRating, threshold, alpha);
                }
                case AlgorithmIdentifiers.CLUB:
                {
                    cursor++;
                    double alpha1 = Parsers.dp.parse(fullAlgorithm.get(cursor));
                    double alpha2 = Parsers.dp.parse(fullAlgorithm.get(cursor));
                    cursor+=2;

                    if(fullAlgorithm.size() == cursor)
                    {
                        hasRating = true;
                    }
                    else
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        cursor++;
                    }

                    return (uIndex, iIndex) -> new CLUBComplete<>(uIndex, iIndex, hasRating, alpha1, alpha2);
                }
                case AlgorithmIdentifiers.CLUBERDOS:
                {
                    cursor++;
                    double alpha1 = Parsers.dp.parse(fullAlgorithm.get(cursor));
                    double alpha2 = Parsers.dp.parse(fullAlgorithm.get(cursor+1));
                    cursor+=2;

                    if(fullAlgorithm.size() == cursor)
                    {
                        hasRating = true;
                    }
                    else
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        cursor++;
                    }
                    return (uIndex, iIndex) -> new CLUBERdos<>(uIndex, iIndex, hasRating, alpha1, alpha2);
                }
                case AlgorithmIdentifiers.COFIBA:
                {
                    cursor++;
                    double alpha1 = Parsers.dp.parse(fullAlgorithm.get(cursor));
                    double alpha2 = Parsers.dp.parse(fullAlgorithm.get(cursor+1));
                    cursor+=2;

                    if(fullAlgorithm.size() == cursor)
                    {
                        hasRating = true;
                    }
                    else
                    {
                        hasRating = fullAlgorithm.get(cursor).equalsIgnoreCase("ignore");
                        cursor++;
                    }
                    return (uIndex, iIndex) -> new COFIBAErdos<>(uIndex, iIndex, hasRating, alpha1, alpha2);
                }
                default:
                {
                    return null;
                }
            }
        }

        return null;
    }


    /**
     * Adds a single algorithm to the selector.
     *
     * @param algorithm The String name of the algorithm.
     * @throws es.uam.eps.ir.knnbandit.selector.UnconfiguredException if the selector is not configured
     */
    public void addAlgorithm(String algorithm) throws UnconfiguredException
    {
        if (!this.configured)
        {
            throw new UnconfiguredException("AlgorithmSelector");
        }

        InteractiveRecommenderSupplier<U, I> rec = this.getAlgorithm(algorithm);
        if (rec != null)
        {
            this.recs.put(algorithm, rec);
        }
    }

    /**
     * Adds a set of algorithms.
     *
     * @param file File containing the configuration of the algorithms.
     * @throws IOException                                            if something fails while reading
     * @throws es.uam.eps.ir.knnbandit.selector.UnconfiguredException if the selector has not been configured
     */
    public void addFile(String file) throws IOException, UnconfiguredException
    {
        if (!this.configured)
        {
            throw new UnconfiguredException("AlgorithmSelector");
        }

        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file))))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                if(!line.startsWith("#"))
                    this.addAlgorithm(line);
            }
        }
    }

    /**
     * Obtains the selection of recommenders.
     *
     * @return the selection of recommenders.
     */
    public Map<String, InteractiveRecommenderSupplier<U, I>> getRecs()
    {
        return this.recs;
    }

    /**
     * Get an item bandit.
     *
     * @param split    A list containing the configuration.
     * @return the corresponding item bandit if everything is ok, null otherwise.
     */
    private BanditSupplier<U, I> getItemBandit(List<String> split)
    {
        BanditSupplier<U, I> ib;
        switch (split.get(0))
        {
            case ItemBanditIdentifiers.EGREEDY:
                double epsilon = Parsers.dp.parse(split.get(1));
                cursor += 2;
                EpsilonGreedyUpdateFunction updateFunc = this.getUpdateFunction(split.subList(2, split.size()));
                ib = (numItems) -> new EpsilonGreedyItemBandit<>(epsilon, numItems, updateFunc);
                break;
            case ItemBanditIdentifiers.UCB1:
                ib = UCB1ItemBandit::new;
                cursor++;
                break;
            case ItemBanditIdentifiers.UCB1TUNED:
                ib = UCB1TunedItemBandit::new;
                cursor++;
                break;
            case ItemBanditIdentifiers.THOMPSON:
                double alpha = Parsers.dp.parse(split.get(1));
                double beta = Parsers.dp.parse(split.get(2));
                ib = (numItems) -> new ThompsonSamplingItemBandit<>(numItems, alpha, beta);
                cursor += 3;
                break;
            case ItemBanditIdentifiers.DELAYTHOMPSON:
                alpha = Parsers.dp.parse(split.get(1));
                beta = Parsers.dp.parse(split.get(2));
                int delay = Parsers.ip.parse(split.get(3));
                ib = (numItems) -> new DelayedThompsonSamplingItemBandit<>(numItems, alpha, beta, delay);
                cursor += 4;
                break;
            case ItemBanditIdentifiers.ETGREEDY:
                alpha = Parsers.dp.parse(split.get(1));
                cursor += 2;
                updateFunc = this.getUpdateFunction(split.subList(2, split.size()));
                ib = (numItems) -> new EpsilonTGreedyItemBandit<>(alpha, numItems, updateFunc);
                break;
            default:
                cursor++;
                return null;
        }
        return ib;
    }

    /**
     * Obtains a function to update an Epsilon-greedy algorithm.
     *
     * @param split Strings containing the configuration.
     * @return the update function if everything is OK, null otherwise.
     */
    private EpsilonGreedyUpdateFunction getUpdateFunction(List<String> split)
    {
        switch (split.get(0))
        {
            case EpsilonGreedyUpdateFunctionIdentifiers.STATIONARY:
                cursor++;
                return EpsilonGreedyUpdateFunctions.stationary();
            case EpsilonGreedyUpdateFunctionIdentifiers.NONSTATIONARY:
                cursor++;
                cursor++;
                return EpsilonGreedyUpdateFunctions.nonStationary(Parsers.dp.parse(split.get(1)));
            case EpsilonGreedyUpdateFunctionIdentifiers.USEALL:
                cursor++;
                return EpsilonGreedyUpdateFunctions.useall();
            case EpsilonGreedyUpdateFunctionIdentifiers.COUNT:
                cursor++;
                return EpsilonGreedyUpdateFunctions.count();
            default:
                cursor++;
                return null;
        }
    }

    /**
     * Obtains a MF Factorizer.
     *
     * @param split Strings containing the configuration.
     * @return the factorizer if everything is OK, null otherwise.
     */
    private Factorizer<U, I> getFactorizer(List<String> split)
    {
        cursor++;
        Factorizer<U, I> factorizer;
        switch (split.get(0))
        {
            case FactorizerIdentifiers.IMF:
                double alphaHKV = Parsers.dp.parse(split.get(1));
                double lambdaHKV = Parsers.dp.parse(split.get(2));
                int numIterHKV = Parsers.ip.parse(split.get(3));
                cursor += 3;
                DoubleUnaryOperator confidence = (double x) -> 1 + alphaHKV * x;
                factorizer = new HKVFactorizer<>(lambdaHKV, confidence, numIterHKV);
                break;
            case FactorizerIdentifiers.FASTIMF:
                double alphaPZT = Parsers.dp.parse(split.get(1));
                double lambdaPZT = Parsers.dp.parse(split.get(2));
                int numIterpzt = Parsers.ip.parse(split.get(3));
                boolean usesZeroes = split.get(4).equalsIgnoreCase("true");
                cursor += 4;
                confidence = (double x) -> 1 + alphaPZT * x;
                factorizer = new PZTFactorizer<>(lambdaPZT, confidence, numIterpzt, usesZeroes);
                break;
            case FactorizerIdentifiers.PLSA:
                int numIterPLSA = Parsers.ip.parse(split.get(1));
                cursor++;
                factorizer = new PLSAFactorizer<>(numIterPLSA);
                break;
            default:
                return null;
        }

        return factorizer;
    }

    /**
     * Obtains the corresponding bandit algorithm based on PMF (see Interactive Collaborative Filtering paper)
     *
     * @param split   the list with the parameters.
     * @param k       the number of latent factors
     * @param lambdaP the prior standard deviation for the user factors.
     * @param lambdaQ the prior standard deviation for the item factors.
     * @param stdev   the prior standard deviation for the ratings.
     * @param numIter the number of iterations.
     * @return the algorithm if everything is fine, null otherwise.
     */
    private InteractiveRecommenderSupplier<U, I> getPMFBanditAlgorithm(List<String> split, int k, double lambdaP, double lambdaQ, double stdev, int numIter)
    {
        cursor++;

        boolean hasRating;
        switch (split.get(0))
        {
            case PMFBanditIdentifiers.EGREEDY:
                double epsilon = Parsers.dp.parse(split.get(1));
                cursor++;
                if (split.size() == 2)
                {
                    hasRating = true;
                }
                else
                {
                    hasRating = split.get(2).equalsIgnoreCase("ignore");
                    cursor++;
                }

                return (uIndex, iIndex) -> new EpsilonGreedyInteractivePMFRecommender<>(uIndex, iIndex, hasRating, k, lambdaP, lambdaQ, stdev, numIter, epsilon);
            case PMFBanditIdentifiers.UCB:
                double alpha = Parsers.dp.parse(split.get(1));
                cursor++;
                if (split.size() == 2)
                {
                    hasRating = true;
                }
                else
                {
                    hasRating = split.get(2).equalsIgnoreCase("ignore");
                    cursor++;
                }

                return (uIndex, iIndex) -> new LinearUCBPMFRecommenderInteractive<>(uIndex, iIndex, hasRating, k, lambdaP, lambdaQ, stdev, numIter, alpha);

            case PMFBanditIdentifiers.GENERALIZEDUCB:
                alpha = Parsers.dp.parse(split.get(1));
                cursor++;
                if (split.size() == 2)
                {
                    hasRating = true;
                }
                else
                {
                    hasRating = split.get(2).equalsIgnoreCase("ignore");
                    cursor++;
                }

                return (uIndex, iIndex) -> new GeneralizedLinearUCBPMFRecommenderInteractive<>(uIndex, iIndex, hasRating, k, lambdaP, lambdaQ, stdev, numIter, alpha);

            case PMFBanditIdentifiers.THOMPSON:
                if (split.size() == 1)
                {
                    hasRating = true;
                }
                else
                {
                    hasRating = split.get(2).equalsIgnoreCase("ignore");
                    cursor++;
                }

                return (uIndex, iIndex) -> new ThompsonSamplingInteractivePMFRecommender<>(uIndex, iIndex, hasRating, k, lambdaP, lambdaQ, stdev, numIter);
            default:
                return null;
        }
    }
}
