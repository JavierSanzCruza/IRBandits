package es.uam.eps.ir.knnbandit.data.datasets;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Definition of a simple offline dataset.
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class GeneralOfflineDataset<U,I> implements OfflineDataset<U,I>
{
    /**
     * User index. Relates integer identifiers to real user values.
     */
    protected final FastUpdateableUserIndex<U> userIndex;
    /**
     * Item index. Relates integer identifiers to real item values.
     */
    protected final FastUpdateableItemIndex<I> itemIndex;
    /**
     * Rating data.
     */
    protected final SimpleFastPreferenceData<U, I> prefData;
    /**
     * Number of relevant ratings.
     */
    protected final int numRel;
    /**
     * Predicate for measuring the relevance.
     */
    DoublePredicate relevance;

    /**
     * Constructor.
     *
     * @param uIndex   User index.
     * @param iIndex   Item index.
     * @param prefData Preference data.
     * @param numRel   Number of relevant (user, item) pairs.
     */
    protected GeneralOfflineDataset(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, int numRel, DoublePredicate relevance)
    {
        this.userIndex = uIndex;
        this.itemIndex = iIndex;
        this.prefData = prefData;
        this.numRel = numRel;
        this.relevance = relevance;
    }

    @Override
    public int item2iidx(I i)
    {
        return itemIndex.item2iidx(i);
    }

    @Override
    public I iidx2item(int i)
    {
        return itemIndex.iidx2item(i);
    }

    @Override
    public int numItems()
    {
        return itemIndex.numItems();
    }

    @Override
    public int user2uidx(U u)
    {
        return userIndex.user2uidx(u);
    }

    @Override
    public U uidx2user(int i)
    {
        return userIndex.uidx2user(i);
    }

    @Override
    public int numUsers()
    {
        return userIndex.numUsers();
    }



    /**
     * Finds a textual representation of the dataset.
     * @return the textual representation of the dataset.
     */
    public String toString()
    {
        return "Users: " +
                this.numUsers() +
                "\nItems: " +
                this.numItems() +
                "\nNum. ratings: " +
                this.prefData.numPreferences() +
                "\nNum. relevant: " +
                this.getNumRel();
    }

    @Override
    public int getNumRel()
    {
        return this.numRel;
    }

    @Override
    public int getNumRel(List<Tuple2<Integer, Integer>> list)
    {
        return list.stream().mapToInt(t ->
        {
            if (prefData.numItems(t.v1) > 0 && prefData.numUsers(t.v2) > 0)
            {
                Optional<IdxPref> opt = prefData.getPreference(t.v1, t.v2);
                if (opt.isPresent() && relevance.test(opt.get().v2))
                {
                    return 1;
                }
            }
            return 0;
        }).sum();
    }

    @Override
    public Stream<U> getUsersWithPreferences()
    {
        return this.prefData.getUsersWithPreferences();
    }

    @Override
    public IntStream getUidxWithPreferences()
    {
        return this.prefData.getUidxWithPreferences();
    }

    @Override
    public Optional<Double> getPreference(U u, I i)
    {
        Optional<Double> opt;
        if (prefData.numItems(u) > 0 && prefData.numUsers(i) > 0)
        {
            Optional<? extends IdPref<I>> optional = prefData.getPreference(u, i);
            if(optional.isPresent())
            {
                return Optional.of(optional.get().v2);
            }
        }
        return Optional.empty();
    }

    @Override
    public Optional<Double> getPreference(int uidx, int iidx)
    {
        Optional<Double> opt;
        if (prefData.numItems(uidx) > 0 && prefData.numUsers(iidx) > 0)
        {
            Optional<IdxPref> optional = prefData.getPreference(uidx, iidx);
            if(optional.isPresent())
            {
                return Optional.of(optional.get().v2);
            }
        }
        return Optional.empty();
    }

    @Override
    public boolean isRelevant(double value)
    {
        return this.relevance.test(value);
    }

    @Override
    public Stream<IdxPref> getUidxPreferences(int uidx)
    {
        return this.prefData.getUidxPreferences(uidx);
    }

    /**
     * Loads a dataset.
     *
     * @param filename       the name of the file where the dataset is.
     * @param uParser        parser for the users.
     * @param iParser        parser for the items.
     * @param separator      separator for the different fields in the file.
     * @param weightFunction function for determining the rating value.
     * @param relevance      function that determines if a (user, item) pair is relevant or not.
     * @param <U>            type of the users.
     * @param <I>            type of the items.
     * @return the dataset.
     * @throws IOException if something fails while reading the dataset file.
     */
    public static <U, I> GeneralOfflineDataset<U, I> load(String filename, Parser<U> uParser, Parser<I> iParser, String separator, DoubleUnaryOperator weightFunction, DoublePredicate relevance) throws IOException
    {
        List<Tuple3<U, I, Double>> triplets = new ArrayList<>();
        Set<U> users = new HashSet<>();
        Set<I> items = new HashSet<>();

        int numrel = 0;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filename))))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] split = line.split(separator);
                U user = uParser.parse(split[0]);
                I item = iParser.parse(split[1]);
                double val = Parsers.dp.parse(split[2]);

                users.add(user);
                items.add(item);

                double rating = weightFunction.applyAsDouble(val);
                if (relevance.test(rating))
                {
                    numrel++;
                }

                triplets.add(new Tuple3<>(user, item, rating));
            }
        }

        // Create the data.
        FastUpdateableUserIndex<U> uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        FastUpdateableItemIndex<I> iIndex = SimpleFastUpdateableItemIndex.load(items.stream());
        SimpleFastPreferenceData<U, I> prefData = SimpleFastPreferenceData.load(triplets.stream(), uIndex, iIndex);

        return new GeneralOfflineDataset<>(uIndex, iIndex, prefData, numrel, relevance);
    }

    /**
     * Loads a dataset from another dataset.
     * @param dataset the original dataset.
     * @param list a list of (user,item) interactions.
     * @param <U> type of the users.
     * @param <I> type of the items.
     * @return the new dataset.
     */
    public static <U, I> GeneralOfflineDataset<U, I> load(GeneralOfflineDataset<U, I> dataset, List<Tuple2<Integer, Integer>> list)
    {
        List<Tuple3<U, I, Double>> triplets = new ArrayList<>();
        FastUpdateableUserIndex<U> userIndex = dataset.userIndex;
        FastUpdateableItemIndex<I> itemIndex = dataset.itemIndex;

        SimpleFastPreferenceData<U, I> datasetPrefData = dataset.prefData;

        int numrel = 0;

        for (Tuple2<Integer, Integer> tuple : list)
        {
            int uidx = tuple.v1;
            int iidx = tuple.v2;
            U u = userIndex.uidx2user(uidx);
            I i = itemIndex.iidx2item(iidx);

            if (datasetPrefData.numItems(uidx) > 0 && datasetPrefData.numUsers(iidx) > 0)
            {
                Optional<IdxPref> pref = datasetPrefData.getPreference(uidx, iidx);
                if (pref.isPresent())
                {
                    double value = pref.get().v2;
                    Tuple3<U, I, Double> t = new Tuple3<>(u, i, value);
                    triplets.add(t);

                    if (dataset.relevance.test(value))
                    {
                        numrel++;
                    }
                }
            }
        }

        SimpleFastPreferenceData<U, I> prefData = SimpleFastPreferenceData.load(triplets.stream(), userIndex, itemIndex);
        return new GeneralOfflineDataset<>(userIndex, itemIndex, prefData, numrel, dataset.relevance);
    }
}
