package es.uam.eps.ir.knnbandit.data.datasets;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;

import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.userknowledge.fast.SimpleFastUserKnowledgePreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.jooq.lambda.tuple.Tuple4;
import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

/**
 * A dataset.
 *
 * @param <U> type of the users.
 * @param <I> type of the items.
 */
public class Dataset<U, I>
{
    protected final FastUpdateableUserIndex<U> uIndex;
    protected final FastUpdateableItemIndex<I> iIndex;
    protected final SimpleFastPreferenceData<U, I> prefData;
    protected final int numRel;
    DoublePredicate relevance;

    /**
     * Constructor.
     *
     * @param uIndex   User index.
     * @param iIndex   Item index.
     * @param prefData Preference data.
     * @param numRel   Number of relevant (user, item) pairs.
     */
    protected Dataset(FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, SimpleFastPreferenceData<U, I> prefData, int numRel, DoublePredicate relevance)
    {
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.prefData = prefData;
        this.numRel = numRel;
        this.relevance = relevance;
    }

    /**
     * Obtains the user index of the dataset.
     *
     * @return the user index of the dataset.
     */
    public FastUpdateableUserIndex<U> getUserIndex()
    {
        return uIndex;
    }

    /**
     * Obtains the item index of the dataset.
     *
     * @return the item index of the dataset.
     */
    public FastUpdateableItemIndex<I> getItemIndex()
    {
        return iIndex;
    }

    /**
     * Obtains the preference data.
     *
     * @return the preference data of the dataset.
     */
    public SimpleFastPreferenceData<U, I> getPrefData()
    {
        return prefData;
    }

    /**
     * Get the number of relevant (user, item) pairs.
     *
     * @return the number of relevant (user, item) pairs.
     */
    public int getNumRel()
    {
        return numRel;
    }

    /**
     * Given a list of (uidx, iidx) pairs, finds how many relevant pairs there are.
     *
     * @param list the list of (uidx, iidx)
     * @return the count of how many relevant (uidx, iidx) pairs appear in the list.
     */
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

    public int numUsers()
    {
        return this.uIndex.numUsers();
    }

    public int numItems()
    {
        return this.iIndex.numItems();
    }

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
    public static <U, I> Dataset<U, I> load(String filename, Parser<U> uParser, Parser<I> iParser, String separator, DoubleUnaryOperator weightFunction, DoublePredicate relevance) throws IOException
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

        return new Dataset<>(uIndex, iIndex, prefData, numrel, relevance);
    }

    public static <U, I> Dataset<U, I> load(Dataset<U, I> dataset, List<Tuple2<Integer, Integer>> list)
    {
        List<Tuple3<U, I, Double>> triplets = new ArrayList<>();
        FastUpdateableUserIndex<U> userIndex = dataset.getUserIndex();
        FastUpdateableItemIndex<I> itemIndex = dataset.getItemIndex();

        SimpleFastPreferenceData<U, I> datasetPrefData = dataset.getPrefData();

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
        return new Dataset<>(userIndex, itemIndex, prefData, numrel, dataset.relevance);
    }


}
