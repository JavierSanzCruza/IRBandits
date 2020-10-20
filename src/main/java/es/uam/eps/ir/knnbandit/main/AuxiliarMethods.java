package es.uam.eps.ir.knnbandit.main;

import es.uam.eps.ir.knnbandit.recommendation.loop.FastRecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.loop.RecommendationLoop;
import es.uam.eps.ir.knnbandit.utils.FastRating;
import es.uam.eps.ir.knnbandit.utils.Rating;
import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.FileInputStream;

import es.uam.eps.ir.knnbandit.io.Writer;

import java.util.*;

/**
 * Auxiliar methods for executing bandit algorithms.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 */
public class AuxiliarMethods
{
    /**
     * Retrieves previous iterations of an execution.
     *
     * @param filename the name of the file.
     * @return a list containing the retrieved (uidx, iidx, time) triplets.
     * @throws IOException if something fails while reading the file.
     */
    public static List<Tuple3<Integer,Integer,Long>> retrievePreviousIterations(String filename) throws IOException
    {
        // Initialize the list
        List<Tuple3<Integer,Integer,Long>> recovered = new ArrayList<>();

        File f = new File(filename);
        if (f.exists() && !f.isDirectory()) // if the file exists, then recover the triplets:
        {
            // Once we know that the file exists, we open it.
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filename))))
            {
                String line = br.readLine();
                int len;
                if (line != null)
                {
                    String[] split = line.split("\t");
                    len = split.length;

                    // Read each line
                    while ((line = br.readLine()) != null)
                    {
                        split = line.split("\t");
                        if (split.length < len)
                        {
                            break;
                        }

                        // Obtain the triplet
                        int uidx = Parsers.ip.parse(split[1]);
                        int iidx = Parsers.ip.parse(split[2]);
                        long time = Parsers.lp.parse(split[len - 1]);

                        // Add it to the recovered list.
                        recovered.add(new Tuple3<>(uidx, iidx, time));
                    }
                }
            }
        }

        return recovered;
    }

    /**
     * Given the list of recovered triplets, updates the recommendation loop.
     *
     * @param loop      the recommendation loop.
     * @param recovered the list of recovered (uidx, iidx, time) triplets.
     * @param writer    a writer for storing the recommendation loop in a file.
     * @param interval  the interval between different data points.
     * @param <U>       type of the users
     * @param <I>       type of the items.
     * @return a map containing the values of the metrics in certain time points.
     * @throws IOException if something fails while writing.
     */
    public static <U, I> Map<String, List<Double>> updateWithPrevious(FastRecommendationLoop<U, I> loop, List<Tuple3<Integer,Integer,Long>> recovered, Writer writer, int interval) throws IOException
    {
        List<String> metricNames = loop.getMetrics();
        Map<String, List<Double>> metricValues = new HashMap<>();

        for(String name : metricNames)
        {
            metricValues.put(name, new ArrayList<>());
        }

        for(Tuple3<Integer,Integer,Long> triplet : recovered)
        {
            int uidx = triplet.v1();
            int iidx = triplet.v2();
            long time = triplet.v3();

            loop.fastUpdate(uidx, iidx);
            int iter = loop.getCurrentIter();

            Map<String, Double> metricVals = loop.getMetricValues();
            writer.writeLine(iter, uidx, iidx, metricVals, time);

            if(iter % interval == 0)
            {
                for(String name : metricNames)
                {
                    double value = metricVals.get(name);
                    metricValues.get(name).add(value);
                }
            }
        }

        return metricValues;
    }

    /**
     * Execute the remaining loop
     *
     * @param loop         the recommendation loop.
     * @param writer       the writer.
     * @param interval     the interval.
     * @param metricValues the list of metric values.
     * @param <U>          type of the users.
     * @param <I>          type of the items.
     * @return the number of iterations for finishing the loop.
     */
    public static <U, I> int executeRemaining(FastRecommendationLoop<U, I> loop, Writer writer, int interval, Map<String, List<Double>> metricValues) throws IOException
    {
        List<String> metricNames = loop.getMetrics();

        // Apply it until the end.
        while (!loop.hasEnded())
        {
            long aa = System.currentTimeMillis();
            FastRating rating = loop.fastNextIteration();
            long bb = System.currentTimeMillis();

            if(rating == null) break; // Everything has finished

            int uidx = rating.uidx();
            int iidx = rating.iidx();
            long time = bb - aa;
            int numIter = loop.getCurrentIter();
            Map<String, Double> metrics = loop.getMetricValues();

            writer.writeLine(numIter, uidx, iidx, metrics, time);

            if (numIter % interval == 0)
            {
                for (String name : metricNames)
                {
                    double value = metrics.get(name);
                    metricValues.get(name).add(value);
                }
            }
        }

        // Store the value of the last iteration.
        int numIter = loop.getCurrentIter();
        if (numIter % interval != 0)
        {
            Map<String, Double> metrics = loop.getMetricValues();
            for (String name : metricNames)
            {
                double value = metrics.get(name);
                metricValues.get(name).add(value);
            }
        }

        return numIter;
    }
}
