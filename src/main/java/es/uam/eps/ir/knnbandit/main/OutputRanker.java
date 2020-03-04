package es.uam.eps.ir.knnbandit.main;

import org.ranksys.core.util.tuples.Tuple2od;

import java.io.*;
import java.util.PriorityQueue;

public class OutputRanker
{
    /**
     * Obtains the rankings for different variants of algorithms
     * @param args Execution arguments
     *             <ul>
     *              <li><b>Input directory:</b> Directory containing the recommendation files</li>
     *              <li><b>Point:</b> The time point we want to compute the ranking for</li>
     *              <li><b>Algorithm list:</b> Comma separated algorithm list</li>
     *              <li><b>Header:</b> true if the results files contain headers, false otherwise</li>
     *              <li><b>Output directory:</b> directory in which to store the rankings.</li>
     *             </ul>
     * @throws IOException if something fails while writing.
     */
    public static void main(String[] args) throws IOException
    {
        // Read the arguments
        if (args.length < 5)
        {
            System.err.println("ERROR: Invalid arguments");
            System.err.println("Input directory: directory containing the recommendation files");
            System.err.println("Point: the time point to consider");
            System.err.println("Algorithm list: the list of algorithms (comma separated)");
            System.err.println("Header: true if the files have a header, false otherwise");
            System.err.println("Output directory: output directory");
        }

        String inputDirectory = args[0];
        int timePoint = Integer.valueOf(args[1]);
        String algorithmList = args[2];
        boolean header = args[3].equalsIgnoreCase("true");
        String outputDir = args[4];

        // Obtain the list of algorithms
        String[] algorithms = algorithmList.split(",");

        // Check whether the input directory is a directory or not.
        int numRecs;
        File directory = new File(inputDirectory);
        if(directory.isDirectory())
        {
            numRecs = directory.list().length;
        }
        else
        {
            System.err.println("ERROR: " + directory + " is not a directory");
            return;
        }

        // For each algorithm, obtain the rankings.
        for(String algorithm : algorithms)
        {
            System.out.println("Started " + algorithm);
            long a = System.currentTimeMillis();
            // Declare the ranking
            PriorityQueue<Tuple2od<String>> queue = new PriorityQueue<>(numRecs, (x, y) -> (int) Math.signum(y.v2() - x.v2()));

            // Obtain the result files for the algorithm
            File[] files = directory.listFiles((dir, name) -> name.startsWith(algorithm));
            for(File file : files)
            {
                // Obtain the recall result.
                try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file))))
                {
                    // iter, user, item, recall
                    String line;
                    if(header)
                    {
                        line = br.readLine();
                    }

                    double value = 0.0;
                    int i = 0;
                    do
                    {
                        line = br.readLine();
                        i++;
                    }
                    while(i < timePoint);

                    String[] split = line.split("\t");
                    value = Double.valueOf(split[3]);
                    queue.add(new Tuple2od<>(file.getName(), value));
                }
            }

            long b = System.currentTimeMillis();
            System.out.println("Ranked " + algorithm + "(" + (b-a) + " ms.)");

            // Write the ranking.
            try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDir + "algorithm-" + algorithm + ".txt"))))
            {
                bw.write("Algorithm\trecall@" + timePoint);
                while(!queue.isEmpty())
                {
                    Tuple2od<String> element = queue.poll();

                    bw.write("\n" + element.v1.split("\\.")[0] + "\t" + element.v2);
                }
            }

            b = System.currentTimeMillis();
            System.out.println("Finished " + algorithm + "(" + (b-a) + " ms.)");
        }
    }
}
