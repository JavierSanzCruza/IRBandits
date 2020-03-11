package es.uam.eps.ir.knnbandit.main;

import java.io.*;

public class ParallelConfigurator
{
    /**
     * Given algorithm rankings for validation splits, obtains the configuration files to run all of them in parallel
     * @param args Execution arguments:
     *             <ul>
     *              <li><b>Directory:</b> The base directory of the validation, where the validation folders are.</li>
     *              <li><b>Num. partitions: </b> The number of partitions to consider.</li>
     *              <li><b>Ranking file:</b> The name of the file containing the ranking.</li>
     *              <li><b>Header:</b> True if the ranking file contains a header, false otherwise.</li>
     *              <li><b>Output:</b> Name of the output configuration file</li>
     *             </ul>
     * @throws IOException
     */
    public static void main(String args[]) throws IOException
    {
        if(args.length < 2)
        {
            System.err.println("ERROR: Invalid arguments");
            System.err.println("Arguments:");
            System.err.println("\tDirectory: directory where the ranking files are");
            System.err.println("\tNum. partitions: number of partitions");
            System.err.println("\tRanking file: name of the ranking file");
            System.err.println("\tHeader: true if the ranking file has a header, false otherwise");
            System.err.println("\tOutput: name of the output file");
            return;
        }

        String directory = args[0];
        int numPartitions = Integer.valueOf(args[1]);
        String rankingFile = args[2];
        boolean header = args[3].equalsIgnoreCase("true");
        String output = args[4];

        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output))))
        {
            for(int i = 0; i < numPartitions; ++i)
            {
                try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(directory + i + File.separator + rankingFile))))
                {
                    String line;
                    if(header) line = br.readLine();
                    line = br.readLine();

                    String[] split = line.split("\t");

                    if(split[0].endsWith(".txt"))
                    {
                        String[] auxSplit = split[0].split("\\.");
                        int length = auxSplit.length;
                        String text = "";
                        for(int j = 0; j < length - 1; ++j)
                        {
                            text += auxSplit[i];
                        }
                        bw.write(text + "\n");
                    }
                    else
                    {
                        bw.write(split[0]+"\n");
                    }
                }
            }
        }
    }

}
