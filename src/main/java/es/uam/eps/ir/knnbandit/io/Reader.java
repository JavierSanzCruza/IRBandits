package es.uam.eps.ir.knnbandit.io;

import org.jooq.lambda.tuple.Tuple2;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Class for reading
 */
public class Reader
{
    /**
     * Given a file, reads the lists of user-item pairs.
     *
     * @param file      name of the file
     * @param delimiter field separator
     * @return the list of user-item pairs in the file.
     */
    public List<Tuple2<Integer, Integer>> read(String file, String delimiter, boolean header)
    {
        List<Tuple2<Integer, Integer>> list = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file))))
        {
            String line;
            if (header)
            {
                String headerLine = br.readLine();
            }

            while ((line = br.readLine()) != null)
            {
                Tuple2<Integer, Integer> pair = this.parseLine(line, delimiter);
                if (pair != null)
                {
                    list.add(pair);
                }
            }
            return list;
        }
        catch (IOException ioe)
        {
            System.err.println("Something failed while reading the file");
            return null;
        }
    }

    /**
     * Parses one line.
     *
     * @param line the line.
     * @return the user-item pair if everything is OK, null otherwise
     */
    private Tuple2<Integer, Integer> parseLine(String line, String delimiter)
    {
        String[] split = line.split(delimiter);
        if (split.length < 3)
        {
            return null;
        }
        int user = Parsers.ip.parse(split[1]);
        int item = Parsers.ip.parse(split[2]);
        return new Tuple2<>(user, item);
    }
}
