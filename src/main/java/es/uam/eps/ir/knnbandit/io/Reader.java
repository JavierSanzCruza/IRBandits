/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.io;

import es.uam.eps.ir.knnbandit.utils.Pair;
import org.jooq.lambda.tuple.Tuple2;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Class for reading a recommendation loop
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
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
    public List<Pair<Integer>> read(String file, String delimiter, boolean header)
    {
        List<Pair<Integer>> list = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file))))
        {
            String line;
            if (header)
            {
                String headerLine = br.readLine();
            }

            while ((line = br.readLine()) != null)
            {
                Pair<Integer> pair = this.parseLine(line, delimiter);
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
    private Pair<Integer> parseLine(String line, String delimiter)
    {
        String[] split = line.split(delimiter);
        if (split.length < 3)
        {
            return null;
        }
        int user = Parsers.ip.parse(split[1]);
        int item = Parsers.ip.parse(split[2]);
        return new Pair<>(user, item);
    }
}
