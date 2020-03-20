/*
 * Copyright (C) 2019 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0.
 *
 */
package es.uam.eps.ir.knnbandit;

import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Value for a random number seed.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class UntieRandomNumber
{
    public static List<Integer> rngSeeds;
    public static int RNG = 0;
    /**
     * Configures the random number seed.
     * @param resume true if we want to use a previous seed.
     * @param route the route from which to read the previous seed / to write the new seed.
     */
    public static void configure(boolean resume, String route) throws IOException
    {
        UntieRandomNumber.configure(resume, route, 1);
    }


    /**
     * Configures a list of random number seeds.
     * @param resume true if we want to use a previous seed.
     * @param route the route from which to read the previous seed / to write the new seed.
     */
    public static void configure(boolean resume, String route, int k) throws IOException
    {
        rngSeeds = new ArrayList<>();
        if(resume)
        {
            File f = new File(route + "rngseedlist");
            if(f.exists())
            {
                try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f))))
                {
                    String line;
                    while((line = br.readLine()) != null)
                    {
                        if(!line.equals(""))
                            rngSeeds.add(Integer.parseInt(line));
                    }
                }
            }
        }

        int remaining = k - rngSeeds.size();
        Random rng = new Random();
        for(int i = 0; i < remaining; ++i)
        {
            rngSeeds.add(rng.nextInt());
        }

        UntieRandomNumber.RNG = rngSeeds.get(0);

        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(route + "rngseedlist"))))
        {
            for(int seed : rngSeeds)
            {
                bw.write(""+ seed +"\n");
            }
        }
    }

}
