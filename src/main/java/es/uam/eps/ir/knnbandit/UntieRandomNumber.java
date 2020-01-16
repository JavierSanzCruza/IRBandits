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
import java.util.Random;

/**
 * Value for a random number seed.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class UntieRandomNumber
{
    /**
     * The configured seed.
     */
    public static int RNG = 0;

    /**
     * Configures the random number seed.
     * @param resume true if we want to use a previous seed.
     * @param route the route from which to read the previous seed / to write the new seed.
     */
    public static void configure(boolean resume, String route) throws IOException
    {
        if (resume)
        {
            File f = new File(route + "rngseed");
            if (f.exists())
            {
                try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f))))
                {
                    UntieRandomNumber.RNG = Parsers.ip.parse(br.readLine());
                }
            }
            else
            {
                Random rng = new Random();
                UntieRandomNumber.RNG = rng.nextInt();
            }
        }
        else
        {
            Random rng = new Random();
            UntieRandomNumber.RNG = rng.nextInt();
        }

        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(route + "rngseed"))))
        {
            bw.write("" + UntieRandomNumber.RNG);
        }
    }

}
